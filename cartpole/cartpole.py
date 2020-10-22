'''
adopted from https://github.com/sanjitjain2/q-learning-for-cartpole/blob/master/qlearning.py
https://mc.ai/openai-gyms-cart-pole-balancing-using-q-learning/
'''

import gym
import numpy as np
import math
import time
from collections import deque

class CartPole():
    def __init__(self, buckets=(1, 1, 6, 12,), n_iter=100, n_episodes=1000, n_win_ticks=195, min_alpha=0.1, min_epsilon=0.1, gamma=0.999, ada_divisor=200, max_env_steps=None, monitor=False, avg = False):
        self.buckets = buckets # down-scaling feature space to discrete rangen
        self.n_iter = n_iter
        self.n_episodes = n_episodes # training episodes
        self.n_win_ticks = n_win_ticks # average ticks over 100 episodes required for wi
        self.min_alpha = min_alpha # learning rate
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount factor
        self.ada_divisor = ada_divisor # only for development purposes
        self.test_after = 50
        self.test_episodes = 1000
        self.per_episode = 210

        ##policy can be 'Q' or 'D-Q'
        self.policy = 'Q'
        ##twofold = 0 : original step size; twofold = 1: step size *= 2
        self.twofold = 0
        ##average: =True, use averaged estimator; =False, only use Qa
        self.average = False

        self.rew_arr = np.zeros((int(self.n_episodes / self.test_after), self.n_iter))
        self.terminate_episodes = np.zeros(self.n_iter)

        self.env = gym.make('CartPole-v0')
        ###for reproduction
        self.env.seed(2020)
        self.env.action_space.seed(2020)

        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        if monitor: self.env = gym.wrappers.Monitor(self.env, 'tmp/cartpole-1', force=True) # record results for upload

    def initialize_Q(self):
        # initialising Q-table
        self.Qa = np.zeros(self.buckets + (self.env.action_space.n,))
        self.Qb = np.zeros(self.buckets + (self.env.action_space.n,))

    def test_performance(self):
        #calculate average reward of current estimators based on multiple episodes
        reward_q = deque(maxlen = self.test_episodes)

        for e in range(self.test_episodes):
            current_state = self.discretize(self.env.reset())
            ep_reward = 0

            for j in range(self.per_episode):
                # Choose greedy action
                curQ = self.Qa
                if (self.policy == 'D-Q') & (self.average == True):
                    curQ = (self.Qa + self.Qb) / 2
                action = self.choose_action_greedy(curQ, current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)
                # Update Q-Table
                current_state = new_state
                ep_reward += reward
                if done:
                    break
            reward_q.append(ep_reward)

        return np.mean(reward_q)

    # Discretizing input space to make Q-table and to reduce dimmensionality
    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)


    # Choosing action greedily
    def choose_action_greedy(self, Q, state):
        return self.choose_action(Q, state, epsilon = 0)

    # Choosing action using \epsilon-greedy
    def choose_action(self, Q, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(Q[state])

    # Updating Q-value of state-action pair based on the update equation
    def update_q(self, state_old, action, reward, state_new, alpha):

        if self.policy == 'Q': #Q-learning
            self.Qa[state_old][action] += alpha * (reward + self.gamma * np.max(self.Qa[state_new]) - self.Qa[state_old][action])

        if self.policy == 'D-Q':
            if np.random.randint(2) < 1: #update Qa
                self.Qa[state_old][action] += alpha * (reward + self.gamma * self.Qb[state_new][np.argmax(self.Qa[state_new])] - self.Qa[state_old][action])
            else:
                self.Qb[state_old][action] += alpha * (reward + self.gamma * self.Qa[state_new][np.argmax(self.Qb[state_new])] - self.Qb[state_old][action])

    # Adaptive learning of Exploration Rate
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1 - math.log10((t + 1) / self.ada_divisor)))

    # Adaptive learning of Learning Rate
    def get_alpha(self, t):
        return 40 / (t+100)

    def run(self):
        reward_data = []
        x_dd = []

        filename = "Reward-"+self.policy
        if self.twofold == 1:
            filename = filename + "-twice"
        if self.average == True:
            filename = filename + "-average"

        print("Running: " + filename)

        for j in range(self.n_iter):

            start_time = time.time()
            self.initialize_Q()

            self.terminate_episodes[j] = self.n_episodes + 1

            for e in range(self.n_episodes):
                current_state = self.discretize(self.env.reset())

                # Get adaptive learning alpha and epsilon decayed over time
                epsilon = self.get_epsilon(e)
                alpha = min(1,(self.twofold + 1) * self.get_alpha(e))

                done = False
                i = 0
                rw = 0

                while not done:
                    # Choose action according to greedy policy and take it
                    curQ = self.Qa
                    if (self.policy == 'D-Q') & (self.average == True):
                        curQ = (self.Qa + self.Qb) / 2
                    action = self.choose_action(curQ, current_state, epsilon)
                    obs, reward, done, _ = self.env.step(action)
                    rw = rw + reward
                    new_state = self.discretize(obs)
                    # Update Q-Table
                    self.update_q(current_state, action, reward, new_state, alpha)
                    current_state = new_state
                    i += 1

                #calculate performance 

                if (e + 1) % self.test_after == 0:
                    mean_reward = self.test_performance()
                    reward_data.append(mean_reward) 
                    self.rew_arr[(e+1)//self.test_after-1, j] = mean_reward #save reward
                    x_dd.append((e + 1))

                    if mean_reward >= 195.0: #Hit time: the #of episodes to get a good policy
                        print(self.Qa)
                        if self.terminate_episodes[j] > self.n_episodes:
                            self.terminate_episodes[j] = e
                        print("Termination: Average reward is :", mean_reward)
                        print("Number of episodes to termination:", e)

                    print("Iteration, Episode", j, e)
                    print("Running: Average reward for latest estimate is :", mean_reward)

            print(self.terminate_episodes)
            if (j + 1) % 50 == 0: #save file 
                print("Save at iteration: ", j)
                file = open(filename, "wb")
                np.save(file, self.rew_arr)
                file.close()

            total_time = time.time() - start_time
            print("Time taken - %f seconds" % total_time)

        print(self.terminate_episodes)


if __name__ == "__main__":

    #reproduction
    np.random.seed(2020)
    # Make an instance of CartPole class 
    solver = CartPole()
    solver.run()