'''
Sutton & Barto's example
'''

import numpy as np
import math
import time
from collections import deque

class Bias():
    def __init__(self, n_iter=1000, n_episodes=200):
        self.N = 10 #State A & State B (Many) in Figure 6.5
        self.A = 2 #0: Right; 1: Left;
        self.n_iter = n_iter
        self.gamma = 0.999
        self.n_episodes = n_episodes # training episodes

        ##policy can be 'Q' or 'D-Q'
        self.policy = 'Q'
        ##twofold = 0 : original step size; twofold = 1: step size *= 2
        self.twofold = 0
        ##average: =True, use averaged estimator; =False, only use Qa
        self.average = False

        self.rew_arr = np.zeros((self.n_episodes, self.n_iter))
        # initialising Q-table
        self.Qa = np.zeros((self.N, self.A))
        self.Qb = np.zeros((self.N, self.A))

    def initialize_Q(self):
        self.Qa = np.zeros((self.N, self.A))
        self.Qb = np.zeros((self.N, self.A))

    def obtain_GoLeft(self):
        curQ = self.Qa
        QvalB = self.Qb
        if (self.policy == 'D-Q') & (self.average == True):
            curQ = (curQ + QvalB) / 2
        if curQ[0,1] > curQ[0,0]:
            return 1
        if curQ[0,1] == curQ[0,0]:
            return 0.5
        return 0

    # Choosing action greedily
    def bestAction(self, state, Q):
        return self.choose_action(Q, state, epsilon = 0)

    def obtain_reward(selfs, state, action):
        if state == 0:  # in State A
            return 0
        else:  # in one of State B
            return np.random.normal(-0.1, 1)

    def obtain_nextState(self, state, action):  # Return (nextState, Reward, Done?)
        rw = self.obtain_reward(state, action)
        if state == 0:  # State A
            if action == 0:
                return self.N - 1, rw, True  # absorbing state
            else:
                return np.random.randint(1, self.N - 1), rw, False  # Goto any one of State B
        else:  # In one of State B
            if action == 0:  # Go left, return to State A
                return 0, rw, False
            else:
                return self.N - 1, rw, True  # absorbing state

    # Choosing action using \epsilon-greedy
    def choose_action(self, Q, state, epsilon):
        if np.random.random() <= epsilon:
            return np.random.randint(0,2)
        else:
            if Q[state,0] == Q[state,1]:
                return np.random.randint(0,2)
            else:
                return np.argmax(Q[state])
    # Updating Q-value of state-action pair based on the update equation
    def update_q(self, state_old, action, reward, state_new, alpha):

        if self.policy == 'Q': #Q-learning
            self.Qa[state_old][action] += alpha * (reward + self.gamma * self.Qa[state_new][self.bestAction(state_new,self.Qa)] - self.Qa[state_old][action])

        if self.policy == 'D-Q':
            if np.random.randint(2) < 1: #update Qa
                self.Qa[state_old][action] += alpha * (reward + self.gamma * self.Qb[state_new][self.bestAction(state_new,self.Qa)] - self.Qa[state_old][action])
            else:
                self.Qb[state_old][action] += alpha * (reward + self.gamma * self.Qa[state_new][self.bestAction(state_new,self.Qb)] - self.Qb[state_old][action])

    # Exploration Rate
    def get_epsilon(self, t):
        return 0.1

    # Adaptive learning of Learning Rate
    def get_alpha(self, t):
        return 10 / (t + 100)

    def run(self):
        reward_data = []
        x_dd = []

        filename = "ProbLeft-"+self.policy
        if self.twofold == 1:
            filename = filename + "-twice"
        if self.average == True:
            filename = filename + "-average"

        print("Running: " + filename)

        for j in range(self.n_iter):

            start_time = time.time()
            self.initialize_Q()

            n_left = 0

            for e in range(self.n_episodes):

                n_left += self.obtain_GoLeft()
                self.rew_arr[e, j] = n_left / (e + 1)  # save error
                if e % 20 == 0:
                    print("Running: " + filename)
                    print("Iteration, Episode", j, e)
                    print("Running: Probability to Go Left latest estimate is :", n_left / (e + 1))

                current_state = 0 #Start from State A

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

                    newState, reward, done = self.obtain_nextState(current_state, action)

                    rw = rw + reward
                    # Update Q-Table
                    self.update_q(current_state, action, reward, newState, alpha)
                    current_state = newState
                    i += 1

                #calculate performance

            if j % 10 == 0:
                print("Save at iteration: ", j)
                file = open(filename, "wb")
                np.save(file, self.rew_arr)
                file.close()

            total_time = time.time() - start_time
            print("Time taken - %f seconds" % total_time)

        print("Save at iteration: ", j)
        file = open(filename, "wb")
        np.save(file, self.rew_arr)
        file.close()


if __name__ == "__main__":

    #reproduction
    np.random.seed(2020)
    # Make an instance of CartPole class 
    solver = Bias()
    solver.run()