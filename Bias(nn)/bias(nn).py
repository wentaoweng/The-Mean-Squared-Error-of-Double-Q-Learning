'''
A variant of Barto & Sutton's example
   Normal(-1,10)          =0        =0
End<-(State2->State 10^9) <-> State1 -> End
'''
import gym
import numpy as np
import math
import time
import torch
from torch.autograd import Variable
from collections import deque

class DQL():
    ''' Deep Q Neural Network class. '''
    def __init__(self, state_dim, action_dim, lr, hidden_dim=4):
            self.criterion = torch.nn.MSELoss()
            self.model = torch.nn.Sequential(
                            torch.nn.Linear(state_dim, hidden_dim),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden_dim, hidden_dim * 2),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden_dim * 2, action_dim)
                    )
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr)

    def scheduler(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def update(self, state, y, lr):
            """Update the weights of the network given a training sample. """
            state = np.array([state])
            self.optimizer = self.scheduler(self.optimizer, lr)
            y_pred = self.model(torch.Tensor(state))
            loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, state):
            """ Compute Q values for all actions using the DQL. """
            state = np.array([state])
            with torch.no_grad():
                return self.model(torch.Tensor(state)).numpy()

class BiasLarge():
    def __init__(self, n_iter=1000, n_episodes=200, lr = 0.05):
        self.n_iter = n_iter
        self.n_state = 10**9 + 2 #State0, State1->State (10^6); State (10^6+1) absorbing state.
        self.n_statesDim = 1
        self.n_actionDim = 2 #Either left or Right
        self.lr = lr #learning rate of the NN
        self.n_episodes = n_episodes # training episodes
        self.gamma = 0.999 # discount factor

        ##policy can be 'Q' or 'D-Q'
        self.policy = 'Q'
        ##twofold = 0 : original step size; twofold = 1: step size *= 2
        self.twofold = 0
        ##average: =True, use averaged estimator; =False, only use Qa
        self.average = False

        self.rew_arr = np.zeros((self.n_episodes, self.n_iter))
        self.terminate_episodes = np.zeros(self.n_iter)

    def initialize_Q(self):
        # initialising Q-table
        self.Qa = DQL(self.n_statesDim, self.n_actionDim, self.lr)
        self.Qb = DQL(self.n_statesDim, self.n_actionDim, self.lr)


    def obtain_GoLeft(self):
        curQ = self.Qa.predict(0)
        QvalB = self.Qb.predict(0)

        if (self.policy == 'D-Q') & (self.average == True):
            curQ = (curQ + QvalB) / 2
        if curQ[1] > curQ[0]:
            return 1
        if curQ[1] == curQ[0]:
            return 0.5

        return 0

    def obtain_reward(selfs,state,action):
        if state == 0: #in State A
            return 0
        else: #in one of State B
            return np.random.normal(-0.1,1)

    def obtain_nextState(self,state,action): #Return (nextState, Reward, Done?)
        rw = self.obtain_reward(state,action)
        if state == 0: #State A
            if action == 0:
                return self.n_state - 1, rw, True #absorbing state
            else:
                return np.random.randint(1,self.n_state - 1), rw, False #Goto any one of State B
        else: #In one of State B
            if action == 0: #Go left, return to State A
                return 0, rw, False
            else:
                return self.n_state - 1, rw, True #absorbing state

    # Choosing action greedily
    def choose_action_greedy(self, Qval):
        return self.choose_action(Qval, epsilon = 0)

    # Choosing action using \epsilon-greedy
    def choose_action(self, Qval, epsilon):
        if np.random.random() <= epsilon:
            return np.random.randint(0,2)
        else:
            if Qval[0] == Qval[1]:
                return np.random.randint(0,2)
            else:
                return np.argmax(Qval)

    # Updating Q-value of state-action pair based on the update equation
    def update_q(self, QvalA, QvalB, state_old, action, reward, state_new, lr, done):

        if done == False:
            QvalANext = self.Qa.predict(state_new) #obtain Qtable for the next state
            QvalBNext = self.Qb.predict(state_new)

       #     print(QvalANext)
       #     print(QvalBNext)
        else:
            QvalANext = np.zeros(1)
            QvalBNext = np.zeros(1)
        if self.policy == 'Q': #Q-learning
            QvalA[action] = reward + self.gamma * np.max(QvalANext)
            self.Qa.update(state_old,QvalA, lr)

        if self.policy == 'D-Q':

            if np.random.randint(2) < 1: #update Qa
                QvalA[action] = reward + self.gamma * QvalBNext[np.argmax(QvalANext)]
                self.Qa.update(state_old,QvalA, lr)
            else:
                QvalB[action] = reward + self.gamma * QvalANext[np.argmax(QvalBNext)]
                self.Qb.update(state_old,QvalB, lr)

    # Exploration
    def get_epsilon(self, t):
        return 0.1

    def get_lr(self,t):
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

            self.terminate_episodes[j] = self.n_episodes + 1

            n_left = 0

            for e in range(self.n_episodes):

                n_left += self.obtain_GoLeft()
                self.rew_arr[e, j] = n_left / (e + 1)  # save error
                if e % 20 == 0:
                    print("Running: " + filename)
                    print("Iteration, Episode", j, e)
                    print("Running: Probability to Go Left latest estimate is :", n_left / (e + 1))

                current_state = 0 #Start from state A

                # Get adaptive learning alpha and epsilon decayed over time
                epsilon = self.get_epsilon(e)
                lr = self.get_lr(e) * (1 + self.twofold)

                done = False
                i = 0
                rw = 0

                while not done:
                    # Choose action according to greedy policy and take it
                    #curQ = self.Qa
                    QvalA = self.Qa.predict(current_state)
                    QvalB = self.Qb.predict(current_state)
                    QvalCur = QvalA #what estimation we're based on to do exploration
                    if (self.policy == 'D-Q') & (self.average == True):
                        QvalCur = (QvalA + QvalB) / 2
                    action = self.choose_action(QvalCur, epsilon)
                    newState, reward, done = self.obtain_nextState(current_state, action)
                    rw = rw + reward
                    #print(rw)
                    # Update Q-Table
                    self.update_q(QvalA, QvalB, current_state, action, reward, newState, lr, done)
                    if done:
                        break
                    current_state = newState
                    i += 1

                #calculate performance 

                self.rew_arr[e, j] = n_left / (e + 1)  # save error
                if e % 20 == 0:
                    print("Running: " + filename)
                    print("Iteration, Episode", j, e)

            if (j + 1) % 50 == 0: #save file 
                print("Save at iteration: ", j)
                file = open(filename, "wb")
                np.save(file, self.rew_arr)
                file.close()

            total_time = time.time() - start_time
            print("Time taken - %f seconds" % total_time)

        print("Final Save")
        file = open(filename, "wb")
        np.save(file, self.rew_arr)
        file.close()


if __name__ == "__main__":

    #reproduction
    np.random.seed(2020)
    torch.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)

    # Make an instance of CartPole class 
    solver = BiasLarge()
    solver.run()
