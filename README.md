# The-Mean-Squared-Error-of-Double-Q-Learning

Here are codes for reproduction of the Neurips 2020 paper "The Mean-Squared Error of Double Q-Learning"



We test Double Q-learning and Q-learning for different environments. 

***All experiments below were run using Matlab R2018b and Python 3.6.9***

Environments that we considered

- Baird's example: bairds
- GridWorld: grid
- CartPole: cartpole
- Maximization Bias: Bias, Bias(nn)

## Baird's Experiment

File:  

1. bairds/GenBaird.m
2. bairds/simulation_baird.m
3. bairds/plot.py

In simulation_baird.m, change the input to the function GenBaird to simulate different settings

Run simulation_baird.m, it generates several files, meanings are the same as GridWorld specified later.

To plot the trace of mean-squared error: python3 plot.py

## GridWorld Experiment 

File: 

1. grid/GenGrid.m
2. grid/simulation_grid.m
3. grid/plot.py

In simulation_grid.m, change the input to the function GenGrid to simulate different size of GridWorld

Run simulation_grid, it generates several files, meanings are specified later.

1. Grid-n=3-errsingle.txt

   mean-squared error of Q-learning over a sample path of length 100000, averaged on 100 tests

2. Grid-n=3-stderrsingle.txt

   the standard deviation of each value in the last file

3. Grid-n=3-errDouble.txt

   Similar setting, for Double Q-learning

4. Grid-n=3-stdDouble.txt

5. Grid-n=3-erravg_d.txt

   Similar setting, for Double Q-learning with twice the step size and averaged estimator

6. Grid-n=3-stderravg_d.txt

7. Grid-n=3-errDouble_d.txt

   Similar setting, for Double Q-learning with twice the step size

8. Grid-n=3-stderrDouble_d.txt

To plot the trace of mean-squared error: python3 plot.py

## CartPole Experiment

File: 

1. cartpole/cartpole.py
2. cartpole/read.py

Dependence

1. OpenAI Gym

To run experiments: python cartpole.py

Change self.policy, self.twofold, self.avg on line 27 in the code to test:

1. Q-learning: self.policy = 'Q', self.twofold=0, self.avg = False
2. Double Q-learning: self.policy = 'D-Q', self.twofold=0, self.avg = False
3. Double Q-learning with twice the step size: self.policy = 'D-Q', self.twofold=1, self.avg = False
4. Double Q-learning with twice tep step size and averaged estimator: self.policy = 'D-Q', self.twofold=1, self.avg = True

It generates a file Reward-Q (or Reward-D-Q, Reward-D-Q-twice, Reward-D-Q-twice-average):

contains a 100*20 vectors (readable using numpy.load): 

the element on the i-th row and j-th column is the mean reward of the estimator on the 50j-th episode for the i-th test 

To plot the distribution of the hit time: python plot.py

## Maximization Bias Experiment

File: 

1. bias/bias.py

2. bias/plot.py

3. bias(nn)/bias(nn).py

4. bias(nn)/plot.py

Dependence
1. OpenAI Gym
2. PyTorch 1.6.0

bias.py is the code for a tabular setting and $M=8$, and bias(nn).py is the code for a setting with neural networks and $M=10^9$. They are basically the same, so we focus on bias(nn).py in the followings.

To run experiments: python bias(nn).py

Change self.policy, self.twofold, self.avg on line 59 in the code to test:

1. Q-learning: self.policy = 'Q', self.twofold=0, self.avg = False
2. Double Q-learning: self.policy = 'D-Q', self.twofold=0, self.avg = False
3. Double Q-learning with twice the step size: self.policy = 'D-Q', self.twofold=1, self.avg = False
4. Double Q-learning with twice tep step size and averaged estimator: self.policy = 'D-Q', self.twofold=1, self.avg = True

It generates a file ProbLeft-Q (or ProbLeft-D-Q, ProbLeft-D-Q-twice, ProbLeft-D-Q-twice-average):

contains a 1000*200 vectors (readable using numpy.load): 

the element on the i-th row and j-th column is the probability of a left action for the first j-th episode in the i-th test 

To plot the convergence of probability to go to the left, run python plot.py

   





