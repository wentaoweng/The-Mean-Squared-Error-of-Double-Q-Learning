'''
adopted from https://github.com/sanjitjain2/q-learning-for-cartpole/blob/master/qlearning.py
https://mc.ai/openai-gyms-cart-pole-balancing-using-q-learning/
'''

import numpy as np
import math
import time
import matplotlib.pyplot as plt
from collections import deque

def find(A):
    epi, iter = A.shape
    v = np.ones(iter) * 1000
    s = np.sum(A > 195,axis=0)
    for j in range(iter):
        if s[j] > 0:
            v[j] = (np.argmax(A[:,j] > 195) + 1) * 50
    return v

if __name__ == "__main__":

    figsize = 8, 4
    figure, ax = plt.subplots(figsize=figsize)

    file = open("Reward-Q", "rb")
    arr = np.load(file)
    terminate = find(arr)
    n_iter = terminate.size
    print(np.mean(terminate), np.std(terminate) / math.sqrt(n_iter))
    plt.hist(terminate, label = 'Q', alpha = 0.5, bins=np.arange(0, 1000 + 50, 50), weights=np.repeat(1.0/len(terminate), len(terminate)))

    file1 = open("Reward-D-Q-twofold-average", "rb")
    arr1 = np.load(file1)
    terminate1 = find(arr1)
    print(np.mean(terminate1), np.std(terminate1) / math.sqrt(n_iter))
    plt.hist(terminate1, label = 'D-Q avg with twice the step size', weights=np.repeat(1.0/len(terminate1), len(terminate1)), alpha = 0.5, bins=np.arange(0, 1000 + 50, 50))


    file2 = open("Reward-D-Q-twofold", "rb")
    arr2 = np.load(file2)
    terminate2 = find(arr2)
    print(np.mean(terminate2), np.std(terminate2) / math.sqrt(n_iter))
    plt.hist(terminate2, label = 'D-Q with twice the step size', weights=np.repeat(1.0/len(terminate2), len(terminate2)), alpha = 0.5, bins=np.arange(0, 1000 + 50, 50))

    file3 = open("Reward-D-Q", "rb")
    arr3 = np.load(file3)
    terminate3 = find(arr3)
    print(np.mean(terminate3), np.std(terminate3) / math.sqrt(n_iter))
    plt.hist(terminate3, label = 'D-Q', weights=np.repeat(1.0/len(terminate3), len(terminate3)), alpha = 0.5, bins=np.arange(0, 1000 + 50, 50))

    plt.tick_params(labelsize=20)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    font2 = {'family': 'Times New Roman',
             'weight': 'black',
             'size': 20,
             }

    plt.xlabel("Number of Episodes to Achieve At Least 195 Reward", font2)

    handles, labels = plt.gca().get_legend_handles_labels()
    legend = plt.legend(loc='best', shadow=True, fontsize='x-large')
    legend.get_title().set_fontsize(fontsize=20)
    legend.get_title().set_fontname('Times New Roman')
    legend.get_title().set_fontweight('black')
    plt.tight_layout()
    plt.savefig('./cartpole-episode.pdf', dpi=600, bbox_inches='tight')
    plt.close()
    print(arr)