import numpy as np
import math
import time
import matplotlib.pyplot as plt
from collections import deque

if __name__ == "__main__":

    figsize = 8, 4
    figure, ax = plt.subplots(figsize=figsize)

    file1 = open("./ProbLeft-Q", "rb")
    file2 = open("./ProbLeft-D-Q", "rb")
    file3 = open("./ProbLeft-D-Q-twice", "rb")
    file4 = open("./ProbLeft-D-Q-twice-average", "rb")
    num_iter = 1000
    arr1 = np.load(file1)
    arr2 = np.load(file2)
    arr3 = np.load(file3)
    arr4 = np.load(file4)
    mean1 = np.mean(arr1,axis=1)
    var1 = np.sqrt(np.var(arr1,axis=1) / num_iter)
    mean2 = np.mean(arr2,axis=1)
    var2 = np.sqrt(np.var(arr2,axis=1) / num_iter)
    mean3 = np.mean(arr3,axis=1)
    var3 = np.sqrt(np.var(arr3,axis=1) / num_iter)
    mean4 = np.mean(arr4,axis=1)
    var4 = np.sqrt(np.var(arr4,axis=1) / num_iter)

    x=range(200)

    #ax.set_yscale('log')

    plt.errorbar(x, mean1, 2 * var1, fmt='r--', capsize=2.5, errorevery=10, markevery=10, label='Q')
    plt.errorbar(x, mean2, 2 * var2, fmt='b-o', capsize=2.5, markevery=10, errorevery=10, label='D-Q')
    plt.errorbar(x, mean3, 2 * var3, fmt='k-*', capsize=2.5, markevery=10, errorevery=10,
                 label='D-Q with twice the step size')
    plt.errorbar(x, mean4, 2 * var4, fmt='g-x', capsize=2.5, markevery=10, errorevery=10,
                 label='D-Q avg with twice the step size')

    handles, labels = plt.gca().get_legend_handles_labels()
    legend = plt.legend(loc='best',
          ncol=1, fancybox=True, shadow=True, prop={'size': 16})
    legend.get_title().set_fontname('Times New Roman')
    legend.get_title().set_fontweight('black')

    plt.tick_params(labelsize=20)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    font2 = {'family': 'Times New Roman',
             'weight': 'black',
             'size': 20,
             }
    plt.xlabel('Number of Episodes', font2)
    plt.ylabel('Probability of a left action', font2)
    plt.tight_layout()
    plt.savefig('./Sutton-Barto(nn).pdf', dpi=600, bbox_inches='tight')
    plt.close()
