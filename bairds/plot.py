
from numpy import *
from decimal import *
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def as_num(x):
    y = '{:.10f}'.format(x)
    return(y)

def readData(data_path):
    f = open(data_path) 
    lines = f.readlines() 
    A = zeros((20000), dtype=float)  
    A_row = 0  
    for line in lines:  
        list = line.strip(' ').split('\n') 
        map(Decimal, list)
        A[A_row] = list[0]  
        A_row += 1 
    return A

err_single = readData('./Baird=type2-errsingle.txt')
std_single = readData('./Baird=type2-stderrsingle.txt')
err_double = readData('./Baird=type2-errDouble.txt')
std_double = readData('./Baird=type2-stderrDouble.txt')
err_double_twice = readData('./Baird=type2-errDouble_d.txt')
std_double_twice = readData('./Baird=type2-stderrDouble_d.txt')
err_avg_twice = readData('./Baird=type2-erravg_d.txt')
std_avg_twice = readData('./Baird=type2-stderravg_d.txt')


x=range(20000)

figsize = 8,4
figure, ax = plt.subplots(figsize=figsize)

ax.set_yscale('log')

plt.errorbar(x, err_single, 2*std_single, fmt='r--', capsize=2.5, errorevery = 500, markevery=500, label='Q')
plt.errorbar(x, (err_double), 2*std_double, fmt='b-o',  capsize=2.5, markevery=500, errorevery=500, label='D-Q')
plt.errorbar(x, (err_double_twice), 2*std_double_twice, fmt='k-*',  capsize=2.5, markevery=500, errorevery=500, label='D-Q with twice the step size')
plt.errorbar(x, (err_avg_twice), 2*std_avg_twice, fmt='g-x',  capsize=2.5, markevery=500, errorevery=500, label='D-Q avg with twice the step size')

handles, labels = plt.gca().get_legend_handles_labels()
legend = plt.legend(loc='best', shadow=True, fontsize='x-large')
legend.get_title().set_fontsize(fontsize = 20)
legend.get_title().set_fontname('Times New Roman')
legend.get_title().set_fontweight('black')

plt.tick_params(labelsize=20)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

font2 = {'family': 'Times New Roman',
         'weight': 'black',
         'size': 20,
         }
plt.xlabel('Number of Samples', font2)
plt.ylabel('Mean-Squared Error', font2)
plt.tight_layout()
plt.savefig('./Baird-type2-log.pdf', dpi=600, bbox_inches='tight')
plt.close()