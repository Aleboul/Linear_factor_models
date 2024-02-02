import numpy as np
from scipy.stats import pareto
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_helpers import *
plt.style.use('qb-light.mplstyle')


d = [100,200,300]
s = [3,4,5,6,7,8,9,10,11,12,13,14,15]
niter = 20

fig, ax = plt.subplots()
markers = ['o', '^', '*', 's', 'p']

l=0
for d_ in d:
    avg = []

    experiment = pd.read_csv(str(d_)+'_time.csv')
    experiment = np.array(experiment)

    print(experiment)
    for s_ in s:
        j = s_-s[0]
        avg_time = 0
        for i in range(niter):
            avg_time += experiment[i,j]
            # compute mean
        avg.append(avg_time / niter)
    print(avg)

    results = pd.DataFrame(np.c_[s,avg], columns=['s','avg'])

    print(results)

    sns.lineplot(x=results.s, y=results.avg, ax=ax, lw=1, label=str(int(d_)),**marker_settings(colors[1], markersize=2), marker = markers[l])
    l +=1
fig.tight_layout()
ax.legend(loc='center right')
ax.set_ylabel(r'$\ln$ Average ratio (in seconds)')
ax.set_xlabel(r'sparsity degree $s$')
plt.show()

fig.savefig('num_res_exact_recovery_l_2_norm.pdf')