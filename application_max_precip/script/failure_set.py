import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('qb-light.mplstyle')

d = 92
n = 228
colors = ['#6fa8dc', '#e06666', '#93c47d']

Maxima = np.array(pd.read_csv('../data/MaxPrecipFallFrance.csv'))

A_hat = np.array(pd.read_csv('Ahat.csv', index_col = 0))
print(A_hat)

vect_x = np.arange(50,100, step = 2)
latent = 1
fig, ax = plt.subplots()
for latent in range(A_hat.shape[1]):
    index = np.where(A_hat[:,latent] > 0)[0]
    Maxima_subset = Maxima[:,index]

    p_max = []
    p_max_emp = []
    for x in vect_x:

        p_max.append(np.sum(np.max(A_hat[index,:], axis=0)/x))

        p_emp = 0
        for i in range(n):
            value = np.where(Maxima_subset[i,:] > x)[0]
            if len(value) > 0:
                p_emp +=1/n
        p_max_emp.append(p_emp)

    ax.plot(vect_x, p_max, linewidth=1, markersize=1, color = colors[latent])
    ax.plot(vect_x, p_max_emp, '--', linewidth=1, markersize=1, color = colors[latent])
    ax.set_ylabel(r'$\hat{p}_{a}$')
    ax.set_xlabel(r'Precipitation threshold in mm')
fig.savefig('results/plot_failure_sets/pmax.pdf')

vect_x = np.arange(10,50, step=2)

fig,ax =plt.subplots()
for latent in range(A_hat.shape[1]):
    index = np.where(A_hat[:,latent] > 0)[0]
    Maxima_subset = Maxima[:,index]

    p_min = []
    for x in vect_x:

        p_min.append(np.sum(np.min(A_hat[index,:], axis=0)/x))

    ax.plot(vect_x, p_min, linewidth=1, markersize=1, color = colors[latent])
    ax.set_ylabel(r'$\hat{p}_{min}$')
    ax.set_xlabel(r'Precipitation threshold in mm')
fig.savefig('results/plot_failure_sets/pmin.pdf')