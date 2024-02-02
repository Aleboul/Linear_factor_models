import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('qb-light.mplstyle')

d = 92
n = 228
colors = ['#6fa8dc', '#e06666', '#93c47d']

wildfire = pd.read_csv('../data/wildfire.csv', index_col=0)

print(wildfire)

wildfire_sub = wildfire[['PIX','FWI']]

print(wildfire_sub)

pixels = np.unique(wildfire_sub.PIX)

index = np.where(wildfire_sub.PIX == pixels[0])

print(index)

print(wildfire_sub.iloc[index])


vect_m = np.repeat([30,31,31,30,31],20) #monthly maxima
data_wildfire = np.zeros([len(vect_m), len(pixels)])
l=0
for pixel in pixels:
    n = 0
    print(pixel)
    index = np.where(wildfire_sub.PIX == pixel)
    val = wildfire_sub.iloc[index].FWI.values
    start = 0
    end = 0
    for m in vect_m:
        end+=m
        indices =np.arange(start, end)
        data_wildfire[n,l] = np.max(val[indices])
        start+=m
        n+=1
    l +=1

Maxima = data_wildfire

A_hat = np.array(pd.read_csv('Ahat.csv', index_col = 0))
print(A_hat)

vect_x = np.arange(170,400, step = 2)
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
    ax.set_xlabel(r'FWI')
fig.savefig('results/plot_failure_sets/pmax.pdf')

vect_x = np.arange(2,20, step=2)

fig,ax =plt.subplots()
for latent in range(A_hat.shape[1]):
    index = np.where(A_hat[:,latent] > 0)[0]
    Maxima_subset = Maxima[:,index]

    p_min = []
    p_min_emp = []
    for x in vect_x:

        p_min.append(np.sum(np.min(A_hat[index,:], axis=0)/x))
        p_emp=0
        for i in range(n):
            value = np.where(Maxima_subset[i,:] > x)[0]
            if len(value) == len(index):
                p_emp+=1/n
        p_min_emp.append(p_emp)

    ax.plot(vect_x, p_min, linewidth=1, markersize=1, color = colors[latent])
    ax.plot(vect_x, p_min_emp, '--', linewidth=1, markersize=1, color = colors[latent])
    ax.set_ylabel(r'$\hat{p}_{min}$')
    ax.set_xlabel(r'FWI')
fig.savefig('results/plot_failure_sets/pmin.pdf')