"""
    The provided code performs an analysis of precipitation data for France
    using a Linear Factor Model. It begins by loading the maximum precipitation
    data and factor weights obtained from the model. The code then calculates and
    visualizes the probabilities of exceeding thresholds for different clusters 
    identified by the factor model. By iterating through each cluster, the code
    computes both model-based probabilities and empirical probabilities
    based on the provided data. Finally, the results are plotted, providing insights into the
    relationship between precipitation levels and cluster membership, aiding in the understanding
    of extreme precipitation patterns across different regions of France.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('qb-light.mplstyle')

# Define the number of locations
d = 92
# Define the number of samples
n = 228
# Define colors for visualization
colors = ['#6fa8dc', '#e06666', '#93c47d', '#FFA500']

# Load the matrix containing maximum precipitation data for France
Maxima = np.array(pd.read_csv('../data/MaxPrecipFallFrance.csv'))

# Load the matrix containing factor weights obtained from the model
A_hat = np.array(pd.read_csv('results/Ahat.csv', index_col=0))
print(A_hat)

# Define the range of precipitation thresholds
vect_x = np.arange(50, 100, step=2)

# Plot the probability of exceeding the threshold for each cluster
fig, ax = plt.subplots()
for latent in range(A_hat.shape[1]):
    # Retrieve indices of locations belonging to the current cluster
    index = np.where(A_hat[:, latent] > 0)[0]
    Maxima_subset = Maxima[:, index]

    p_max = []
    p_max_emp = []
    for x in vect_x:
        # Calculate the probability of exceeding the threshold based on the model
        p_max.append(np.sum(np.max(A_hat[index, :], axis=0) / x))

        # Calculate the empirical probability of exceeding the threshold
        p_emp = 0
        for i in range(n):
            value = np.where(Maxima_subset[i, :] > x)[0]
            if len(value) > 0:
                p_emp += 1 / n
        p_max_emp.append(p_emp)

    # Plot the model-based and empirical probabilities
    ax.plot(vect_x, p_max, linewidth=1, markersize=1, color=colors[latent])
    ax.plot(vect_x, p_max_emp, '--', linewidth=1, markersize=1, color=colors[latent])
    ax.set_ylabel(r'$\hat{p}_{a}$')
    ax.set_xlabel(r'Precipitation threshold in mm')
fig.savefig('results/plot_failure_sets/pmax.pdf')

# Define the range of precipitation thresholds
vect_x = np.arange(10, 50, step=2)

# Plot the probability of falling below the threshold for each cluster
fig, ax = plt.subplots()
for latent in range(A_hat.shape[1]):
    # Retrieve indices of locations belonging to the current cluster
    index = np.where(A_hat[:, latent] > 0)[0]
    Maxima_subset = Maxima[:, index]

    p_min = []
    for x in vect_x:
        # Calculate the probability of falling below the threshold based on the model
        p_min.append(np.sum(np.min(A_hat[index, :], axis=0) / x))

    # Plot the model-based probabilities
    ax.plot(vect_x, p_min, linewidth=1, markersize=1, color=colors[latent])
    ax.set_ylabel(r'$\hat{p}_{min}$')
    ax.set_xlabel(r'Precipitation threshold in mm')
fig.savefig('results/plot_failure_sets/pmin.pdf')
