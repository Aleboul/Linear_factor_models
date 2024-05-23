"""
    The provided code performs a series of tasks related to precipitation data analysis.
    Initially, it imports the necessary libraries and reads the maxima precipitation data from a CSV file.
    Then, it calculates empirical ranks and dependence coefficients to prepare the data for model calibration. 
    Next, it calibrates the model using a range of tuning parameters to determine the optimal parameter values.
    The calibrated parameters are then used to estimate a matrix representing the relationship between different factors.
    Subsequently, the code visualizes the weights of each factor as a function of Euclidean distance, providing insights
    into the spatial relationships within the data.
"""

import pandas as pd
import numpy as np
import est_impure
import est_pure
import matplotlib.pyplot as plt
import networkx as nx
plt.style.use('qb-light.mplstyle')

def compute_crit(pure, A_hat, Theta):
    """Compute the criterion for model evaluation

    Args:
        pure (list): Estimated pure set [K]
        A_hat (np.array): Estimated matrix A representing the linear factor model
        Theta_test (np.array): Extremal correlation matrix

    Returns:
        crit (float): Criterion value for model evaluation

    This function computes the criterion value for evaluating the model. It takes the list of pure indices,
    the estimated matrix A (A_hat), and bivariate extremal correlations (Theta) as inputs. It constructs a binary
    matrix (mat_index) indicating estimated overlapping clusters, then calculates the difference between
    A_hat and the product of the selected columns over the estimated set [K] of Theta and mat_index. Finally,
    it returns the sum of squared differences as the criterion value (crit).
    """
    # Get the number of latent factors
    K_hat = A_hat.shape[1]

    # Initialize a matrix to indicate the presence of pure indices in A_hat
    mat_index = np.zeros((d, K_hat))

    # Iterate over each latent factor
    for a in range(K_hat):
        # Identify the indices where A_hat is greater than 0
        index = np.where(A_hat[:, a] > 0.0)[0]
        # Create a binary vector indicating the overlapping clusters
        in_index = np.zeros(d)
        in_index[index] = 1.0
        # Assign the binary vector to the corresponding column in mat_index
        mat_index[:, a] = in_index

    # Calculate the difference between A_hat and the product of Theta_test selected for the estimated set [K] and mat_index
    value_1 = A_hat - Theta[:, pure] * mat_index

    # Compute the criterion value as the sum of squared differences
    crit = np.sum(np.power(value_1, 2))

    # Return the criterion value
    return crit

def calibrate(Theta, parameters):
    """Calibrate the model by tuning parameters

    Args:
        Theta_train (np.array): Extremal correlation matrix
        parameters (list): List of parameter values to iterate over

    Returns:
        tuned_delta (float): Tuned value for the parameter delta
        tuned_pure (list): List of tuned pure set [K]

    This function calibrates the model by tuning parameters. It takes the training values for Theta (Theta_train),
    test values for Theta (Theta_test), and a list of parameter values to iterate over (parameters) as inputs. 
    It iterates over the parameter values, constructs an adjacency matrix based on Theta_train and the current
    parameter value, identifies maximal cliques in the graph, estimates the matrix A and pure indices using 
    the current parameter value, computes the criterion value for each clique, and selects the best-tuned 
    parameter value and corresponding pure indices based on the minimum criterion value. Finally, it returns 
    the tuned parameter value (tuned_delta) and tuned pure indices (tuned_pure).
    """
    # Initialize variables to store the tuned parameter and pure indices
    value_crit = 10e6
    tuned_delta = 2
    tuned_pure = []

    # Initialize an empty list to store criterion values for each parameter
    crit_vector = []

    # Iterate over each parameter value
    for delta in parameters:
        # Construct the adjacency matrix based on Theta_train and the current parameter value
        adjacency_matrix = ((Theta < delta) * 1.0)
        # Create a graph from the adjacency matrix
        G = nx.from_numpy_array(np.array(adjacency_matrix))
        # Find all maximal cliques in the graph
        cliques = nx.find_cliques(G)
        # Find the maximal clique with the maximum length
        clique_max = max(nx.find_cliques(G), key=len)

        # Initialize an empty list to store criterion values for each clique
        crit_in = []

        # Iterate over each clique
        for clique in cliques:
            # Check if the clique has the same length as the maximal clique
            if len(clique) == len(clique_max):
                # Estimate the matrix A and pure set [K] using the current parameter value and clique
                A_hat, pure = est_impure.est_A(Theta, delta, clique)
                # Compute the criterion value for the current clique
                crit = compute_crit(pure, A_hat, Theta)
                # Append the criterion value to the list
                crit_in.append(crit)

                # Update the tuned parameter value and estimated pure set [K] if the criterion value is lower
                if crit < value_crit:
                    value_crit = crit
                    tuned_delta = delta
                    tuned_pure = pure
                    print(tuned_pure, value_crit)

        # Append the minimum criterion value for the current parameter value to the vector
        crit_vector.append(np.min(crit_in))

    # Return the tuned parameter value and pure indices
    return tuned_delta, tuned_pure

# Import necessary libraries
Maxima = pd.read_csv('../data/MaxPrecipFallFrance.csv')  # Read maxima precipitation data from CSV file

n = 228  # Set the number of observations
d = 92  # Set the dimensionality of the data

# Calculate the empirical rank of the maxima data and compute outer dependence coefficients
erank = np.array(Maxima.rank() / (n + 1))
outer = (np.maximum(erank[:,:,None], erank[:, None,:])).sum(0) / n
extcoeff = -np.divide(outer, outer - 1)
Theta = np.maximum(2 - extcoeff, 10e-5)

# Define a range of tuning parameters
_delta_ = np.array(np.arange(0.5, 2.0, step=0.005)) * np.sqrt(np.log(d) / n)

# Calibrate the model and obtain the tuned delta and pure indices
tuned_delta, tuned_pure = calibrate(Theta, parameters=_delta_)

# Output the tuned delta and pure indices
print(tuned_delta)
print(tuned_pure)

# Estimate the matrix A using the calibrated parameters
A_hat, pure = est_impure.est_A(Theta, tuned_delta, clique_max=tuned_pure)
print(A_hat)
K_hat = A_hat.shape[1]

# Convert the matrix A_hat to a pandas DataFrame and save it to a CSV file
A_hat = pd.DataFrame(A_hat, columns=range(K_hat))
pd.DataFrame.to_csv(A_hat, "results/Ahat.csv")

# Read latitude and longitude data from CSV files
Longitudes = pd.read_csv('../data/LongitudesMaxPrecipFallFrance.csv')
Latitudes = pd.read_csv('../data/LatitudesMaxPrecipFallFrance.csv')

# Define colors for plotting
colors = ['#6fa8dc', '#e06666', '#93c47d', '#FFA500']

# Plot the weights of each latent factor as a function of Euclidean distance
for latent in range(A_hat.shape[1]):
    eu_di = []
    coordinate_pure = np.array([Longitudes.x[pure[latent]], Latitudes.x[pure[latent]]])
    for i in range(d):
        coordinate_impure = np.array([Longitudes.x[i], Latitudes.x[i]])
        eu_di.append(np.linalg.norm(coordinate_impure - coordinate_pure))

    fig, ax = plt.subplots()

    ax.scatter(x=eu_di, y=A_hat.iloc[:,latent].values, color=colors[latent])
    ax.set_ylabel('$A_{i\,'+str(latent)+'}$')
    ax.set_xlabel('Euclidean Distance')
    fig.savefig('results/plot_lfm/weights_distance' + str(latent)+'.pdf')
