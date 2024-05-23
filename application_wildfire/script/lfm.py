"""
    This code performs the data-driven estimation of the matrix A in a Linear Factor Model
    alongside Conditions (i)-(ii).

    The provided Python code performs several operations related to analyzing
    wildfire data. Initially, it constructs a DataFrame named "Maxima" from the
    wildfire dataset, which contains observations of Fire Weather Index (FWI) at different
    geographical locations. The code then computes extremal coefficients. A range of tuning
    parameters is defined to calibrate the model, and the "calibrate" function is called
    to determine the optimal tuning parameter and identify pure set [K] in the dataset.
    The calibrated parameters are then used to estimate the matrix A, which represents the
    structure of clusters in the wildfire dataset. Finally, the code saves the estimated matrix
    A to a CSV file named "Ahat.csv" for further analysis or visualization.

    You may count roughly 5 minutes for the code to process.
"""

import numpy as np  # Library for numerical computing
import pandas as pd  # Library for data manipulation
import pandas as pd
import numpy as np
import est_impure
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


# Importing necessary libraries

# Reading wildfire data from a CSV file
wildfire = pd.read_csv('../data/wildfire.csv', index_col=0)

# Selecting specific columns from the dataset
wildfire_sub = wildfire[['PIX', 'FWI']]

# Extracting unique pixel values
pixels = np.unique(wildfire_sub.PIX)

# Defining monthly maximum values
vect_m = np.repeat([30, 31, 31, 30, 31], 20)

# Initializing an array to store processed wildfire data
data_wildfire = np.zeros([len(vect_m), len(pixels)])

# Iterating over each pixel
l = 0
for pixel in pixels:
    n = 0
    print(pixel)
    # Finding indices where pixel values match the current pixel
    index = np.where(wildfire_sub.PIX == pixel)
    # Extracting FWI values corresponding to the pixel
    val = wildfire_sub.iloc[index].FWI.values
    start = 0
    end = 0
    # Iterating over each month
    for m in vect_m:
        end += m
        # Generating indices for the current month
        indices = np.arange(start, end)
        # Storing the maximum FWI value for the current month
        data_wildfire[n, l] = np.max(val[indices])
        start += m
        n += 1
    l += 1

# Printing the processed wildfire data
print(data_wildfire)

# Creating a DataFrame from the wildfire data
Maxima = pd.DataFrame(data_wildfire)

# Defining variables for number of observations and dimensions
n = 100
d = data_wildfire.shape[1]

# Computing empirical rank
erank = np.array(Maxima.rank() / (n+1))

# Calculating outer rank correlation
outer = (np.maximum(erank[:, :, None], erank[:, None, :])).sum(0) / n

# Computing extremal coefficients
extcoeff = -np.divide(outer, outer-1)

# Computing threshold parameters
Theta = np.maximum(2-extcoeff, 10e-5)

# Defining a range of tuning parameters
_delta_ = np.array(np.arange(0.5, 1.0, step=0.005)) * np.sqrt(np.log(d) / n)

# Calibrating the model
tuned_delta, tuned_pure = calibrate(Theta=Theta, parameters=_delta_)

# Printing the tuned delta and [K]
print(tuned_delta)
print(tuned_pure)

# Estimating matrix A with calibrated parameters
A_hat, pure = est_impure.est_A(Theta, tuned_delta, clique_max=tuned_pure)

# Determining the number of clusters
K_hat = A_hat.shape[1]

# Creating a DataFrame for matrix A
A_hat = pd.DataFrame(A_hat, columns=range(K_hat))

# Saving matrix A to a CSV file
pd.DataFrame.to_csv(A_hat, "results/Ahat.csv")

