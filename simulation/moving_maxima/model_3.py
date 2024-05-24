"""
    This code contains all the necessary elements to reproduce experiments in Section 5.2 when
    we consider a fixed block size n and varying k, m and d.

    The provided code simulates a data generation process for a Linear Factor Model and estimates the model parameters
    using extreme value theory. It initializes various parameters and runs a single iteration to generate synthetic
    data with specific dependencies using a Clayton copula. The matrix A representing the linear combination of factors,
    is constructed with random support and values. Time series data X is generated with added Gaussian noise. Block maxima
    are then calculated from this data, and empirical ranks are computed to derive extreme value coefficients. Using these
    coefficients and a calculated tuning parameter δ, the code estimates the parameter matrix A. The estimated matrix A
    and the original matrix A are saved as CSV files for further analysis, providing a comprehensive simulation
    and estimation framework for the Linear Factor Model.

    Explanations:

    • The script simulates a data generation process for a Linear Factor Model over 50 iteration (niter=50).
 	• It begins by setting a random seed for reproducibility and defining various parameters such as the number of blocks (k),
      dimensionality (d), total sample size (n), number of clusters (K), and block size (m).
 	• In each iteration, the script:
     	• Samples from a Clayton copula and transforms the samples to Pareto margins.
     	• Constructs the matrix A, which represents the linear combination of factors, with random support and values.
     	• Generates time series data X using the defined model and adds Gaussian noise.
     	• Calculates block maxima from the generated data.
     	• Computes empirical ranks and extreme value coefficients from the block maxima.
     	• Estimates the parameter matrix A_hat using the est_impure.est_A function.
     	• Finally, the script saves the estimated parameter matrix A_hat and the original matrix A as CSV files for further analysis.
	• This process is repeated for each iteration.
"""

from clayton.rng.archimedean import Clayton
import numpy as np
from scipy.stats import pareto
import pandas as pd
import est_impure
import matplotlib.pyplot as plt


def runif_in_simplex(n):
    ''' 
    Return a random vector in the n-simplex.

    Args:
        n (int): Dimensionality of the simplex.

    Returns:
        np.array[float]: A random vector in the n-simplex.
    '''

    # Generate a random vector k of size n with values uniformly distributed between 0.35 and 0.65
    k = np.random.uniform(0.35, 0.65, size=n)

    # Normalize the vector k by dividing each element by the sum of all elements
    return k / sum(k)

# Parameters


np.random.seed(42)

# Parameters

np.random.seed(42)  # Set the random seed for reproducibility

p = 2  # Parameter of the moving maxima process
rho = 0.8  # Parameter of the moving maxima process
k = 1000  # Number of blocks
d = 200  # Dimensionality of the data
n = 5000  # Total sample size
K = 20  # Number of clusters
niter = 50  # Number of iterations
m = int(n / k)  # Block size

# Iterate over each iteration
for i in range(niter):
    print(i)  # Print the current iteration number

    # Sample from Clayton copula
    clayton = Clayton(dim=K, n_samples=n + p, theta=1.0)
    sample_unimargin = clayton.sample_unimargin()
    sample_Cau = 1 / (1 - sample_unimargin)  # Transform to Pareto margins

    mean = np.zeros(d)  # Mean vector for data generation
    cov = np.eye(d)  # Covariance matrix for data generation

    # Initialize matrix A representing the linear combination of factors
    A = np.zeros((d - K, K))
    for j in range(d - K):
        # Randomly choose the number of non-zero elements in each row
        s = np.random.randint(2, 5, 1)
        # Randomly choose the positions of non-zero elements
        support = np.random.choice(K, size=s, replace=False)
        # Set the non-zero elements using a random vector in the simplex
        A[j, support] = runif_in_simplex(s)
    # Concatenate identity matrix with A, setting pure variables
    A = np.concatenate((np.eye(K), A))
    d = A.shape[0]  # Update the dimensionality of A

    # Generate time series data X
    W = np.array([np.max(np.c_[np.power(rho, 0) * sample_Cau[i, :], np.power(rho, 1) * sample_Cau[i - 1, :],
                               np.power(rho, 2) * sample_Cau[i - 2, :]], axis=1) for i in range(2, n + p)])
    X = np.array([np.matmul(A, W[i, :]) for i in range(n)]) + \
        np.random.multivariate_normal(mean, cov, size=n)

    # Calculate block maxima
    block_maxima = np.zeros((k, d))
    for j in range(d):
        sample = X[0:(k * m), j]
        sample = sample.reshape((k, m))
        block_maxima[:, j] = np.max(sample, axis=1)

    # Convert block maxima to DataFrame
    block_maxima = pd.DataFrame(block_maxima)
    # Calculate empirical ranks
    erank = np.array(block_maxima.rank() / (k + 1))

    # Calculate outer product of ranks
    outer = (np.maximum(erank[:, :, None], erank[:, None, :])).sum(0) / k

    # Calculate extreme value coefficients
    extcoeff = -np.divide(outer, outer - 1)
    Theta = np.maximum(2 - extcoeff, 10e-5)  # Ensure Theta is non-negative

    # Calculate tuning parameter delta
    delta = (1 / m + 1.2*np.sqrt(np.log(d) / k))
    print(delta)

    print('estimation')
    # Estimate parameters using the provided function
    A_hat = est_impure.est_A(Theta, delta)
    K_hat = A_hat.shape[1]  # Get the number of clusters in estimated A_hat
    print(K_hat)

# Convert A_hat and A to DataFrames and save them as CSV files
A_hat = pd.DataFrame(A_hat, columns=range(K_hat))
A = pd.DataFrame(A, columns=range(K))
pd.DataFrame.to_csv(A_hat, "results/results_model_3/model_3_" +
                    str(int(d)) + "_" + str(int(k)) + "/Ahat_" + str(int(i)) + ".csv")
pd.DataFrame.to_csv(A, "results/results_model_3/model_3_" +
                    str(int(d)) + "_" + str(int(k)) + "/A" + str(int(i)) + ".csv")
