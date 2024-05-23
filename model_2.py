"""
    This code contains all the necessary elements to reproduce experiments in Section 5.2 when
    we consider a fixed block size m and varying k and d.

    The goal of the given code is to simulate a data generation process for a Linear Factor Model
    and estimate the model parameters using extreme value theory over multiple iterations. The
    process involves generating synthetic data with specific dependencies and noise, calculating
    block maxima, and applying empirical rank transformations to derive extreme value coefficients.
    These coefficients are then used to estimate the matrix A that represents the linear combination
    of factors influencing the data. The estimated parameters are saved for each iteration, allowing
    for analysis and evaluation of the estimation method's performance across different simulated scenarios.

    Explanation:

    • The script simulates data from a specified model over a certain number of iterations.
    • It iterates over niter iterations, where each iteration represents a simulation run.
 	• In each iteration:
        • It samples data from a Clayton copula and transforms it to Pareto margins.
     	• Noise is generated using the multivariate Gaussian distribution.
     	• The data generation model is defined, including the construction of matrix A representing linear combinations of factors.
     	• Time series data X is generated based on the defined model.
     	• Block maxima are calculated from the generated data.
     	• Empirical ranks and extreme value coefficients are computed.
     	• Parameters are estimated using a specified function est_impure.est_A.
     	• The estimated parameters A_hat and the original matrix A are saved as CSV files.
 	• This process is repeated for each iteration.
"""

from clayton.rng.archimedean import Clayton
from clayton.rng.evd import Logistic
import numpy as np
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


np.random.seed(42)

# Parameters
# Define parameters for the simulation study
d = 200  # Dimensionality of the data
p = 2  # Parameter of the moving maxima process
rho = 0.8  # Parameter of the moving maxima process
k = 1000  # Number of blocks
m = 15  # Block size
n = k * m  # Total sample size
K = 20  # Number of columns
niter = 50  # Number of iterations

# Iterate over each iteration
for i in range(niter):
    print(i)  # Print the current iteration number

    # Sample from Clayton copula
    clayton = Clayton(dim=K, n_samples=n+p, theta=1.0)
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
    A = np.concatenate((np.eye(K), A))  # Concatenate identity matrix with A
    d = A.shape[0]  # Update the dimensionality of A

    # Generate time series data X
    W = np.array([np.max(np.c_[np.power(rho, 0) * sample_Cau[i, :], np.power(rho, 1) * sample_Cau[i - 1, :],
                               np.power(rho, 2) * sample_Cau[i - 2, :]], axis=1) for i in range(2, n + p)])
    X = np.array([np.matmul(A, W[i, :]) for i in range(n)]) + \
        np.random.multivariate_normal(mean, cov, size=1)[0]

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

    # Calculate extreme correlations
    extcoeff = -np.divide(outer, outer - 1)
    Theta = np.maximum(2 - extcoeff, 10e-5)  # Ensure Theta is non-negative

    # Calculate tuning parameter delta
    delta = (1 / m + 1.2 * np.sqrt(np.log(d) / k))

    print('... Estimation ...')
    # Estimate parameters using the provided function
    A_hat = est_impure.est_A(Theta, delta)
    K_hat = A_hat.shape[1]  # Get the number of clusters in estimated A_hat

    # Convert A_hat and A to DataFrames and save them as CSV files
    A_hat = pd.DataFrame(A_hat, columns=range(K_hat))
    A = pd.DataFrame(A, columns=range(K))
    pd.DataFrame.to_csv(A_hat, "results/results_model_2_calib/model_2_" + str(int(d)) + "_" + str(
        int(k)) + "/Ahat_" + str(int(i)) + ".csv")
    pd.DataFrame.to_csv(A, "results/results_model_2_calib/model_2_" + str(int(d)) + "_" + str(int(k)) + "/A" + str(
        int(i)) + ".csv")
