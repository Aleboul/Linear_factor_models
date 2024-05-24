"""
    This code conducts a simulation study to evaluate the performance of two latent factor estimation methods,
    SKmeans and SCRAM, on synthetic data generated using a Clayton copula.

    The goal of the given code is to simulate a data generation process for a Linear Factor Model
    and estimate the model parameters using extreme value theory over multiple iterations. The
    process involves generating synthetic data with specific dependencies and noise, calculating
    block maxima, and applying empirical rank transformations to derive extreme correlations. These
    coefficients are then used to estimate the matrix A that represents the linear combination of
    factors influencing the data. Two methods are then used to estimate the normalised columns of A.
    Errors are then stored for further analysis.

    Explanation:

    • The script simulates data from a specified model over a certain number of iterations.
    • It iterates over niter iterations, where each iteration represents a simulation run.
 	• In each iteration:
        • It samples data from a Clayton copula and transforms it to Pareto margins.
     	• Noise is generated using the multivariate Gaussian distribution.
     	• The data generation model is defined, including the construction of matrix A representing linear combinations of factors.
     	• Time series data X is generated based on the defined model.
        • Normalised columns of A are estimated using sKmeans.
            • An error is computed and stored.
     	• Normalised columns of A are estimated using Scram procedure.
            • An error is computed and stored
 	• This process is repeated for each iteration.
"""

# Import necessary libraries and modules
from clayton.rng.archimedean import Clayton
import numpy as np
import pandas as pd
import est_impure
import matplotlib.pyplot as plt
import utilities as ut
import itertools
from rpy2 import robjects as ro


def runif_in_simplex(n):
    ''' Return random vector in the n-simplex '''
    k = np.random.uniform(0.35, 0.65, size=n)
    return k / sum(k)


# Set random seed for reproducibility
np.random.seed(42)

# Parameters
_d_ = [1000]  # List of dimensions
_k_ = [1000]  # List of block sizes

p = 2   # Parameter of the moving process
rho = 0.8  # Autoregressing parameter
m = 15  # Number of samples per block
K = 6   # Number of latent factors
niter = 50  # Number of iterations
tau = 1.35  # Scaling factor for delta calculation

# Loop over each combination of dimensions and block sizes
for d in _d_:
    for k in _k_:
        print('d: ', d, "k: ", k)
        n = k * m  # Total number of samples

        # Initialize lists to store SKmeans and SCRAM errors
        vect_error_skmeans = []
        vect_error_scram = []

        # Perform simulations
        for i in range(niter):
            # Sample Clayton copula
            clayton = Clayton(dim=K, n_samples=n+p, theta=1.0)
            sample_unimargin = clayton.sample_unimargin()
            sample_Cau = 1 / (1 - sample_unimargin)  # Set to Pareto margins

            # Generate mean and covariance matrices
            mean = np.zeros(d)
            cov = np.eye(d)

            # Generate matrix A
            A = np.zeros((d - K, K))
            for j in range(d - K):
                s = np.random.randint(2, 3, 1)
                support = np.random.choice(K, size=s, replace=False)
                A[j, support] = runif_in_simplex(s)
            A = np.concatenate((np.eye(K), A))

            # Generate synthetic data X
            true_centers = A / np.linalg.norm(A, axis=0)
            W = np.array([np.max(np.c_[np.power(rho, 0) * sample_Cau[i, :], np.power(rho, 1) *
                                       sample_Cau[i - 1, :], np.power(rho, 2) * sample_Cau[i - 2, :]], axis=1)
                          for i in range(2, n + p)])
            X = np.array([np.matmul(A, W[i, :]) for i in range(n)]) + \
                np.random.multivariate_normal(mean, cov, size=n)

            # Save data for R source file
            V = ut.rank_transformation(X)  # Empirical Pareto margins
            export = pd.DataFrame(V)
            export.to_csv("results_model_1/data/" + "data_" + str(1) + ".csv")

            # Source R files
            r = ro.r
            r.source('model_1.R')

            # SKmeans Estimation
            centers_skmeans = np.matrix(pd.read_csv(
                'results_model_1/results_skmeans/centers_' + str(1) + '.csv'))
            permutations = list(itertools.permutations(range(0, K)))
            error_skmeans = 10e6
            for perm in permutations:
                norm = np.sqrt(np.sum(np.power(np.linalg.norm(
                    centers_skmeans.T - true_centers[:, perm], axis=0), 2)))
                if norm < error_skmeans:
                    error_skmeans = norm
            vect_error_skmeans.append(error_skmeans)

            # SCRAM Estimation

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

            # Calculate outer maximum of ranks
            outer = (np.maximum(erank[:, :, None],
                     erank[:, None, :])).sum(0) / k

            # Calculate extreme correlations
            extcoeff = -np.divide(outer, outer - 1)
            Theta = np.maximum(2 - extcoeff, 10e-5)

            # Calculate tuning parameters and use the data-driven tuning parameter
            delta = tau * (1 / m + 1.0 * np.sqrt(np.log(d) / k))
            A_hat = est_impure.est_A(Theta, delta)
            K_hat = A_hat.shape[1]
            centers_scram = A_hat / np.linalg.norm(A_hat, axis=0)

            # Compute if the exact number of latent variables is recovered
            if K_hat == K:
                error_scram = np.sqrt(
                    np.sum(np.power(np.linalg.norm(centers_scram - true_centers, axis=0), 2)))
                vect_error_scram.append(error_scram)
            else:
                vect_error_scram.append(np.nan)

        # Save SKmeans and SCRAM errors to CSV files
        vect_error_skmeans = pd.DataFrame(vect_error_skmeans)
        vect_error_scram = pd.DataFrame(vect_error_scram)
        pd.DataFrame.to_csv(
            vect_error_skmeans, "results/results_model_1/results/error_skmeans_" + str(int(d)) + "_" + str(int(k)) + ".csv")
        pd.DataFrame.to_csv(
            vect_error_scram, "results/results_model_1/results/error_scram_" + str(int(d)) + "_" + str(int(k)) + ".csv")
