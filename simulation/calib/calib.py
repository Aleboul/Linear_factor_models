"""
    This code contains numerical experiment to find the tuning parameter delta that minimizes
    the evaluation criterion and estimate the matrix A_hat representing the linear combination
    of factors in the model.

    The provided code simulates the estimation of parameters for a Linear Factor Model using
    extreme value theory. It generates synthetic data by sampling from a Clayton copula and
    adding Gaussian noise. The matrix A, representing factor loadings, is constructed with
    random support and values. The code then computes block maxima from the generated data
    and calculates empirical ranks and extreme value coefficients. By iterating over a range
    of tuning parameters, it estimates the matrix A_hat and evaluates the model using a
    predefined criterion. The best tuning parameter is selected based on the criterion,
    and the resulting criterion is saved for further analysis. The entire process is
    repeated for a specified number of iterations, and the results are saved to CSV files.

    Simulation loop:

        • For each iteration (`niter`):
         	• Generate samples from a Clayton copula and transform them to Pareto margins.
            • Construct the matrix `A` with random support and values.
            • Generate time series data `X` using the matrix `A` and the generated samples,
              with added Gaussian noise.
            • Calculate block maxima from the generated data.
            • Compute empirical ranks and extreme correlations.
            • Iterate over a range of `tau` values to compute the tuning parameter `delta` and
              estimate the matrix `A_hat`.
            • Evaluate the model using the computed criterion and store the results.

    Results saving:
        • Save the criterion values and average L2 norm values to CSV files for further analysis.
             
"""

from clayton.rng.archimedean import Clayton
import numpy as np
import pandas as pd
import est_impure
import est_pure
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)


def runif_in_simplex(n):
    ''' Return uniformly random vector in the n-simplex '''
    k = np.random.uniform(0.35, 0.65, size=n)
    return k / sum(k)


def compute_crit(pure, A_hat, Theta_test):
    ''' Compute the criterion for model evaluation '''
    K_hat = A_hat.shape[1]
    mat_index = np.zeros((d, K_hat))
    for a in range(K_hat):
        # Identify indices where A_hat is greater than 0
        index = np.where(A_hat[:, a] > 0.0)[0]
        in_index = np.zeros(d)
        in_index[index] = 1.0
        mat_index[:, a] = in_index
    value_1 = A_hat - Theta_test[:, pure] * mat_index
    # Compute the sum of squared differences
    crit = np.sum(np.power(value_1, 2))
    return crit


# Parameters
d = 400  # Number of variables
p = 2  # Parameter of the moving maxima process
rho = 0.8  # Autoregressive parameter
k = 300  # Number of blocks
m = 15  # Block size
n = k * m  # Total number of samples
K = 20  # Number of factors
niter = 50  # Number of iterations

crit_df = []  # List to store criterion values for each iteration
norm_df = []  # List to store L2 norm values for each iteration

for i in range(niter):
    print(i)
    # Sample Clayton copula
    clayton = Clayton(dim=K, n_samples=n + p, theta=1.0)
    sample_unimargin = clayton.sample_unimargin()
    sample_Cau = 1 / (1 - sample_unimargin)  # Set to Pareto margins

    mean = np.zeros(d)  # Mean for Gaussian noise
    cov = np.eye(d)  # Covariance matrix for Gaussian noise
    A = np.zeros((d - K, K))  # Initialize factor loading matrix
    for j in range(d - K):
        # Randomly choose number of non-zero entries
        s = np.random.randint(2, 5, 1)
        # Choose support indices
        support = np.random.choice(K, size=s, replace=False)
        A[j, support] = runif_in_simplex(s)  # Assign random values in simplex
    # Combine with identity matrix for pure indices
    A = np.concatenate((np.eye(K), A))
    d = A.shape[0]

    # Generate time series data
    W = np.array([np.max(np.c_[np.power(rho, 0) * sample_Cau[i, :], np.power(rho, 1) *
                 sample_Cau[i - 1, :], np.power(rho, 2) * sample_Cau[i - 2, :]], axis=1) for i in range(2, n + p)])
    X = np.array([np.matmul(A, W[i, :]) for i in range(n)]) + \
        np.random.multivariate_normal(mean, cov, size=1)[
        0]  # Add Gaussian noise

    block_maxima = np.zeros((k, d))  # Initialize block maxima array
    for j in range(d):
        sample = X[0:(k * m), j]  # Reshape data into blocks
        sample = sample.reshape((k, m))
        block_maxima[:, j] = np.max(sample, axis=1)  # Compute block maxima

    block_maxima = pd.DataFrame(block_maxima)  # Convert to DataFrame
    erank = np.array(block_maxima.rank() / (k + 1))  # Compute empirical ranks

    outer = (np.maximum(erank[:, :, None], erank[:, None, :])).sum(
        0) / k  # Compute outer product

    extcoeff = -np.divide(outer, outer - 1)  # Compute extremal coefficients
    Theta = np.maximum(2 - extcoeff, 10e-5)  # Compute Theta matrix

    # Range of tau values for tuning
    _tau_ = np.array(np.arange(0.75, 1.5, step=0.01))
    _K_ = []  # List to store number of pure indices
    vect_crit = []  # List to store criterion values
    norm = []  # List to store L2 norm values
    value_crit = 10e6  # Initial high value for comparison
    tuned_delta = 1.5  # Initial tuned delta value

    for tau in _tau_:
        delta = tau * (1 / m + np.sqrt(np.log(d) / k))  # Compute delta
        # Estimate A matrix and pure indices
        A_hat, pure = est_impure.est_A(Theta, delta)
        K_hat = A_hat.shape[1]
        crit = compute_crit(pure, A_hat, Theta)  # Compute criterion value
        vect_crit.append(crit)
        _K_.append(len(pure))
        if K_hat == K:
            norm_l2 = np.sum(np.linalg.norm(A_hat - A, axis=1)
                             ) / d  # Compute L2 norm
            norm.append(norm_l2)
        else:
            norm.append(np.nan)
        if crit < value_crit:  # Check if current criterion is the best
            tuned_delta = delta
            value_crit = crit
            K_tuned = K_hat

    # Re-estimate A with tuned delta
    A_hat, pure = est_impure.est_A(Theta, tuned_delta)

    crit_df.append(vect_crit)  # Store criterion values
    norm_df.append(norm)  # Store norm values

# Save results to CSV files
crit_df = pd.DataFrame(crit_df)
pd.DataFrame.to_csv(crit_df, 'results/results_calib/crit_' +
                    str(int(d)) + "_" + str(int(k)) + ".csv")

norm_df = pd.DataFrame(norm_df)
pd.DataFrame.to_csv(norm_df, 'results/results_calib/norm_' +
                    str(int(d)) + "_" + str(int(k)) + ".csv")
