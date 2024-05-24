from clayton.rng.archimedean import Clayton
import numpy as np
from scipy.stats import pareto
import pandas as pd
import est_impure
import matplotlib.pyplot as plt
import utilities as ut

import damex as dmx

from rpy2 import robjects as ro

def runif_in_simplex(n):
    ''' Return uniformly random vector in the n-simplex '''

    k = np.random.uniform(0.35, 0.65, size=n)
    return k / sum(k)


# Set the random seed for reproducibility
np.random.seed(42)

# Define parameters for the simulation
_d_ = [200]  # List of dimensions
_k_ = [1000]  # List of block sizes

# Set additional parameters
tau = 1.35  # Scaling factor for delta calculation
p = 2       # Parameter of moving maxima process
rho = 0.8   # Autoregressive parameter
m = 15      # Block size
K = 20      # Number of latent factors
niter = 50  # Number of iterations

# Loop over each combination of dimensions and block sizes
for d in _d_:
    for k in _k_:
        print('d: ', d, 'k: ', k)
        n = k * m  # Total number of samples

        # Initialize lists to store DAMEX and SCRAM evaluation results
        data_damex = []  # List for DAMEX method
        data_scram = []  # List for SCRAM method

        # Perform simulations
        for i in range(niter):
            # Sample Clayton copula
            clayton = Clayton(dim=K, n_samples=n + p, theta=1.0)
            sample_unimargin = clayton.sample_unimargin()
            sample_Cau = 1 / (1 - sample_unimargin)  # Set to Pareto margins

            # Generate mean and covariance matrices
            mean = np.zeros(d)
            cov = np.eye(d)

            # Generate mixing matrix A
            A = np.zeros((d - K, K))
            for j in range(d - K):
                s = np.random.randint(2, 5, 1)
                support = np.random.choice(K, size=s, replace=False)
                A[j, support] = runif_in_simplex(s)
            A = np.concatenate((np.eye(K), A))
            d = A.shape[0]

            # Generate synthetic data X
            W = np.array([np.max(np.c_[np.power(rho, 0) * sample_Cau[i, :], np.power(rho, 1) *
                                       sample_Cau[i - 1, :], np.power(rho, 2) * sample_Cau[i - 2, :]], axis=1)
                          for i in range(2, n + p)])
            X = np.array([np.matmul(A, W[i, :]) for i in range(n)]) + \
                np.random.multivariate_normal(mean, cov, size=n)

            # DAMEX Estimation
            V = ut.rank_transformation(X)
            R = int(np.sqrt(n))
            eps = 0.3  # Threshold parameter
            nb_min = 5  # Minimum number of faces per pure direction
            faces_dmx = dmx.damex(V, R, eps, nb_min)

            # Process DAMEX results
            # Here you perform some calculations based on the faces_dmx output
            FP = []
            FN = []
            hat_positive = []
            hat_null = []
            positive = []
            null = []

            clusters = {}
            l = 1
            S = np.arange(len(faces_dmx))
            # We merge directions having the same pure index.
            while len(S) > 0:
                i = S[0]
                clusters[l] = faces_dmx[i]
                iterand = S
                while len(iterand) > 0:
                    boolean = False
                    j = iterand[0]
                    test_set = np.intersect1d(clusters[l], faces_dmx[j])
                    if len(test_set) == len(faces_dmx[j]):
                        boolean += (test_set == faces_dmx[j]).all()
                    if len(test_set) == len(clusters[l]):
                        boolean += (test_set == clusters[l]).all()
                    if boolean:  # or (test_set == faces_dmx[j]).all():
                        clusters[l] = np.union1d(clusters[l], faces_dmx[j])
                        S = np.setdiff1d(S, j)
                        iterand = S
                    else:
                        iterand = np.setdiff1d(iterand, j)
                l += 1
            faces_dmx = [clusters[key] for key in clusters.keys()]
            if len(faces_dmx) == A.shape[1]:
                for faces in faces_dmx:
                    index_pure = faces[0]
                    null_group = np.where(A[:, index_pure] == 0.0)[0]
                    positive_group = np.where(A[:, index_pure] > 0)[0]
                    hat_positive_group = faces
                    hat_null_group = np.setdiff1d(range(0, d), faces)

                    hat_positive.append(len(hat_positive_group))
                    hat_null.append(len(hat_null_group))
                    positive.append(len(positive_group))
                    null.append(len(null_group))

                    FP.append(
                        len(np.intersect1d(null_group, hat_positive_group)))
                    FN.append(
                        len(np.intersect1d(positive_group, hat_null_group)))

                TFPP = np.sum(FP) / np.sum(null)
                TFNP = np.sum(FN) / np.sum(positive)

                data_damex.append([1.0, TFPP, TFNP])
            else:
                data_damex.append([0.0, np.nan, np.nan])

            # SCRAM Estimation

            block_maxima = np.zeros((k, d))
            for j in range(d):
                sample = X[0:(k*m), j]
                sample = sample.reshape((k, m))
                block_maxima[:, j] = np.max(sample, axis=1)

            block_maxima = pd.DataFrame(block_maxima)
            erank = np.array(block_maxima.rank() / (k+1))

            outer = (np.maximum(erank[:, :, None],
                     erank[:, None, :])).sum(0) / k

            extcoeff = -np.divide(outer, outer-1)
            Theta = np.maximum(2-extcoeff, 10e-5)

            # Setting tuning parameter using the data-driven tuning parameter
            delta = tau*(1/m+np.sqrt(np.log(d)/k))
            A_hat = est_impure.est_A(Theta, delta)
            K_hat = A_hat.shape[1]

            # Process SCRAM results
            # Here you perform some calculations based on the A_hat output

            FP = []
            FN = []
            hat_positive = []
            hat_null = []
            positive = []
            null = []

            if K_hat == A.shape[1]:

                for a in range(K_hat):
                    hat_positive_group = np.where(A_hat[:, a] > 0.0)[0]
                    hat_null_group = np.where(A_hat[:, a] == 0.0)[0]
                    positive_group = np.where(A[:, a] > 0.0)[0]

                    null_group = np.where(A[:, a] == 0.0)[0]

                    hat_positive.append(len(hat_positive_group))
                    hat_null.append(len(hat_null_group))
                    positive.append(len(positive_group))
                    null.append(len(null_group))

                    FP.append(
                        len(np.intersect1d(null_group, hat_positive_group)))
                    FN.append(
                        len(np.intersect1d(positive_group, hat_null_group)))

                TFPP = np.sum(FP) / np.sum(null)
                TFNP = np.sum(FN) / np.sum(positive)

                data_scram.append([1.0, TFPP, TFNP])
            else:
                data_scram.append([0.0, np.nan, np.nan])
        data_damex = pd.DataFrame(
            data_damex, columns=['damex', 'TFPP', 'TFNP'])
        print(np.mean(data_damex['damex']))
        data_scram = pd.DataFrame(
            data_scram, columns=['scram', 'TFPP', 'TFNP'])

        pd.DataFrame.to_csv(data_damex, "results/results_model_2/results/data_damex_" +
                            str(int(d))+"_"+str(int(k))+".csv")
        pd.DataFrame.to_csv(data_scram, "results/results_model_2/results/data_scram_" +
                            str(int(d))+"_"+str(int(k))+".csv")
