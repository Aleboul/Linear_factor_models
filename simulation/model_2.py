from clayton.rng.archimedean import Clayton
import numpy as np
from scipy.stats import pareto
import pandas as pd
import est_impure
import matplotlib.pyplot as plt

# Parameters

d = 1000
p = 2
k = 1000
m = 20
n = k * m
K = 20
niter = 1

for i in range(niter):
    print(i)
    # Sample Clayton copula
    clayton = Clayton(dim=K, n_samples=n+p, theta=1.0)
    sample_unimargin = clayton.sample_unimargin()
    sample_Cau = 1/(1-sample_unimargin) # Set to Pareto margins
    #sample_Cau = -np.power(np.log(sample_unimargin), -1)

    mean = np.zeros(d)
    cov = np.eye(d)
    rho = 0.8
    A = np.zeros((d-K,K))
    for j in range(d-K):
        s = np.random.randint(2,5,1)
        support = np.random.choice(K, size=s, replace=False)
        A[j,support] = 1 / s
    A = np.concatenate((np.eye(K),A))
    #np.random.shuffle(A)
    d = A.shape[0]

    W = np.array([np.matmul(A,sample_Cau[i,:]) for i in range(n+p)]) #  + np.random.multivariate_normal(mean, cov, size = 1)[0] le bruit à ajouter sur une ligne quelconque
    X = np.array([np.max(np.c_[np.power(rho,0)*W[i,:], np.power(rho,1)*W[i-1,:], np.power(rho,2)*W[i-2,:]], axis =1) for i in range(2,n+p)]) + np.random.multivariate_normal(mean, cov, size = 1)[0]

    # Modèle dépendant.

    block_maxima = np.zeros((k,d))
    for j in range(d):
        sample =X[0:(k*m),j]
        sample = sample.reshape((k,m))
        block_maxima[:,j] = np.max(sample, axis = 1)
        #block_maxima[:,j] = np.power(np.max(sample, axis = 1),m)

    block_maxima = pd.DataFrame(block_maxima)
    erank = np.array(block_maxima.rank() / (k+1))

    outer = (np.maximum(erank[:,:,None], erank[:, None,:])).sum(0) / k

    extcoeff = -np.divide(outer, outer-1)
    Theta = np.maximum(2-extcoeff,10e-5)

    delta = 0.55*(1/m+np.sqrt(np.log(d)/k)) # 0.55 for d = 800, k = 300
    print(delta)
    print('estimation')
    A_hat = est_impure.est_A(Theta, delta)
    K_hat = A_hat.shape[1]
    print(K_hat)

    A_hat = pd.DataFrame(A_hat, columns = range(K_hat))
    A = pd.DataFrame(A, columns = range(K))


    #pd.DataFrame.to_csv(A_hat, "results/results_model_2/model_2_" + str(int(d)) + "_" + str(int(k)) + "/Ahat_" + str(int(i)) + ".csv")
    #pd.DataFrame.to_csv(A, "results/results_model_2/model_2_" + str(int(d)) + "_" + str(int(k)) + "/A" + str(int(i)) + ".csv")


# Pour sparsité 5, 0.40 for d = 1000, k = 300