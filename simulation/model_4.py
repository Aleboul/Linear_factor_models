from clayton.rng.archimedean import Clayton
import numpy as np
from scipy.stats import pareto
import pandas as pd
import est_impure
import matplotlib.pyplot as plt
import eco_alg
from sklearn.metrics.cluster import adjusted_rand_score

# Parameters

D = [110,110,110,110,110] # length of clusters
p = 2
k = 300
m = 20
n = k * m
K = 50
d = np.sum(D) + K
niter = 50
step = int(K / len(D))

true_cluster = {'1' : np.arange(0,120), '2' : np.arange(120,240), '3' : np.arange(240,360), '4' : np.arange(360,480), '5' : np.arange(480,600)}

true_labels = np.zeros(d)

for key, values in true_cluster.items():
    true_labels[values] = key

ari_lfm = []
ari_eco = []

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
    l=0
    A = np.zeros((d,K))
    for dim in D:
        for j in range(l*(dim+step),(l+1)*(dim+step)):
            s = np.random.randint(2,5,1)
            index_sample = np.arange(l*step,step*(l+1))
            support = np.random.choice(index_sample, size=s, replace=False)
            A[j,support] = 1 / s
        A[range(dim+(dim+step)*l, dim+(dim+step)*l+step),:]=np.eye(K)[range(l*(step),(l+1)*step),:]
        l += 1
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

    block_maxima = pd.DataFrame(block_maxima)
    erank = np.array(block_maxima.rank() / (k+1))

    outer = (np.maximum(erank[:,:,None], erank[:, None,:])).sum(0) / k

    extcoeff = -np.divide(outer, outer-1)
    Theta = np.maximum(2-extcoeff,10e-5)

    delta = 0.55*(1/m+np.sqrt(np.log(d)/k)) # 0.55 for d = 800, k = 300
    print(delta)
    A_hat = est_impure.est_A(Theta, delta)
    overlapping_groups = {}
    for latent in range(K):
        groups = np.where(A[:,latent]>0)[0]
        overlapping_groups[latent] = groups
    clusters = {}
    l = 1
    S = np.arange(K)
    while len(S) > 0:
        i = S[0]
        clusters[l] = overlapping_groups[i]
        iterand = S
        while len(iterand)>0:
            j = iterand[0]
            if set(clusters[l]) & set(overlapping_groups[j]):
                clusters[l] = np.union1d(clusters[l], overlapping_groups[j])
                S = np.setdiff1d(S,j)
                iterand = S
            else:
                iterand = np.setdiff1d(iterand,j)
        l+=1
    pred_labels_lfm = np.zeros(d)
    for key, values in clusters.items():
        pred_labels_lfm[values] = key
    ari_lfm.append(adjusted_rand_score(pred_labels_lfm, true_labels))

    O_hat = eco_alg.clust(Theta, 2*delta)
    pred_labels_eco = np.zeros(d)
    for key, values in O_hat.items():
        pred_labels_eco[values] = key
    ari_eco.append(adjusted_rand_score(pred_labels_eco, true_labels))
    
    K_hat = A_hat.shape[1]

print(ari_eco)
print(ari_lfm)

ari_lfm = pd.DataFrame(ari_lfm)
ari_eco = pd.DataFrame(ari_eco)
    
pd.DataFrame.to_csv(ari_eco, "results/results_model_4/model_4_" + str(int(d)) + "_" + str(int(k)) + "/ari_eco" + ".csv")
pd.DataFrame.to_csv(ari_lfm, "results/results_model_4/model_4_" + str(int(d)) + "_" + str(int(k)) + "/ari_lfm" + ".csv")
