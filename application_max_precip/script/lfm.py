import random
from tracemalloc import stop
from turtle import color
import pandas as pd
import numpy as np
import est_impure
import est_pure
import matplotlib.pyplot as plt
plt.style.use('qb-light.mplstyle')

np.random.seed(3)

Maxima = pd.read_csv('../data/MaxPrecipFallFrance.csv')

n = 228
d = 92
#rand_index=np.random.permutation(d)
#Maxima = Maxima.iloc[:,rand_index]
erank = np.array(Maxima.rank() / (n+1))
outer = (np.maximum(erank[:,:,None], erank[:, None,:])).sum(0) / n
extcoeff = -np.divide(outer, outer-1)
Theta = np.maximum(2-extcoeff,10e-5)

print(Theta.shape[0])
plt.imshow(Theta)
plt.colorbar()
plt.show()
ext_coeff_vector = []
alpha_ =  np.arange(0.1,1.0, step=0.005) # alpha_ =  np.arange(0.2,0., step=0.005)
for alpha in alpha_:
    delta = alpha*np.sqrt(np.log(d)/n)
    clique, ext_coeff = est_pure.est_clique_calibrate(Theta, delta, Maxima, ec=True)
    ext_coeff_vector.append(ext_coeff)

fig, ax = plt.subplots()
ax.plot(alpha_, ext_coeff_vector, '-o', linewidth=1, markersize=1)
ax.set_ylabel('Extremal Coefficient')
ax.set_xlabel(r'$c_\ell$')
fig.savefig('results/plot_lfm/extremal_coefficient_delta.pdf')
plt.show()

delta = 0.56*np.sqrt(np.log(d)/n) #0.505
print(delta)

A_hat, pure = est_impure.est_A(Theta, delta, Maxima)
K_hat = len(pure)


print(pure)

index = [18,28,64,65]
print(Theta[index,:][:,index])
plt.imshow(Theta[index,:][:,index], cmap='Blues')
plt.text(1,1,"West", fontsize = 12, color = 'white')
plt.text(2,2,"South", fontsize = 12, color = 'white')
plt.text(3,3,"East", fontsize = 12, color = 'white')
plt.colorbar()
plt.show()

pure = np.hstack(pure)

A_hat = pd.DataFrame(A_hat, columns = range(K_hat))

pd.DataFrame.to_csv(A_hat, "Ahat.csv")

Longitudes = pd.read_csv('../data/LongitudesMaxPrecipFallFrance.csv')
Latitudes = pd.read_csv('../data/LatitudesMaxPrecipFallFrance.csv')

colors = ['#6fa8dc', '#e06666', '#93c47d', '#FFA500']

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