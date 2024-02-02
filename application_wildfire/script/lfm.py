import pandas as pd
import numpy as np
import est_impure
import est_pure
import matplotlib.pyplot as plt
plt.style.use('qb-light.mplstyle')

wildfire = pd.read_csv('../data/wildfire.csv', index_col=0)

print(wildfire)

wildfire_sub = wildfire[['PIX','FWI']]

print(wildfire_sub)

pixels = np.unique(wildfire_sub.PIX)

index = np.where(wildfire_sub.PIX == pixels[0])

print(index)

print(wildfire_sub.iloc[index])


vect_m = np.repeat([30,31,31,30,31],20) #monthly maxima
data_wildfire = np.zeros([len(vect_m), len(pixels)])
l=0
for pixel in pixels:
    n = 0
    print(pixel)
    index = np.where(wildfire_sub.PIX == pixel)
    val = wildfire_sub.iloc[index].FWI.values
    start = 0
    end = 0
    for m in vect_m:
        end+=m
        indices =np.arange(start, end)
        data_wildfire[n,l] = np.max(val[indices])
        start+=m
        n+=1
    l +=1

print(data_wildfire)

Maxima = pd.DataFrame(data_wildfire)
n = 100
d = data_wildfire.shape[1]
print(Maxima)
erank = np.array(Maxima.rank() / (n+1))
outer = (np.maximum(erank[:,:,None], erank[:, None,:])).sum(0) / n

extcoeff = -np.divide(outer, outer-1)
Theta = np.maximum(2-extcoeff,10e-5)

print(Theta.shape[0])
plt.imshow(Theta)
plt.colorbar()
plt.show()

ext_coeff_vector = []
alpha_ =  np.arange(0.1,0.7, step=0.01)
for alpha in alpha_:
    delta = alpha*np.sqrt(np.log(d)/n)
    clique, ext_coeff = est_pure.est_clique_calibrate(Theta, delta, Maxima, ec=True)
    ext_coeff_vector.append(ext_coeff)

fig, ax = plt.subplots()
ax.plot(alpha_, ext_coeff_vector, '-o', linewidth=1, markersize=1)
ax.set_ylabel('Extremal Coefficient')
ax.set_xlabel(r'$c_\ell$')
plt.show()
fig.savefig('results/plot_lfm/extremal_coefficient_delta.pdf')

delta = 0.36*np.sqrt(np.log(d)/n) #0.505

A_hat, clique_max = est_impure.est_A(Theta, delta, Maxima)
K_hat = len(clique_max)

print(Theta[clique_max,:][:,clique_max])

print(A_hat)
clique_max = np.hstack(clique_max)

A_hat = pd.DataFrame(A_hat, columns = range(K_hat))

pd.DataFrame.to_csv(A_hat, "Ahat.csv")

print(clique_max)