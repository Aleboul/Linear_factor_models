import numpy as np
import est_pure

def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def est_AI(A, pure):
    """Estimate the matrix of pure indices

    Args:
        A (np.array[float, float]): d \times K matrix
        pure (list[]): List of pure indices
    """

    for clst, index in enumerate(pure):
        for i in index:
            A[i, clst] = 1

    return A


def est_AJ(A, Chi, delta, pure):
    """Estimate the matrix of impure index

    Args:
        A (np.array[float, float]): d \times K matrix
        Chi (np.array([float, float])): Extremal correlation matrix
        delta (float): Tuning parameter
        pure (list[]): List of pure indices
    """
    K_hat = len(pure)
    d = Chi.shape[0]
    impure = np.setdiff1d(np.array(range(d)), np.hstack(pure))
    for j in impure:
        chi_bar = [np.mean(Chi[j, pure[i]]) for i in range(K_hat)]
        hard_threshold = chi_bar * (chi_bar > 2*delta)
        index = np.where(hard_threshold > 0)[0]
        if len(index) > 0:
            beta_hat = np.zeros(K_hat)
            beta_hat[index] = projection_simplex_sort(hard_threshold[index])
        else:
            beta_hat = np.zeros(K_hat)
        A[j, :] = beta_hat
    
    return A

def est_A(Chi, delta, df):
    """Estimate the matrix A in a Linear Factor Model

    Args:
        Chi (np.array([float, float])): Extremal correlation matrix
        delta (float): Tuning parameter
    """
    d = Chi.shape[0]
    pure, clique_max = est_pure.est_pure_set(Chi, delta, df) # estimate pure indices
    K_hat = len(pure)
    A_hat = np.zeros((d,K_hat))
    A_hat = est_AI(A_hat, pure) # set value to pure indices
    A_hat = est_AJ(A_hat, Chi, delta, pure) # set value to impur indices
    return A_hat, clique_max
    

# Faire trois algorithmes pour estimer les variables pures, l'une avec la méthode de Bron, l'autre avec le MILP et celle avec la calibration 
# Prendre la clique maximale qui a le plus faible coefficient extrémale. Faire en sorte que cela soit propre et traduit par une seule option
# par l'utilisateur.