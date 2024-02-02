import numpy as np
import networkx as nx
from scipy.optimize import LinearConstraint
from scipy.optimize import milp


def est_clique(Chi, delta):
    """Estimate the maximum clique in the adjacency matrix using extremal correlation


        Args:
            Chi (np.array[float, float]):
                Extremal correlation matrix
            delta (float) : 
                Tuning parameter of the algorithm

        Returns:
            List of clique
    """
    adjacency_matrix = ((Chi < 2*delta) * 1.0)  # compute adjacency matrix
    # transform the adjacency matrix
    G = nx.from_numpy_array(np.array(adjacency_matrix))
    # into a python graph.
    clique = max(nx.find_cliques(G), key=len)
    clique = list(np.sort(clique))
    
    return clique

def est_clique(Chi, delta):
    complementary_matrix = ((Chi >= 2*delta)*1.0) #- np.eye(Chi.shape[0])
    d = complementary_matrix.shape[0]

    vect = []
    for j in range(d):

        index = np.where(complementary_matrix[j,:] > min(j,0))[0]

        for i in index:
            input = np.zeros(d)
            input[j] = 1
            input[i] = 1
            vect.append(input)
    vect = np.array(vect)
    b_u = np.ones(vect.shape[0])
    b_l = np.zeros(vect.shape[0])

    constraints = LinearConstraint(vect, b_l, b_u)
    c = -np.ones(complementary_matrix.shape[0])

    integrality = np.ones_like(c)

    res = milp(c=c, constraints = constraints, integrality = integrality)
    clique = np.where(res.x>0.5)[0]
    return clique

def est_clique_calibrate(Chi, delta, df, ec = False):
    """ Find a clique on the graph using Bron, C. and Kerbosch, J. [1973] algorithm and
    return the largest clique with the lowest estimated extremal coefficient.
    """
    adjacency_matrix = ((Chi < 2*delta) * 1.0)
    n = df.shape[0]
    G = nx.from_numpy_array(np.array(adjacency_matrix))
    cliques = nx.find_cliques(G)
    clique_max = max(nx.find_cliques(G), key=len)
    ext_clique = 10e6
    for clique in cliques:
        if len(clique) == len(clique_max):
            K_hat = len(clique)
            clique = np.hstack(clique)
            maxima_subset = df.iloc[:,clique]
            erank = np.array(maxima_subset.rank() / (n+1))
            madogram = np.mean(np.max(erank, axis = 1)) - 0.5
            ext_coeff = np.maximum(K_hat-(0.5 + madogram) / (0.5-madogram),0)
            if ext_coeff < ext_clique:
                ext_clique = ext_coeff
                chosen_clique = clique
    clique = list(np.sort(chosen_clique))
    if ec:
        return clique, ext_clique
    else:
        return clique
        
def est_pure_set(Chi, delta,df = None):
    """Estimate pure indices set

    Args:
        Chi (np.array[float, float]): Extremal correlation matrix
        delta (float): Tuning parameter of the algorithm
        
    Returns:
        Pure indices    
    """
    clique_max = est_clique_calibrate(Chi, delta, df) # compute maximum clique
    
    pure = [] # initialization of the pure set

    for i in clique_max:
        index = np.where(1-Chi[i,:] < 2*delta)[0] # add pure index in the set
        pure.append(index) # increment
        
    return pure , clique_max
