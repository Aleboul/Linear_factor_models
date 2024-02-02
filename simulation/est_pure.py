from pickletools import optimize
import numpy as np
import networkx as nx
from scipy.optimize import LinearConstraint
from scipy.optimize import milp

def est_clique(Chi, delta):
    complementary_matrix = ((Chi >= 2*delta)*1.0) #- np.eye(Chi.shape[0])
    d = complementary_matrix.shape[0]

    vect = []
    for j in range(d):

        index = np.where(complementary_matrix[j,:] > min(j,0))[0]
        index = index[np.where(index > j)] # index[np.where(index > j)]

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

def est_pure_set(Chi, delta):
    """Estimate pure indices set

    Args:
        Chi (np.array[float, float]): Extremal correlation matrix
        delta (float): Tuning parameter of the algorithm
        
    Returns:
        Pure indices    
    """
    clique_max = est_clique(Chi, delta) # compute maximum clique

    pure = [] # initialization of the pure set
    S = np.array(np.setdiff1d(range(Chi.shape[0]), clique_max))
    for i in clique_max:
        index = np.where(1-Chi[i,:] < 2*delta)[0] # add pure index in the set
        index = np.union1d(np.intersect1d(S, index),i)
        pure.append(index) # increment
        S = np.setdiff1d(S, index)
        
    return pure 
