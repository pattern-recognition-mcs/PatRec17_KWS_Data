"""

Adapted to our project from the code of Claudio Bellei in https://github.com/cbellei/DTW/blob/master/DTW.py

--------------------
DYNAMIC TIME WARPING
--------------------
Input: list of feature vectors 1, list of feature vectors 2
Output: cost (distance between the warped time series)
options:
1. plot. If True, plots a few figures.
2. test. If True, uses test data.
"""

import numpy as np
import matplotlib.pyplot as plt

def dtw(ts1=[], ts2=[]):

    m = len(ts1)
    n = len(ts2)
    DTW = np.zeros((n, m), dtype=float)

    # first row
    for i in range(1, m):
        DTW[0, i] = distance(ts1[i], ts2[0]) + DTW[0, i - 1]
        
    # first column
    for i in range(1, n):
        DTW[i,0] = distance(ts1[0],ts2[i]) + DTW[i-1,0]

    for i in range(1,n):
        for j in range(1,m):
            cost = distance(ts1[j],ts2[i])
            DTW[i,j] = cost + np.min([DTW[i-1,j],  \
                                        DTW[i,j-1],  \
                                        DTW[i-1,j-1]])

    # now find best path, going backwards
    path = dict()
    path[0] = (n - 1, m - 1, DTW[n - 1, m - 1])
    c = 0
    finished = False
    i = n - 1
    j = m - 1
    while (not finished):
        v = np.array([DTW[i-1,j],DTW[i-1,j-1],DTW[i,j-1]])
        cost = np.min(v)
        k = np.where(v==cost)[0][0]
        if k==0:
            path[c] = (i-1,j,cost)
            i = i-1
        elif k==1:
            path[c] = (i-1,j-1,cost)
            j = j-1
            i = i-1
        else:
            path[c] = (i,j-1,cost)
            j = j-1
        if path[c][0]==0 and path[c][1]==0:
            finished = True
        c += 1

    path_i = np.array([])
    path_j = np.array([])
    cost = np.array([])
    for k in path.keys():
        path_i = np.append(path_i, path[k][0])
        path_j = np.append(path_j,path[k][1])
        cost = np.append(cost, path[k][2])


    path_j = np.asarray(path_j, dtype=int)
    path_i = np.asarray(path_i, dtype=int)
    
    return np.sum(cost)


def distance(featureVector1, featureVector2):
    return np.linalg.norm(featureVector1 - featureVector2)

