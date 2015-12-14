from pandas import Series
from igraph import *
# from numba import jit
import numpy as np
import os
# import time

# Gather all the files.
files = os.listdir('timeseries/')

# Concatenate (or stack) all the files.
# Approx 12.454981 seconds
i = 0
for f in files:
    if i == 0:
        ts_matrix = np.loadtxt('timeseries/' + f).T
        i += 1
    else:
        new_ts = np.loadtxt('timeseries/' + f).T
        ts_matrix = np.hstack((ts_matrix, new_ts))

"""
Compute the correlation matrix
"""
corr_mat = np.corrcoef(ts_matrix.T)

# Save in .npz file
# np.savez_compressed('corr_mat.npz', corr_mat=corr_mat)

# X = np.load('corr_mat.npz')
# X = X['corr_mat']

# a flatten function optimized by numba
# @jit
def fast_flatten(X):
    k = 0
    length = X.shape[0] * X.shape[1]
    X_flat = empty(length)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_flat[k] = X[i, j]
            k += 1
    return X_flat

# helper function that returns the min of the number of
# unique values depending on the threshold
def min_thresh_val(X, threshold):
    X_flat = fast_flatten(X)
    index = int(len(X_flat) * threshold)
    return unique(sort(X_flat))[::-1][:index].min()

# Computes the threshold matrix without killing the python kernel
# @jit
def thresh_mat(X, threshold):
    min_val = min_thresh_val(X, threshold)
    print("Done with min_thresh_val")
    # M = zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # if X[i, j] >= min_val:
                # M[i, j] = X[i, j]
            if X[i, j] < min_val:
                X[i, j] = 0

thresh_mat(X, .01)
print("Finished Threshold Matrix")
# savez_compressed('threshold_mat.npz', threshold_mat=X)

# from: http://stackoverflow.com/questions/29655111/igraph-graph-from-numpy-or-pandas-adjacency-matrix

# get the row, col indices of the non-zero elements in your adjacency matrix
conn_indices = np.where(X)

# get the weights corresponding to these indices
weights = X[conn_indices]

# a sequence of (i, j) tuples, each corresponding to an edge from i -> j
edges = zip(*conn_indices)

# initialize the graph from the edge sequence
G = Graph(edges=edges, directed=False)

# assign node names and weights to be attributes of the vertices and edges
# respectively
G.vs['label'] = np.arange(X.shape[0])
G.es['weight'] = weights

# get the vertex clustering corresponding to the best modularity
cm = G.community_multilevel()

# save the cluster membership of each node in a csv file
Series(cm.membership).to_csv('mem.csv', index=False)
