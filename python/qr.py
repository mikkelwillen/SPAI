import numpy as np

def qr_factorization(A):
    # A is a numpy array representing a matrix
    m = A.shape[0] # number of rows
    n = A.shape[1] # number of columns
    Q = np.zeros((m, n)) # initialize Q as a zero matrix
    R = np.zeros((n, n)) # initialize R as a zero matrix

    # Modified Gram-Schmidt algorithm
    for j in range(n):
        v = A[:, j] # copy the j-th column of A
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j]) # compute the i,j entry of R
            v = v - R[i, j] * Q[:, i] # subtract the projection of A[j] onto Q[i]
        R[j, j] = np.linalg.norm(v) # compute the j,j entry of R
        Q[:, j] = v / R[j, j] # normalize v and store it as the j-th column of Q

    return Q, R