import numpy as np
import scipy as sp

# A = The matrix we want to perform the SPAI on
# tol = tolerance
# max_fill_in = constraint for the maximal number of iterations
# s = number of rho_j, the most profitable indices
def SPAI(A, tol = 0.5, max_fill_in = 100, s = 5):
    # M = sparsity matrix, set to diagonal
    M =  np.identity(A.shape[0])
    print(A)
    print(M)

    # m_k = column in M
    for m_k in M.T:
        # a) Find initial sparsity J of m_k
        J = []
        for i in range(len(m_k)):
            if m_k[i] != 0:
                J.append(i)
        # b) Compute the row indices I of the corrosponding nonzero entries of A(i, J)
        I = []
        for i in range(len(m_k)):
            keep = False
            for j in range(len(J)):
                if A[i, J[j]] != 0:
                    keep = True
            if keep:
                I.append(i)
        
        # c) Create Â = A(I, J)
        AHat = np.zeros((len(I), len(J)), dtype = float)
        for i in range(len(I)):
            for j in range(len(J)):
                AHat[i, j] = A[I[i], J[j]]
        
        # densification
        ADense = np.zeros(A.shape, dtype = float)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i in I and j in J:
                    ADense[i, j] = A[i, j]
        
        # d) Do QR decomposition
        Q, R = np.linalg.qr(ADense)

        # e) compute ĉ = Q^T ê_k
        eHat_k = // lav vektoren, hvor det k'te element er 1 og resten er 0
        cHat = Q.T 
A = np.array([[0, 1, 2],[3, 4, 0], [6, 0, 0]])
SPAI(A)
