import numpy as np
import scipy as sp
import qr
import updateQR

# A = The matrix we want to perform the SPAI on
# tol = tolerance
# max_iter = constraint for the maximal number of iterations
# s = number of rho_j, the most profitable indices
def SPAI(A, tol = 0.001, max_iter = 100, s = 5):
    # M = sparsity matrix, set to diagonal
    M = np.zeros((A.shape[1], A.shape[0]))
    for i in range(M.shape[0]):
        M[i, i] = 1
    print(A)
    print(M)

    # index for the k'th column
    k = 0

    # m_k = column in M
    for m_k in M.T:
        # iteration number for this column
        iter = 0
        print("_______________NEW COLUMN: %a_______________" %k)

        # a) Find initial sparsity J of m_k
        J = []
        for i in range(len(m_k)):
            if m_k[i] != 0:
                J.append(i)

        # b) Compute the row indices I of the corresponding nonzero entries of A(i, J)
        # I = []
        # for i in range(A.shape[0]):
        #     keep = False
        #     for j in range(len(J)):
        #         if A[i, J[j]] != 0:
        #             keep = True
        #     if keep:
        #         I.append(i)

        A_J = A[:,J]
        I = list(np.unique(A_J.nonzero()[0]))
        
        # c) Create Â = A(I, J)
        AHat = np.zeros((len(I), len(J)), dtype = float)
        for i in range(len(I)):
            for j in range(len(J)):
                AHat[i, j] = A[I[i], J[j]]
        print("I:", I)
        print("J:", J)

        # d) Do QR decomposition
        Q1, R1 = qr.qr_factorization(AHat)
        Q, R = np.linalg.qr(AHat, mode="complete")
        
        # e) compute ĉ = Q^T ê_k
        e_kHat = np.zeros(len(I))
        for i in range(len(I)):
            if k == I[i]:
                e_kHat[i] = 1
        e_k = np.zeros(M.shape[1])
        e_k[k] = 1
        print("e_k:", e_kHat)
        cHat = np.matmul(Q1.T, e_kHat)
        print("cHat:", cHat)

        # f) compute ^m_k = R^-1 ĉ
        mHat_k = np.matmul(np.linalg.inv(R1), cHat)
        print("mHat_k:", mHat_k)

        # g) set m_k(J) = ^m_k
        i = 0
        for j in J:
            m_k[j] = mHat_k[i]
            i += 1
        
        print("A[:,J]:", A[:,J])
        # h) compute residual
        residual = np.subtract(np.matmul(A[:,J], mHat_k), e_k)
        print("res:", residual)

        # while residual > tolerance
        while np.linalg.norm(residual) > tol and max_iter > iter:
            iter+=1

            # a) Set L to the set of indices, where r(l) =/= 0
            L = []
            for i in range(len(residual)):
                if residual[i] != 0:
                    L.append(i)
            
            # b) Set Ĵ to all new column indices of A that appear in all L rows
            JTilde = []
            for i in L:
                for j in range(A.shape[1]):
                    if A[i, j] != 0 and j not in JTilde and j not in J:
                        JTilde.append(j)
            JTilde.sort()
            print("JTilde:", JTilde)

            # c) For each j in Ĵ compute:
            rhoSq = []
            for j in JTilde:
                e_j = np.zeros(A.shape[1])
                e_j[j] = 1
                µ = (np.matmul(np.matmul(residual.T, A), e_j) ** 2) / (np.linalg.norm(np.matmul(A, e_j)) ** 2)
                tmp = np.linalg.norm(residual) - µ
                rhoSq.append(tmp)
            print("rho:", rhoSq)

            # d) find the indices Ĵ corresponding to the to the smallest s elements of rho^2
            smallestIndices = sorted(range(len(rhoSq)), key = lambda sub: rhoSq[sub])[:s]
            print("smallest i:", smallestIndices)

            # e) determine the new indices Î
            ITilde = []
            for i in range(len(m_k)):
                keep = False
                if i not in I:
                    for j in range(len(JTilde)):
                        if A[i, JTilde[j]] != 0:
                            keep = True
                if keep:
                    ITilde.append(i)

            # f) Update the QR decomposition with algo 17 (too simple now)
            print("I: ", I)
            mHat_k, newI, newJ, Q, R1 = updateQR.updateQR(A, Q, R1, I, J, ITilde, JTilde, k)

            I = newI
            J = newJ

            # h) set m_k(J) = ^m_k
            i = 0
            print("J", J)
            for j in range(len(J)):
                print("(i, j): ", i, j)
                m_k[J[j]] = mHat_k[i]
                i += 1
            
        #iterate k
        k += 1
    
    print("A:\n", A)
    print("M:\n", M)
    return M
            
# A = np.array([[0, 1, 2],[3, 4, 0], [6, 0, 0]])
# B = np.array([[9, 1, 2],[3, 4, 5], [6, 7, 8]])
# C = np.array([[1], [2], [3]])
# SPAI(A)
