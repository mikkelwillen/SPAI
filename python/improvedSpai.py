import numpy as np
import scipy
import warnings
warnings.filterwarnings("ignore")
import permutation

def improvedSPAI(A, tol = 0.001, max_iter = 100, s = 5):
    determinant = np.linalg.det(A)
    if determinant == 0:    
        print("A is singular")
        return None
    
    M = scipy.sparse.identity(A.shape[1], format='csr')

    # index for the k'th column
    k = 0
    
    for m_k in M.T:
        print("_______________NEW COLUMN: %a_______________" %k)

        # 1) Find initial sparsity J of m_k
        m_k = M[:,k]

        J = m_k.nonzero()[0]
        n2 = J.size

        # 2) Compute the row indices I of the corresponding nonzero entries of A(i, J)
        A_J = A[:,J]

        I = np.unique(A_J.nonzero()[0])
        n1 = I.size

        # 3) Create Â = A(I, J)
        AHat = A[np.ix_(I, J)]

        # 4) Do QR decomposition of AHat. R_1 upper triangular n1 x n1 matrix. 0 is an (n1 − n2)×n2 zero matrix. Q1 is m×n, Q2 is m×(m − n).
        Q, R = np.linalg.qr(AHat.todense(), mode="complete")
        R_1 = R[0:n2,:]
        
        # 5) Compute the solution m_k for the least squares problem
        # 5.1) Compute cHat = Q^T * eHat_k
        e_k = np.matrix([0]*M.shape[1]).T
        e_k[k] = 1

        eHat_k = e_k[I]
        cHat_k = Q.T * eHat_k

        # 5.2) Compute the inverse of R
        invR_1 = np.linalg.inv(R_1)

        # 5.3) Compute mHat_k = R^-1 * cHat
        mHat_k = invR_1 * cHat_k[0:n2,:]

        m_k[J] = mHat_k

        # 6) Compute residual
        residual = A_J * mHat_k - e_k

        # iterations of the while-loop
        iter = 0

        while np.linalg.norm(residual) > tol and iter < max_iter:

            iter += 1

            # 7) Set L set of indices, where r(l) =/= 0
            L = np.nonzero(residual)[0]

            # 8) Set JTilde to all new column indices of A that appear in all L rows, but is not in J yet
            JTilde = np.array([],dtype=int)
            for l in L:
                nonzerosInA_l = np.unique(A[l,:].nonzero()[1])
                N_l = np.setdiff1d(nonzerosInA_l, J)
                JTilde = np.union1d(JTilde,N_l)

            # 9) For each j in JTilde solve the minimisation problem by computing: rho^2_j = ||r_new||^2 - (r^T A e_j)^2 / ||A e_j||^2
            rhoSq = []
            for j in JTilde:
                e_j = np.matrix([0]*M.shape[1]).T
                e_j[j] = 1

                rhoSq_j = ((np.linalg.norm(residual)) ** 2) - ((residual.T * A * e_j) ** 2) / (np.linalg.norm(A * e_j)) ** 2

                rhoSq.append(rhoSq_j)

            # 10) Find the indices JTilde corresponding to the smallest s elements of rho^2
            smallestIndices = sorted(range(len(rhoSq)), key = lambda sub: rhoSq[sub])[:s]

            # 11) Determine the new indices ITilde
            J = np.array(J)
            I = np.array(I)

            JTilde = np.sort(JTilde)
            ITilde = np.setdiff1d(np.unique(A[:,np.union1d(JTilde,J)].nonzero()[0]), I)
            
            n2Tilde = len(JTilde)
            n1Tilde = len(ITilde)

            # 12) Make I U ITilde and J U JTilde
            unionI = np.union1d(ITilde, I)
            unionJ = np.union1d(JTilde, J)

            # 13) Update the QR decomposition
            # 13.1) Compute ATilde from AHat, A(I, JTilde) and A(ITilde, JTilde)
            AIJTilde = A[np.ix_(I, JTilde)]

            AITildeJTilde = A[np.ix_(ITilde,JTilde)]

            # Find permutation matrices
            Pc = permutation.perm(J, n2, JTilde, n2Tilde, "col")
            Pr = permutation.perm(I, n1, ITilde, n1Tilde, "row")
            
            # 13.2) Compute ABreve = Q^T * A(I, JTilde)
            ABreve = Q.T * AIJTilde

            # 13.3) Compute B_1 = ABreve(0 : n2, 0 : n2)
            B_1 = ABreve[:n2,:]

            # 13.4) Compute B2 = ABreve(n2 + 1 : n1, 0 : n2Tilde) above AITildeJTilde
            B_2 = np.vstack((ABreve[n2:n1,:], AITildeJTilde.todense()))

            # 13.5) Do QR decomposition of B2
            Q_B, R_B = np.linalg.qr(B_2, mode="complete") 

            # 13.6) Compute Q_B and R_B from algorithm 17
            QZeroZeroIdentity = np.hstack((np.vstack((Q, np.zeros((n1Tilde, n1)))), np.vstack((np.zeros((n1, n1Tilde)), np.identity(n1Tilde)))))

            IdentityZeroZeroQ_B = np.hstack((np.vstack((np.identity(n2), np.zeros((n1-n2+n1Tilde,n2)))), np.vstack((np.zeros((n2,n1-n2+n1Tilde)), Q_B))))

            Q = QZeroZeroIdentity * IdentityZeroZeroQ_B
            
            R = np.hstack((np.vstack((R_1, np.zeros((n1Tilde + n1 - n2, n2)))), np.vstack((B_1, R_B))))

            # 13.7) Solve the augmented LS problem for mHat_k and compute new residual
            R_1 = R[0:n2,:]  

            eTilde_k = Pr.T * e_k[I]

            cHat_k = Q.T * eTilde_k 

            invR_1 = np.linalg.inv(R_1)

            mTilde_k = invR_1 * cHat_k[0:n2,:]

            # 15) set I = I U I_tilde, J = J U J_tilde and A' = A(I, J)
            J = unionJ
            n2 = J.size

            I = unionI
            n1 = I.size
            
            m_k[J] = Pc.T * mTilde_k

            # 14) Compute residual r
            residual = A[:,J] * m_k[J] - e_k

            # Permute Q and R to be used in next iteration
            Q = Pr * Q 
            R_1 = R_1 * Pc

        # 16) Set m_k(J) = mHat_k
        M[:,k] = m_k

        # iterate k
        k += 1
    
    return M


def printSPAI(A, tol = 0.001, max_iter = 100, s = 5):
    print("A = \n%a" % A.A)
    
    determinant = np.linalg.det(A.todense())
    print("det(A) = %a" % determinant)

    if determinant == 0:    
        print("A is singular")
        return None
    

    M = scipy.sparse.identity(A.shape[1], format='csr')
    print("M = \n%a" % M)

    # index for the k'th column
    k = 0
    
    for m_k in M.T:
        print("_______________NEW COLUMN: %a_______________" %k)

        # a) Find initial sparsity J of m_k
        m_k = M[:,k]
        print("m_k = %a" % m_k)

        J = m_k.nonzero()[0]
        n2 = J.size
        print("J = %a" % J)
        print("n2 = %a" % n2)

        # b) Compute the row indices I of the corresponding nonzero entries of A(i, J)
        A_J = A[:,J]

        I = np.unique(A_J.nonzero()[0])
        n1 = I.size
        print("I = %a" % I)
        print("n1 = %a" % n1)
        print("\n")

        # c) Create Â = A(I, J)
        AHat = A[np.ix_(I, J)]

        # d) Do QR decomposition of AHat. R_1 upper triangular n1 x n1 matrix. 0 is an (n1 − n2)×n2 zero matrix. Q1 is m×n, Q2 is m×(m − n).
        Q, R = np.linalg.qr(AHat.todense(), mode="complete")
        R_1 = R[0:n2,:]
        
        # e) Compute the solution m_k for the least sqaures problem
        # a) Compute cHat = Q^T * eHat_k
        e_k = np.matrix([0]*M.shape[1]).T
        e_k[k] = 1

        eHat_k = e_k[I]
        cHat_k = Q.T * eHat_k

        # b) Compute the inverse of R
        invR_1 = np.linalg.inv(R_1)

        # c) Compute mHat_k = R^-1 * cHat
        mHat_k = invR_1 * cHat_k[0:n2,:]

        m_k[J] = mHat_k

        # d) Compute residual
        residual = A_J * mHat_k - e_k

        # iterations of the while-loop
        iter = 0

        while np.linalg.norm(residual) > tol and iter < max_iter:
            print("_______________NEW ITERATION: %a_______________" %iter)
            print("\n")
            iter += 1

            # a) Set L set of indices, where r(l) =/= 0
            L = np.nonzero(residual)[0]

            # b) Set JTilde to all new column indices of A that appear in all L rows, but is not in J yet
            JTilde = np.array([],dtype=int)
            for l in L:
                A_l = A[[l],:]
                nonzerosInA_l = np.unique(A_l.nonzero()[1])
                N_l = np.setdiff1d(nonzerosInA_l, J)
                JTilde = np.union1d(JTilde,N_l)

            # c) For each j in JTilde solve the minimisation problem by computing: rho^2_j = ||r_new||^2 - (r^T A e_j)^2 / ||A e_j||^2
            rhoSq = []
            for j in JTilde:
                e_j = np.matrix([0]*M.shape[1]).T
                e_j[j] = 1

                rhoSq_j = ((np.linalg.norm(residual)) ** 2) - ((residual.T * A * e_j) ** 2) / (np.linalg.norm(A * e_j)) ** 2

                rhoSq.append(rhoSq_j)

            # d) Find the indices JTilde corresponding to the smallest s elements of rho^2
            # smallestIndices = sorted(range(len(rhoSq)), key = lambda sub: rhoSq[sub])[:s]

            # e) Determine the new indices ITilde
            J = np.array(J)
            I = np.array(I)

            JTilde = np.sort(JTilde)
            ITilde = np.setdiff1d(np.unique(A[:,np.union1d(JTilde,J)].nonzero()[0]), I)
            
            n2Tilde = len(JTilde)
            n1Tilde = len(ITilde)

            # f) Make I U ITilde and J U JTilde
            unionI = np.union1d(ITilde, I)
            unionJ = np.union1d(JTilde, J)

            # g) Update the QR decomposition
            # a) Compute ATilde from AHat, A(I, JTilde) and A(ITilde, JTilde)
            print("A =", A)
            print("I = ", I)
            print("JTilde = ", JTilde)
            # AIJTilde = A[np.ix_(I, JTilde)]

            AIJTilde = np.zeros((len(I), len(JTilde)))
            for i in range(len(I)):
                for j in range(len(JTilde)):
                    AIJTilde[i, j] = A[I[i], JTilde[j]]

            print("AIJTilde = ", AIJTilde)

            AITildeJTilde = A[np.ix_(ITilde,JTilde)]

            # Find permutation matrices
            Pc = permutation.perm(J, n2, JTilde, n2Tilde, "col")
            Pr = permutation.perm(I, n1, ITilde, n1Tilde, "row")

            # Virker også, men ekstremt langsomt for some reason
            # Pc = permutation.permutation(unionJ, "col")
            # Pr = permutation.permutation(unionI, "row")
            
            # b) Compute ABreve = Q^T * A(I, JTilde)
            print("Q = ", Q)
            # ABreve = Q.T * AIJTilde
            ABreve = np.dot(Q.T, AIJTilde)

            # c) Compute B_1 = ABreve(0 : n2, 0 : n2)
            B_1 = ABreve[:n2,:]

            # d) Compute B2 = ABreve(n2 + 1 : n1, 0 : n2Tilde) above AITildeJTilde
            print("ABreve = ", ABreve)
            print("ABreve[n2:n1,:] = ", ABreve[n2:n1,:])
            print("AITildeJTIlde = ", AITildeJTilde.todense())
            B_2 = np.vstack((ABreve[n2:n1,:], AITildeJTilde.todense()))

            # f) Do QR decomposition of B2
            Q_B, R_B = np.linalg.qr(B_2, mode="complete") 

            # g) Compute Q_B and R_B from algorithm 17
            QZeroZeroIdentity = np.hstack((np.vstack((Q, np.zeros((n1Tilde, n1)))), np.vstack((np.zeros((n1, n1Tilde)), np.identity(n1Tilde)))))

            IdentityZeroZeroQ_B = np.hstack((np.vstack((np.identity(n2), np.zeros((n1-n2+n1Tilde,n2)))), np.vstack((np.zeros((n2,n1-n2+n1Tilde)), Q_B))))

            Q = QZeroZeroIdentity * IdentityZeroZeroQ_B
            
            R = np.hstack((np.vstack((R_1, np.zeros((n1Tilde + n1 - n2, n2)))), np.vstack((B_1, R_B))))

            # Update I and J
            J = unionJ
            n2 = J.size

            I = unionI
            n1 = I.size

            # h) Solve the augmented LS problem for mHat_k and compute new residual
            R_1 = R[0:n2,:]  
            print("R_1 = ", R_1)

            eTilde_k = Pr.T * e_k[I]

            cHat_k = Q.T * eTilde_k 

            invR_1 = np.linalg.inv(R_1)

            mTilde_k = invR_1 * cHat_k[0:n2,:]

            # i) Set M(J U JTilde) 
            m_k[J] = Pc.T * mTilde_k

            # Compute residual r
            print("A[:, J] = ", A[:,J])
            print("m_k[J] = ", m_k[J])
            print("e_k = ", e_k)
            A_JDense = A[:,J].todense()
            residual = A_JDense * m_k[J] - e_k

            # Permute Q and R to be used in next iteration
            Q = Pr * Q 
            R_1 = R_1 * Pc
            print("R_1 after the permutation: ", R_1)
            print("\n")

        # Place result column in matrix
        M[:,k] = m_k

        # iterate k
        k += 1
    
    return M