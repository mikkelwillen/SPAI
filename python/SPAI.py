import numpy as np
import scipy
import warnings
warnings.filterwarnings("ignore")
import permutation

def SPAI(A, tol = 0.001, max_iter = 39, s = 1):

    determinant = np.linalg.det(A.todense())
    if determinant == 0:    
        print("A is singular")
        return None
    

    M = scipy.sparse.identity(A.shape[1], format='csc')

    # index for the k'th column
    k = 0
    
    for m_k in M.T:
        # print("_______________NEW COLUMN: %a_______________" %k)

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

        # 4) Do QR decomposition of AHat
        Q, R = np.linalg.qr(AHat.todense(), mode="complete")
        R_1 = R[0:n2,:]
        
        # 5) Compute the solution m_k for the least sqaures problem
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
            # print("_______________NEW ITERATION: %a_______________" %iter)
            # print("\n")
            iter += 1

            # 7) Set L set of indices, where r(l) =/= 0
            L = np.nonzero(residual)[0]

            # 8) Set JTilde to all new column indices of A that appear in all L rows, but is not in J yet
            JTilde = np.array([],dtype=int)
            for l in L:
                A_l = A[[l],:]
                nonzerosInA_l = np.unique(A_l.nonzero()[1])
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
            # To be implemented in the future

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
            # 13.1) Create A(I, JTilde) and A(ITilde, JTilde)
            AIJTilde = np.zeros((len(I), len(JTilde)))
            for i in range(len(I)):
                for j in range(len(JTilde)):
                    AIJTilde[i, j] = A[I[i], JTilde[j]]

            AITildeJTilde = A[np.ix_(ITilde,JTilde)]

            # Find permutation matrices
            Pc = permutation.perm(J, n2, JTilde, n2Tilde, "col")
            Pr = permutation.perm(I, n1, ITilde, n1Tilde, "row")
            
            # 13.2) Compute ABreve = Q^T * A(I, JTilde)
            ABreve = np.dot(Q.T, AIJTilde)

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

            # set I = I U I_tilde, J = J U J_tilde
            J = unionJ
            n2 = J.size

            I = unionI
            n1 = I.size

            # 13.7) Solve the augmented LS problem for mHat_k 
            R_1 = R[0:n2,:]  

            cHat_k = Q.T * e_k[I]

            invR_1 = np.linalg.inv(R_1)

            m_k[J] = invR_1 * cHat_k[0:n2,:]

            # 14) Compute residual r
            A_JDense = A[:,J].todense()
            residual = A_JDense * m_k[J] - e_k

            # Permute Q and R to be used in next iteration
            Q = Pr * Q 
            R_1 = R_1 * Pc

        # 16) Set m_k(J) = mHat_k
        M[:,k] = m_k

        # iterate k
        k += 1
    
    return M