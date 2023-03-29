import numpy as np
import qr

def updateQR(A, Q, R, I, J, ITilde, JTilde, k):
    print("\n----UPDATEQR----")
    # 1. ABar = A(I U ITilde, J U JTilde)
    # 2. ATilde = Pr ABar Pc
    UnionI = I.copy()
    for i in ITilde:
        if i not in UnionI:
            UnionI.append(i)
    UnionI.sort()
    
    UnionJ = J.copy()
    for j in JTilde:
        if j not in UnionJ:
            UnionJ.append(j)
    UnionJ.sort()

    ATilde = np.zeros((len(UnionI), len(UnionJ)))
    for i in range(len(UnionI)):
        for j in range(len(UnionJ)):
            ATilde[i, j] = A[UnionI[i], UnionJ[j]]

    # 3. ABreve = Q^T A(I, JTilde)
    AIJTilde = np.zeros((len(I), len(JTilde)))
    print("AIJTildeShape: ", AIJTilde.shape)
    print("QShape: ", Q.shape)
    for i in range(len(I)):
        for j in range(len(JTilde)):
            AIJTilde[i, j] = A[I[i], JTilde[j]]
    ABreve = np.matmul(Q.T, AIJTilde)
    print("I: ", I)
    print("JTilde: ", JTilde)
    print("ABreve: ", ABreve)
    # 4. B1 = ABreve from 1 to q (q = |J|)
    B1 = ABreve[:len(J), :]

    # 5. B2 = (ABreve from q + 1 to p)
    #         (  A(ITilde, JTilde)   )
    AITildeJTilde = np.zeros((len(ITilde), len(JTilde)))
    for i in range(len(ITilde)):
        for j in range(len(JTilde)):
            AITildeJTilde[i, j] = A[ITilde[i], JTilde[j]]

    B2 = np.zeros((len(I) + len(ITilde) - len(J), len(JTilde)))
    for i in range(len(J), len(I)):
        for j in range(len(JTilde)):
            B2[i, j] = ABreve[i, j]
    
    for i in range(len(ITilde)):
        for j in range(len(JTilde)):
            B2[len(I) - len(J) + i, j] = AITildeJTilde[i, j]
    print("B2:", B2)
    print("B2 shape:", B2.shape)
    print("Itilde len:", len(ITilde))
    print("I + ITilde - J:", len(I)+len(ITilde)+len(J))
    
    # 6. QB, RB = qr(B2)
    QB1, RB1 = qr.qr_factorization(B2)
    QB, RB = np.linalg.qr(B2, mode="complete")

    print("QB.shape: ", QB.shape)
    # Stack matrices (from eq 17)
    unsortedQ = np.zeros((Q.shape[1] + QB.shape[0], Q.shape[1] + QB.shape[1]))
    for i in range(Q.shape[1]):
        for j in range(Q.shape[1]):
            unsortedQ[i, j] = Q[i, j]
    
    for i in range(QB.shape[0]):
        for j in range(QB.shape[1]):
            unsortedQ[Q.shape[0] + i, Q.shape[1] + j] = QB[i, j]
    
    print("UnsortedQ: ", unsortedQ)

    newQ = np.zeros((Q.shape[1] + QB.shape[0], Q.shape[1] + QB.shape[1]))
    # sort matrix
    for j in range(len(UnionJ)):
        for jj in range(len(J)):
            if UnionJ[j] == J[jj]:
                for i in range(newQ.shape[0]):
                    newQ[i, j] = unsortedQ[i, jj]
        for jj in range(len(JTilde)):
            if UnionJ[j] == JTilde[jj]:
                for i in range(newQ.shape[0]):
                    newQ[i, j] = unsortedQ[i, len(J) + jj]

    print("newQ: ", newQ)
    print("newQ - UnsortedQ: ", newQ - unsortedQ)

    # 7. MTilde = do minimization
    #  a. compute ĉ = Q^T ê_k
    print("UnionI: ", len(UnionI))
    e_kHat = np.zeros(len(UnionI))
    for i in range(len(UnionI)):
        if k == UnionI[i]:
            e_kHat[i] = 1
    e_k = np.zeros(len(UnionI))
    e_k[k] = 1
    print("e_kHat:", e_kHat)
    cHat = np.matmul(newQ.T, e_kHat)
    print("cHat:", cHat)

    #  b. compute ^m_k = R^-1 ĉ
    mHat_k = np.matmul(np.linalg.inv(RB1), cHat)
    print("cHat:", cHat)

    # 9. return m_k, new I and new J
    return mHat_k, I, J