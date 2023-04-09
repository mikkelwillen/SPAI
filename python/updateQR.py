import numpy as np
import qr
import permutation

def updateQR(A, Q, R, I, J, ITilde, JTilde, k):
    print("\n----UPDATEQR----")
    # 1. ABar = A(I U ITilde, J U JTilde)
    # 2. ATilde = Pr ABar Pc

    
    UnionI = I.copy()
    for i in ITilde:
        if i not in UnionI:
            UnionI.append(i)
    SortedUnionI =  UnionI.sort()
    
    UnionJ = J.copy()
    for j in JTilde:
        if j not in UnionJ:
            UnionJ.append(j)
    SortedUnionJ =  UnionJ.sort()

    # ATilde = np.zeros((len(UnionI), len(UnionJ)))
    # for i in range(len(UnionI)):
    #     for j in range(len(UnionJ)):
    #         ATilde[i, j] = A[UnionI[i], UnionJ[j]]

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
    QB, RB = np.linalg.qr(B2, mode="complete")
    

    print("QB.shape: ", QB.shape)
    # Stack matrices (from eq 17)
    unsortedQ = np.zeros((len(UnionI), len(UnionJ)))
    for i in range(len(J)):
        for j in range(len(J)):
            unsortedQ[i, j] = Q[i, j]
    
    for i in range(len(UnionI) - len(J)):
        for j in range(len(JTilde)):
            unsortedQ[len(J) + i, len(J) + j] = QB[i, j]
    
    print("UnsortedQ: ", unsortedQ)

    # sort Q with permutation matrices
    Pr = permutation.permutation(UnionI, "row")
    print("Pr:", Pr)
    Pc = permutation.permutation(UnionJ, "col")
    print("Pc: ", Pc)

    newQ = np.matmul(Pr, unsortedQ)

    # newQ = np.zeros((len(UnionI), len(UnionJ)))
    # # sort matrix
    # for j in range(len(UnionJ)):
    #     for jj in range(len(J)):
    #         if UnionJ[j] == J[jj]:
    #             for i in range(newQ.shape[0]):
    #                 newQ[i, j] = unsortedQ[i, jj]
    #     for jj in range(len(JTilde)):
    #         if UnionJ[j] == JTilde[jj]:
    #             for i in range(newQ.shape[0]):
    #                 newQ[i, j] = unsortedQ[i, len(J) + jj]

    # print("newQ: ", newQ)
    # print("newQ - UnsortedQ: ", newQ - unsortedQ)

    newR = np.hstack((np.vstack((R, np.zeros((len(UnionI) - len(J), len(J))))), np.vstack((B1, RB))))
    newR = np.matmul(newR, Pc)

    RB1 = newR[0:len(UnionJ),:]
    # 7. MTilde = do minimization
    #  a. compute ĉ = Q^T ê_k
    print("UnionI: ", UnionI)
    print("UnionJ: ", UnionJ)
    e_kHat = np.zeros(len(UnionI))
    for i in range(len(UnionI)):
        if k == UnionI[i]:
            e_kHat[i] = 1
    e_kHat = np.matmul(Pr.T, e_kHat)

    # e_k = np.zeros(len(UnionI))
    # e_k[k] = 1
    print("e_kHat:", e_kHat)

    cHat = np.matmul(newQ.T, e_kHat)
    print("cHat:", cHat)

    #  b. compute ^m_k = R^-1 ĉ
    mHat_k = np.matmul(np.linalg.inv(RB1), cHat)
    print("mHat_k:", mHat_k)

    print(UnionJ)
    # 9. return m_k, new I and new J
    return mHat_k, UnionI, UnionJ, newQ, newR


# nogen gange er Jtilde uden elementer, og det fucker derfor på linje 102. Vi burde egentligt ikke kunne opdatere noget, hvis Jtilde er tom
# vi bruger pt. ikke s og tilføjer bare alle elementer, som ikke allerede er med.