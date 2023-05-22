import numpy as np

def rowPermutation(I):
    P = np.zeros((len(I), len(I)))
    print("Prow size: ", P.shape)
    print("len of I:", len(I))
    I = list(I)
    for i, row in enumerate(I): 
        P[i, row] = 1
    print("P:\n", P)
    return P


def columnPermutation(J):
    P = np.zeros((len(J), len(J)))
    print("Pcol size: ", P.shape)
    J = list(J)
    for j, col in enumerate(J):
        P[col, j] = 1
    print("P:\n", P)
    return P

def permutation(set, mode):
    setaslist = list(zip(list(np.arange(0, len(set))), list(set)))
    sort = sorted(setaslist, key = lambda x: x[1])
    swaps, rest = [[i for i, j in sort], [j for j, j in sort]]
    if mode == "col":
        P = np.identity(len(set))[:, swaps]
    elif "row":
        P = np.identity(len(set))[swaps, :]
    return P

def perm(set, n, settilde, ntilde, mode):
    setsettilde = list(zip(list(np.arange(0,n + ntilde)),list(set) + list(settilde)))
    sor = sorted(setsettilde, key=lambda x: x[1])
    swaps, rest = [[i for i, j in sor], [j for i, j in sor]]
    if mode == "col":
        P = np.identity(n + ntilde)[:,swaps]
    elif mode == "row":
        P = np.identity(n + ntilde)[swaps,:]
    return P


def perm1(input_set, n, input_settilde, ntilde, mode):
    setsettilde = np.vstack((np.arange(0, n + ntilde), np.concatenate((input_set, input_settilde))))
    sor_indices = np.lexsort(setsettilde[::-1])
    swaps = sor_indices[:n]
    rest = sor_indices[n:]
    if mode == "col":
        P = np.eye(n + ntilde)[:, swaps]
    elif mode == "row":
        P = np.eye(n + ntilde)[swaps, :]
    return P




# Testing
# A = np.array([[5, 9, 1], [2, 8, 7], [6, 3, 4]])
# I = [2, 0, 1]
# Pr = rowPermutation(I)
# sortedRowsA = Pr.dot(A)
# print("Sorted rows:\n", sortedRowsA)

# J = [2, 0, 1]
# Pc = columnPermutation(J)
# sortedColsA = A.dot(Pc)
# print("Sorted cols:\n", sortedColsA)

# sortedA = (Pr.dot(A)).dot(Pc)
# print("Sorted:\n", sortedA)
