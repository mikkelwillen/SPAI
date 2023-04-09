import numpy as np

def rowPermutation(I):
    P = np.zeros((len(I), len(I)))
    print("Prow size: ", P.shape)
    print("len of I:", len(I))
    I = list(I)
    for i, row in enumerate(I): # and another way
        P[i, row] = 1
    print("P:\n", P)
    return P


def columnPermutation(J):
    P = np.zeros((len(J), len(J)))
    print("Pcol size: ", P.shape)
    J = list(J)
    for j, col in enumerate(J): # and another way
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
