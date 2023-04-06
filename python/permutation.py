import numpy as np

def rowPermutation(A, I):
    P = np.zeros_like(A)
    P[np.array(list(I)), np.arange(len(I))] = 1 # one way of doing it
    print("P:\n", P)
    return P


def columnPermutation(A, J):
    P = np.zeros_like(A)
    J = list(J)
    for j, col in enumerate(J): # and another way
        P[col, j] = 1
    print("P:\n", P)
    return P


# Testing
A = np.array([[5, 9, 1], [2, 8, 7], [6, 3, 4]])
I = [2, 0, 1]
Pr = rowPermutation(A, I)
sortedRowsA = Pr.dot(A)
print("Sorted rows:\n", sortedRowsA)

J = [2, 0, 1]
Pc = columnPermutation(A, J)
sortedColsA = A.dot(Pc)
print("Sorted cols:\n", sortedColsA)

sortedA = (Pr.dot(A)).dot(Pc)
print("Sorted:\n", sortedA)
