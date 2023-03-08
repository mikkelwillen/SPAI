import spai
import numpy as np
import random

def arrayGen(m, n):
    A = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if(random.randint(0, 1) == 1):
                A[i, j] = random.randint(0, 99)
    return A

def compare(A):
    spaiTest = spai.SPAI(A)
    npTest = np.linalg.inv(A)
    diff = spaiTest - npTest
    print("testDif:\n", diff)
    print("norm:\n", np.linalg.norm(diff))

def checkIdentity(A):
    spaiTest = spai.SPAI(A)
    identity = np.matmul(A, spaiTest)
    print("identity:\n", identity)
    print("normI:\n", np.linalg.norm(identity))

def test(A):
    spai.SPAI(A)

test1 = arrayGen(10, 10)
checkIdentity(test1)
