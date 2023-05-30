import spai
import spaiUpdate
import numpy as np
import random
import improvedSpai
import scipy

# Function for generating an array with random 
# numbers between 0 and 99
def arrayGen(m, n):
    A = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if(random.randint(0, 1) == 1):
                A[i, j] = random.randint(0, 99)
    return A

# Test functions for SPAI with normal QR
# decomposition
# Comparison bewteen SPAI and numpy inverse
def compare(algo, A):
    spaiTest = algo(A)
    npTest = np.linalg.inv(A)
    diff = spaiTest - npTest
    print("testDif:\n", diff)
    print("norm:\n", np.linalg.norm(diff))

# checking the dot product of A and M with the
# identity matrix
def checkIdentity(algo, A):
    spaiTest = algo(A)
    print("shapeA:\n %a, shapeM:\n %a" %(A.shape, spaiTest.shape))
    identity = np.matmul(A, spaiTest)
    print("identity:\n", identity)
    print("normI:\n", np.linalg.norm(identity))
    print("shape:\n", identity.shape)

# Check the error difference between numpy inverse
# and SPAI by computing the dot product of A and M
# for both of them and checking how close they are
# the identity matrix
def errorTest(algo, A):
    spaiTest = algo(A)
    npTest = np.linalg.inv(A)

    identitySPAI = np.matmul(A, spaiTest)
    identityNP = np.matmul(A, npTest)

    normISPAI = np.linalg.norm(identitySPAI)
    normINP = np.linalg.norm(identityNP)

    error = (normISPAI - normINP) / normINP
    print("Error of the SPAI algorithm: ", error)

# Run SPAI
def test(algo, A):
    algo(A)


# Test functions for SPAI with updating QR
# decomposition
# Comparison bewteen SPAI and numpy inverse
def compareU(algo, A):
    spaiTest = algo(A)
    npTest = scipy.sparse.linalg.inv(A)
    diff = spaiTest - npTest
    print("testDif:\n", diff)
    print("norm:\n", np.linalg.norm(diff))

# checking the dot product of A and M with the
# identity matrix
def checkIdentityU(algo, A):
    spaiTest = algo(A)
    print("shapeA:\n %a, \nshapeM:\n %a" %(A.shape, spaiTest.shape))
    identity = A * spaiTest
    print("identity:\n", identity)
    for i in range(identity.shape[0]):
        for j in range(identity.shape[1]):
            if i == j:
                print("(%a, %a) = %a" % (i, j, identity[i,j]))
    print("normI:\n", np.linalg.norm(identity))
    print("shape:\n", identity.shape)

    print("Norm of implementation:", np.linalg.norm(A * spaiTest - np.identity(spaiTest.shape[1])))

    print("Norm of library-implementation: ", np.linalg.norm(A * AInv - np.identity(spaiTest.shape[1])))

# Check the error difference between numpy inverse
# and SPAI by computing the dot product of A and M
# for both of them and checking how close they are
# the identity matrix
def errorTestU(algo, A):
    spaiTest = algo(A)
    AInv = scipy.sparse.linalg.inv(A)

    identitySPAI = A * spaiTest
    identityNP = A * AInv

    normISPAI = np.linalg.norm(identitySPAI)
    normINP = np.linalg.norm(identityNP)

    error = (normISPAI - normINP) / normINP
    print("Error of the SPAI algorithm: %a" % error)

# Run SPAI
def testU(algo, A):
    M = algo(A)

    print("Norm of implementation:", np.linalg.norm(A * M - np.identity(M.shape[1])))

    print("Norm of library-implementation: ", np.linalg.norm(A * AInv - np.identity(M.shape[1])))

A = scipy.sparse.random(10, 10, density=0.5, format='csr', random_state=1)
AInv = scipy.sparse.linalg.inv(A)

B = [[0, 0, 24.1, 0, 0, 0, 0, 0, 61.24, 13.48], 
     [0, 45.95, 0, 0, 85.9, 0, 67.39, 0, 0, 97.53],
     [0, 0, 0, 0, 0, 33.72, 10.06, 87.5, 36.03, 0],
     [0, 0, 0, 46.05, 0, 0, 0, 0, 0, 0],
     [19.81, 0, 0, 0, 62.48, 0, 0, 65.23, 0, 0],
     [0, 0, 19.94, 87.49, 0, 0, 0, 0, 57.64, 0],
     [0, 0, 26.07, 0, 0, 0, 0, 51.20, 0, 0],
     [0, 0, 3.61, 0, 93.12, 0, 0, 0, 0, 68.28],
     [0, 0, 0, 72.09, 0, 0, 0, 0, 0, 28.52]]

C = [[0, 0, 24.1, 0, 0, 0, 0, 0, 61.24, 13.48],
     [0, 45.95, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 85.9, 0, 67.39, 0, 0, 97.53],
     [0, 0, 0, 0, 0, 33.72, 10.06, 87.5, 36.03, 0],
     [0, 0, 0, 46.05, 0, 0, 0, 0, 0, 0],
     [19.81, 0, 0, 0, 62.48, 0, 0, 65.23, 0, 0],
     [0, 0, 19.94, 87.49, 0, 0, 0, 0, 57.64, 0],
     [0, 0, 26.07, 0, 0, 0, 0, 51.20, 0, 0],
     [0, 0, 3.61, 0, 93.12, 0, 0, 0, 0, 68.28],
     [0, 0, 0, 72.09, 0, 0, 0, 0, 0, 28.52]
     ]

D = [[20, 0, 0],
     [0, 30, 0],
     [25, 0, 10]]

DSparse = scipy.sparse.csr_array(D)
CSparse = scipy.sparse.csr_array(C)

checkIdentityU(improvedSpai.SPAI, A)

