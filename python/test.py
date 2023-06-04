import SPAI
import numpy as np
import random
import scipy
import timeit

# Function for generating an array with random 
# numbers between 0 and 99
def arrayGen(m, n):
    A = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if(random.randint(0, 1) == 1):
                A[i, j] = random.randint(0, 99)
    return A

# Test functions for SPAI with updating QR
# decomposition
# Comparison bewteen SPAI and numpy inverse
def compare(algo, A):
    spaiTest = algo(A)
    npTest = scipy.sparse.linalg.inv(A)
    diff = spaiTest - npTest
    print("testDif:\n", diff)
    print("norm:\n", np.linalg.norm(diff))

# checking the dot product of A and M with the
# identity matrix
def checkIdentity(algo, A):
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
def errorTest(algo, A):
    spaiTest = algo(A)
    AInv = scipy.sparse.linalg.inv(A)

    identitySPAI = A * spaiTest
    identityNP = A * AInv

    normISPAI = np.linalg.norm(identitySPAI)
    normINP = np.linalg.norm(identityNP)

    error = (normISPAI - normINP) / normINP
    print("Error of the SPAI algorithm: %a" % error)

# Run SPAI
def testErrorAndSpeed(algo, A):
    start = timeit.default_timer()

    M = algo(A)

    stop = timeit.default_timer()

    print('Time of SPAI: ', stop - start)  

    print("Norm of SPAI-implementation:", np.linalg.norm(A * M - np.identity(M.shape[1])))

def testofScipy(A):
    start = timeit.default_timer()

    AInv = scipy.sparse.linalg.inv(A)

    stop = timeit.default_timer()

    print('Time of Scipy: ', stop - start)  

    print("Norm of Scipy-implementation: ", np.linalg.norm(A * AInv - np.identity(AInv.shape[1])))

# For n = 10, 100, 1000, 10000, 100000:
size = [10, 100, 1000, 10000, 100000]
den = [0.1, 0.3, 0.5]
for n in size:
    for d in den:
        print("\nTesting for n = %a and density = %a" % (n, d))
        if n > 10 or d > 0.1:
            A = scipy.sparse.random(n, n, density=d, format='csc', random_state=1)

            testofScipy(A)

            testErrorAndSpeed(SPAI.SPAI, A)

        else:
            print("A is singular")

