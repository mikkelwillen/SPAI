import spai
import spaiUpdate
import numpy as np
import random

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
def compare(A):
    spaiTest = spai.SPAI(A)
    npTest = np.linalg.inv(A)
    diff = spaiTest - npTest
    print("testDif:\n", diff)
    print("norm:\n", np.linalg.norm(diff))

# checking the dot product of A and M with the
# identity matrix
def checkIdentity(A):
    spaiTest = spai.SPAI(A)
    print("shapeA:\n %a, shapeM:\n %a" %(A.shape, spaiTest.shape))
    identity = np.matmul(A, spaiTest)
    print("identity:\n", identity)
    print("normI:\n", np.linalg.norm(identity))
    print("shape:\n", identity.shape)

# Check the error difference between numpy inverse
# and SPAI by computing the dot product of A and M
# for both of them and checking how close they are
# the identity matrix
def errorTest(A):
    spaiTest = spai.SPAI(A)
    npTest = np.linalg.inv(A)

    identitySPAI = np.matmul(A, spaiTest)
    identityNP = np.matmul(A, npTest)

    normISPAI = np.linalg.norm(identitySPAI)
    normINP = np.linalg.norm(identityNP)

    error = (normISPAI - normINP) / normINP
    print("Error of the SPAI algorithm: ", error)

# Run SPAI
def test(A):
    spai.SPAI(A)


# Test functions for SPAI with updating QR
# decomposition
# Comparison bewteen SPAI and numpy inverse
def compareU(A):
    spaiTest = spaiUpdate.SPAI(A)
    npTest = np.linalg.inv(A)
    diff = spaiTest - npTest
    print("testDif:\n", diff)
    print("norm:\n", np.linalg.norm(diff))

# checking the dot product of A and M with the
# identity matrix
def checkIdentityU(A):
    spaiTest = spaiUpdate.SPAI(A)
    print("shapeA:\n %a, shapeM:\n %a" %(A.shape, spaiTest.shape))
    identity = np.matmul(A, spaiTest)
    print("identity:\n", identity)
    print("normI:\n", np.linalg.norm(identity))
    print("shape:\n", identity.shape)

# Check the error difference between numpy inverse
# and SPAI by computing the dot product of A and M
# for both of them and checking how close they are
# the identity matrix
def errorTestU(A):
    spaiTest = spaiUpdate.SPAI(A)
    npTest = np.linalg.inv(A)

    identitySPAI = np.matmul(A, spaiTest)
    identityNP = np.matmul(A, npTest)

    normISPAI = np.linalg.norm(identitySPAI)
    normINP = np.linalg.norm(identityNP)

    error = (normISPAI - normINP) / normINP
    print("Error of the SPAI algorithm: %a", error)

# Run SPAI
def testU(A):
    spaiUpdate.SPAI(A)

test1 = arrayGen(10, 10)
errorTestU(test1)
