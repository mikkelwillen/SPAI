# function for testing the inverse of a matrix
# takes executable as input
# returns latex code for a table of tests and a graph of the results

import sys
import os

def main(argv):
    print(argv[0])

    sizeOfMatrix = 5
    numberOfTests = 5
    sparsity = 0.1
    tolerance = 0.0001
    maxIterations = 5
    s = 1
    curDir = os.getcwd()
    srcPath = curDir.rfind("tests")

    if srcPath > 0:
        basePath = curDir[0:srcPath]
    
    print(basePath)

    os.system("make compile -C " + basePath + argv[0]+ "/")
    os.system("./../" + argv[0] + "/testSpai " + str(sizeOfMatrix) + " " + str(numberOfTests) + " " + str(sparsity) + " " + str(tolerance) + " " + str(maxIterations) + " " + str(s))

    # print tallene råt i seqTest
    # tag output fra seqTest og lav til latex tabel
    # lav graf ud fra output fra seqTest
    
if __name__ == "__main__":
   main(sys.argv[1:])