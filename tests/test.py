# function for testing the inverse of a matrix
# takes executable as input
# returns latex code for a table of tests and a graph of the results

import sys
import os
import subprocess
import array_to_latex as a2l
import numpy as np
import matplotlib.pyplot as plt

def main(argv):
    print(argv[0])

    sizeOfMatrix = 5
    numberOfTests = 5
    sparsity = 0.1
    tolerance = 0.0001
    maxIterations = 5
    s = 1

    # find path to executable
    curDir = os.getcwd()
    srcPath = curDir.rfind("tests")

    if srcPath > 0:
        basePath = curDir[0:srcPath]

    # compile executable
    os.system("make compile -C " + basePath + argv[0] + "/")

    # set command to run executable
    command = ["./../" + argv[0] + "/testSpai", str(sizeOfMatrix), str(numberOfTests), str(sparsity), str(tolerance), str(maxIterations), str(s)]

    # run executable and save output to variable
    output = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0]
    print(output)


    # test array, replace with data from output
    testArray = np.zeros((numberOfTests, 3))
    for i in range(numberOfTests):
        for j in range(3):
            testArray[i, j] = i * 3 + j
    
    
    # make graph of test results
    plt.plot(testArray[:,0], testArray[:,1], linestyle="-", color="blue", label= "test1")
    plt.plot(testArray[:,0], testArray[:,2], linestyle="-", color="red", label= "test2")
    plt.xlabel("matrix size")
    plt.ylabel("time")
    plt.title("testing of SPAI")
    plt.show()

    # make latex code for table of test results
    a2l.to_ltx(testArray, frmt = '{:6.4f}', arraytype = 'array', print_out = True)


    # print tallene råt i seqTest
    # tag output fra seqTest og lav til latex tabel
    # lav graf ud fra output fra seqTest
    
if __name__ == "__main__":
   main(sys.argv[1:])