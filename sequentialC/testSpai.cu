#include <stdio.h>
#include <stdlib.h>
#include "csc.cu.h"
#include "constants.cu.h"
#include "sequentialSpai.cu.h"
#include "sequentialTest.cu.h"
#include "qrBatched.cu.h"
#include "invBatched.cu.h"
#include "permutation.cu.h"
#include "updateQR.cu.h"
#include "cuSOLVERInv.cu.h"

/*
The main function for running the test
If arguments are given in the terminal, they will be used for the test
*/
int main(int argc, char** argv) {
    if (argc == 1) {
        initHwd();
        int m = 100;
        int n = m;
        float sparsity = 0.1;
        float tolerance = 0.001;
        int maxIterations = n - 1;
        int s = 1;

        // create a random matrix
        CSC* csc = createRandomCSC(m, n, sparsity);

        runIdentityTest(csc, m, n, sparsity, tolerance, maxIterations, s);
    } else if (argc == 7) {
        // Read argumnets from terminal
        int sizeOfMatrix = atoi(argv[1]);
        int numberOfTests = atoi(argv[2]);
        float sparsity = atof(argv[3]);
        float tolerance = atof(argv[4]);
        int maxIterations = atoi(argv[5]);
        int s = atoi(argv[6]);

        printf("sizeOfMatrix: %d\n", sizeOfMatrix);
        printf("numberOfTests: %d\n", numberOfTests);
        printf("sparsity: %f\n", sparsity);
        printf("tolerance: %f\n", tolerance);
        printf("maxIterations: %d\n", maxIterations);
        printf("s: %d\n", s);

        for (int i = 0; i < numberOfTests; i++) {
            CSC* csc = createRandomCSC(sizeOfMatrix, sizeOfMatrix, sparsity);

            sequentialTest(csc, tolerance, maxIterations, s);
        }
    }

    return 0;
}