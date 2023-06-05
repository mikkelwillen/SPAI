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

        // Set tolerance and s to defualt values
        float tolerance = 0.01;
        int s = 1;

        // Test for different sizes and sparsity
        int* sizeArray = (int*) malloc(sizeof(int) * 5);
        sizeArray[0] = 10;
        sizeArray[1] = 100;
        sizeArray[2] = 1000;

        float* sparsityArray = (float*) malloc(sizeof(float) * 3);
        sparsityArray[0] = 0.1;
        sparsityArray[1] = 0.3;
        sparsityArray[2] = 0.5;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int n = sizeArray[i];
                float sparsity = sparsityArray[j];
                int maxIterations = n - 1;

                printf("\n\nStarting test for size: %d and sparsity: %f \n", n, sparsity);
                printf("Running sequential SPAI test\n");
                float* test = (float*) malloc(sizeof(float) * n * n);
                CSC* cscTest = createRandomCSC(n, n, sparsity);
                sequentialTest(cscTest, tolerance, maxIterations, s);
            }
        }

    

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