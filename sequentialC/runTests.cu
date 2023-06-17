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


int main(int argc, char** argv) {
    initHwd();

    // Set tolerance and s to defualt values
    float tolerance = 0.01;
    int s = 1;

    // Test for different sizes and sparsity
    int* sizeArray = (int*) malloc(sizeof(int) * 5);
    sizeArray[0] = 10;
    sizeArray[1] = 100;
    sizeArray[2] = 200;
    sizeArray[3] = 300;
    sizeArray[4] = 400;
    sizeArray[5] = 500;
    sizeArray[6] = 1000;
    sizeArray[7] = 1500;
    sizeArray[8] = 2000;
    sizeArray[9] = 2500;
    sizeArray[10] = 3000;
    sizeArray[11] = 3500;
    sizeArray[12] = 4000;
    sizeArray[13] = 4500;
    sizeArray[14] = 5000;

    float* sparsityArray = (float*) malloc(sizeof(float) * 3);
    sparsityArray[0] = 0.1;
    sparsityArray[1] = 0.3;
    sparsityArray[2] = 0.5;

    for (int i = 0; i < 15; i++) {
        for (int j = 1; j < 2; j++) {
            int n = sizeArray[i];
            float sparsity = sparsityArray[j];
            int maxIterations = n - 1;

            printf("\n\nStarting test for size: %d and sparsity: %f \n", n, sparsity);
            printf("Running sequential SPAI test\n");
            float* test = (float*) malloc(sizeof(float) * n * n);
            CSC* cscTest = createRandomCSC(n, n, sparsity);
            sequentialTest(cscTest, tolerance, maxIterations, s);

            // Kør cuSOLVER test på samme
            printf("Running cuSOLVER test\n");
            CSC* cscTest2 = createRandomCSC(n, n, sparsity);

            int* I = (int*) malloc(sizeof(int) * (n));
            int* J = (int*) malloc(sizeof(int) * (n));

            for (int i = 0; i < n; i++) {
                I[i] = i;
                J[i] = i;
            }

            float* test2Dense = CSCToDense(cscTest2, I, J, n, n);
            sequentialTestCuSOLVER(test2Dense, n);
        }
    }
}