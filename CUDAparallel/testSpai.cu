#include <stdio.h>
#include <stdlib.h>
#include "csc.cu.h"
#include "constants.cu.h"
#include "parallelSpai.cu.h"
#include "parallelTest.cu.h"
#include "qrBatched.cu.h"
#include "invBatched.cu.h"
#include "permutation.cu.h"
#include "updateQR.cu.h"


int main(int argc, char** argv) {
    if (argc == 1) {
        initHwd();
        int m = 4;
        int n = 3;
    
    
    
        float* A = (float*) malloc(sizeof(float) * m * n);
        float* B = (float*) malloc(sizeof(float) * m * n);
        float* C = (float*) malloc(sizeof(float) * n * n);
    
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = (float) i * n + j;
            }
        }
    
        C[0] = 20.0; C[1] = 0.0;   C[2] = 0.0; 
        C[3] = 0.0;  C[4] = 30.0;  C[5] = 10.0; 
        C[6] = 0.0;  C[7] = 0.0;   C[8] = 10.0;
    
        B[0] = 20.0; B[1] = 0.0;   B[2] = 0.0; 
        B[3] = 0.0;  B[4] = 30.0;  B[5] = 10.0; 
        B[6] = 0.0;  B[7] = 0.0;   B[8] = 0.0; 
        B[9] = 0.0;  B[10] = 40.0; B[11] = 0.0;
    
        struct CSC* cscA = createCSC(A, m, n);
        struct CSC* cscB = createCSC(B, m, n);
        struct CSC* cscC = createRandomCSC(10, 10, 0.2);
        struct CSC* cscD = createCSC(C, n, n);

        parallelSpai(cscC, 0.1, 2, 1, 2);
    } else if (argc == 8) {
        // read args
        printf("hallo?\n");
        int sizeOfMatrix = atoi(argv[1]);
        int numberOfTests = atoi(argv[2]);
        float sparsity = atof(argv[3]);
        float tolerance = atof(argv[4]);
        int maxIterations = atoi(argv[5]);
        int s = atoi(argv[6]);
        int batchsize = atoi(argv[7]);

        printf("sizeOfMatrix: %d\n", sizeOfMatrix);
        printf("numberOfTests: %d\n", numberOfTests);
        printf("sparsity: %f\n", sparsity);
        printf("tolerance: %f\n", tolerance);
        printf("maxIterations: %d\n", maxIterations);
        printf("s: %d\n", s);

        for (int i = 0; i < numberOfTests; i++) {
            CSC* csc = createRandomCSC(sizeOfMatrix, sizeOfMatrix, sparsity);

            parallelTest(csc, tolerance, maxIterations, s, batchsize);
        }
    }

    return 0;
}