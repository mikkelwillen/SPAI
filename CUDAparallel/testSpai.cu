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


int runIdentityTest(CSC* cscA, int m, int n, float sparsity, float tolerance, int maxIterations, int s, int batchsize) {
    float* identity = (float*) malloc (sizeof(float) * n * n);

    struct CSC* res = parallelSpai(cscA, tolerance, maxIterations, s, batchsize);
    printf("After parallelSpai\n");
    int* I = (int*) malloc(sizeof(int) * m);
    int* J = (int*) malloc(sizeof(int) * n);
    for (int i = 0; i < m; i++) {
        I[i] = i;
    }
    for (int i = 0; i < n; i++) {
        J[i] = i;
    }

    float* A = CSCToDense(cscA, I, J, m, n);
    float* inv = CSCToDense(res, I, J, m, n);
    
    // identity = A * inv
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n;j++) {
            identity[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                identity[i * n + j] += A[i * n + k] * inv[k * n + j];
            }
        }
    }

    // print A
    printf("A:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n;j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
    }

    // print inv
    printf("inv:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n;j++) {
            printf("%f ", inv[i * n + j]);
        }
        printf("\n");
    }

    // print identity
    printf("identity:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j++) {
            printf("%f ", identity[i * n + j]);
        }
        printf("\n");
    }

    // calculate error
    float error = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j++) {
            error += (identity[i * n + j] - (i == j ? 1.0 : 0.0)) * (identity[i * n + j] - (i == j ? 1.0 : 0.0));
        }
    }

    printf("Error: %f%\n", error);
}

int main(int argc, char** argv) {
    if (argc == 1) {
        initHwd();
        int m = 4;
        int n = 16;
        float sparsity = 1.0;
        float tolerance = 0.01;
        int maxIterations = n - 1;
        int s = 1;
        int batchsize = 4;
    
    
    
        float* A = (float*) malloc(sizeof(float) * m * n);
        float* B = (float*) malloc(sizeof(float) * m * n);
        float* C = (float*) malloc(sizeof(float) * n * n);
        float* m4 = (float*) malloc(sizeof(float) * 4 * 4);
    
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
        
        m4[0] = 10.0; m4[1] = 10.0; m4[2] = 1.2; m4[3] = 14.0;
        m4[4] = 0.0; m4[5] = 10.0; m4[6] = 2.0; m4[7] = 0.0;
        m4[8] = 13.0; m4[9] = 0.0; m4[10] = 5.3; m4[11] = 1.0;
        m4[12] = 0.0; m4[13] = 5.0; m4[14] = 0.0; m4[15] = 0.0;

        struct CSC* cscA = createCSC(A, m, n);
        struct CSC* cscB = createCSC(B, m, n);
        struct CSC* cscC = createRandomCSC(n, n, sparsity);
        struct CSC* cscD = createCSC(C, n, n);
        struct CSC* cscM4 = createCSC(m4, 4, 4);

        runIdentityTest(cscC, n, n, sparsity, tolerance, maxIterations, s, batchsize);
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