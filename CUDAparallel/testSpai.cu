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
    
    
    
    
    
    
        // // test of multiply
        // CSC* cscD = multiplyCSC(cscB, cscC);
        // float* denseB = CSCToDense(cscB);
        // float* denseC = CSCToDense(cscC);
        // float* denseD = (float*) malloc(sizeof(float) * m * n);
        // // multiply denseB with denseC
        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < n; j++){
        //         denseD[i * n + j] = 0.0;
        //         for (int k = 0; k < n; k++) {
        //             denseD[i * n + j] += denseB[i * n + k] * denseC[k * n + j];
        //         }
        //     }
        // }
    
        // float* CSCDdense = CSCToDense(cscD);
        // // print denseD
        // printf("denseD:\n");
        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < n; j++){
        //         printf("%f ", denseD[i * n + j]);
        //     }
        //     printf("\n");
        // }
    
        // print CSCDdense
        // printf("CSCDdense:\n");
        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < n; j++){
        //         printf("%f ", CSCDdense[i * n + j]);
        //     }
        //     printf("\n");
        // }
    
        // struct CSC* cscDia = createDiagonalCSC(m, n);
        // printCSC(cscA);
        // printCSC(cscB);
        // printCSC(cscDia);
        // struct CSC* M = sequentialSpai(cscB, 0.01, 5, 1);
        // printCSC(M);
    
        // sequentialTest(cscB);
    
        // float* Q = (float*) malloc(m * m * sizeof(float));
        // float* R = (float*) malloc(sizeof(float) * m * n);
        // float* invC = (float*) malloc(sizeof(float) * n * n);
    
        // invBatched(C, n, invC);
    
        // // print invC
        // printf("invC:\n");
        // for (int i = 0; i < n; i++) {
        //     for (int j = 0; j < n; j++){
        //         printf("%f ", invC[i * n + j]);
        //     }
        //     printf("\n");
        // }
    
        // // TEST PERMUTATIONS
        // float* Pr = (float*) malloc(sizeof(float) * m * m);
        // float* Pc = (float*) malloc(sizeof(float) * n * n);
        // int* I = (int*) malloc(sizeof(int) * m);
        // I[0] = 1; I[1] = 0; I[2] = 2; I[3] = 3;
        // int* J = (int*) malloc(sizeof(int) * n);
        // J[0] = 1; J[1] = 0; J[2] = 2;
        // createPermutationMatrices(I, J, m, n, Pr, Pc);
        // float* switchRows = (float*) malloc(sizeof(float) * m * n);
        // float* switchCols = (float*) malloc(sizeof(float) * m * n);
        // // compute switchRows = Pr * A
        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < n; j++){
        //         switchRows[i * n + j] = 0.0;
        //         for (int k = 0; k < m; k++) {
        //             switchRows[i * n + j] += Pr[i * m + k] * A[k * n + j];
        //         }
        //     }
        // }
        // // compute switchCols = A * Pc
        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < n; j++){
        //         switchCols[i * n + j] = 0.0;
        //         for (int k = 0; k < n; k++) {
        //             switchCols[i * n + j] += A[i * n + k] * Pc[k * n + j];
        //         }
        //     }
        // }
        // //print switchRows
        // printf("switchRows:\n");
        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < n; j++){
        //         printf("%f ", switchRows[i * n + j]);
        //     }
        //     printf("\n");
        // }
        // //print switchCols
        // printf("switchCols:\n");
        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < n; j++){
        //         printf("%f ", switchCols[i * n + j]);
        //     }
        //     printf("\n");
        // }
        
    
        
        struct CSC* res = parallelSpai(cscD, 0.01, 2, 1);
        printf("hallo?\n");
    
        int* I = (int*) malloc(sizeof(int) * 3);
        int* J = (int*) malloc(sizeof(int) * 3);
        for (int i = 0; i < 3; i++) {
            I[i] = i;
            J[i] = i;
        }
        float* CDense = CSCToDense(cscC, I, J, 3, 3);
        float* resDense = CSCToDense(res, I, J, 3, 3);
        
        // multiply CDense with resDense
        float* identity = (float*) malloc(sizeof(float) * 3 * 3);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3;j++) {
                identity[i * 3 + j] = 0.0;
                for (int k = 0; k < 3; k++) {
                    identity[i * 3 + j] += CDense[i * 3 + k] * resDense[k * 3 + j];
                }
            }
        }
    
        printf("identity:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3;j++) {
                printf("%f ", identity[i * 3 + j]);
            }
            printf("\n");
        }
    
        // free memory
        freeCSC(cscA);
        freeCSC(cscB);
        freeCSC(cscC);
        freeCSC(res);
    
        free(A);
        free(B);
        free(C);
    
    
        // qrBatched(B, m, n, Q, R);
    
        // // compute QR
        // float* QR = (float*) malloc(sizeof(float) * m * n);
        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < n; j++){
        //         QR[i * n + j] = 0.0;
        //         for (int k = 0; k < m; k++) {
        //             QR[i * n + j] += Q[i * m + k] * R[k * n + j];
        //         }
        //     }
        // }
        
        // // print QR
        // printf("QR:\n");
        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < n; j++){
        //         printf("%f ", QR[i * n + j]);
        //     }
        //     printf("\n");
        // }
    } else if (argc == 7) {
        // read args
        printf("hallo?\n");
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

            parallelTest(csc, tolerance, maxIterations, s);
        }
    }

    return 0;
}