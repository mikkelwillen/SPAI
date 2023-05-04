#ifndef UPDATE_QR_H
#define UPDATE_QR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"
#include "permutation.cu.h"

void* updateQR(CSC* A, float* Q, float* R, int* I, int* J, int* ITilde, int* JTilde, int* IUnion, int* JUnion, int n1, int n2, int n1Tilde, int n2Tilde, int n1Union, int n2Union, float* m_kOut) {
    printf("------UPDATE QR------\n");

    // ABar = A(UnionI, UnionJ)
    float* ABar = CSCToDense(A, IUnion, JUnion, n1Union, n2Union);
    
    printf("ABar:\n");
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n2Union; j++) {
            printf("%f ", ABar[i*n2Union + j]);
        }
        printf("\n");
    }

    // Create permutation matrices Pr and Pc
    float* Pr = (float*)malloc(n1Union * n1Union * sizeof(float));
    float* Pc = (float*)malloc(n2Union * n2Union * sizeof(float));
    createPermutationMatrices(IUnion, JUnion, n1Union, n2Union, Pr, Pc);

    printf("Pr:\n");
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n1Union; j++) {
            printf("%f ", Pr[i*n1Union + j]);
        }
        printf("\n");
    }
    printf("Pc:\n");
    for (int i = 0; i < n2Union; i++) {
        for (int j = 0; j < n2Union; j++) {
            printf("%f ", Pc[i*n2Union + j]);
        }
        printf("\n");
    }

    // ATilde = Pr * ABar * Pc
    // Compute Pr * ABar
    float* ATildeTemp = (float*)malloc(n1Union * n2Union * sizeof(float));
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n2Union; j++) {
            ATildeTemp[i*n2Union + j] = 0;
            for (int k = 0; k < n1Union; k++) {
                ATildeTemp[i*n2Union + j] += Pr[i*n1Union + k] * ABar[k*n2Union + j];
            }
        }
    }

    // Compute ATilde = ATildeTemp * Pc
    float* ATilde = (float*)malloc(n1Union * n2Union * sizeof(float));
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n2Union; j++) {
            ATilde[i*n2Union + j] = 0;
            for (int k = 0; k < n2Union; k++) {
                ATilde[i*n2Union + j] += ATildeTemp[i*n2Union + k] * Pc[k*n2Union + j];
            }
        }
    }

    printf("ATilde:\n");
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n2Union; j++) {
            printf("%f ", ATilde[i*n2Union + j]);
        }
        printf("\n");
    }

    // Create AIJTilde
    float* AIJTilde = CSCToDense(A, I, JTilde, n1, n2Tilde);

    printf("AIJTilde:\n");
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            printf("%f ", AIJTilde[i*n2Tilde + j]);
        }
        printf("\n");
    }

    // ABreve = Q^T * AIJTilde (CHECK IF Q IS TRANSPOSED!!!)
    float* ABreve = (float*)malloc(n1 * n2Tilde * sizeof(float));
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            ABreve[i*n2Tilde + j] = 0;
            for (int k = 0; k < n1; k++) {
                ABreve[i*n2Tilde + j] += Q[k*n1 + i] * AIJTilde[k*n2Tilde + j];
            }
        }
    }

    printf("ABreve:\n");
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            printf("%f ", ABreve[i*n2Tilde + j]);
        }
        printf("\n");
    }

    free(ABar);
    free(Pr);
    free(Pc);
    free(ATildeTemp);
    free(ATilde);
    free(AIJTilde);
    free(ABreve);

    printf("done with updateQR\n");
}




#endif