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

void* updateQR(CSC* A, float* AHat, float* Q, float* R, int* I, int* J, int* ITilde, int* JTilde, int* IUnion, int* JUnion, int n1, int n2, int n1Tilde, int n2Tilde, int n1Union, int n2Union, float* m_kOut) {
    printf("\n------UPDATE QR------\n");

    // create AIJTilde
    float* AIJTilde = CSCToDense(A, I, JTilde, n1, n2Tilde);

    // create AITildeJTilde of size n1Tilde x n2Tilde
    float* AITildeJTilde = CSCToDense(A, ITilde, JTilde, n1Tilde, n2Tilde);

    // create ATilde of size n1Union x n2Union
    float* ATilde = (float*) malloc(n1Union * n2Union * sizeof(float));

    // set upper left square to AHat of size n1 x n2
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            ATilde[i*n2Union + j] = AHat[i*n2 + j];
        }
    }

    // set upper right square to AIJTilde of size n1 x n2Tilde
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            ATilde[i*n2Union + n2 - 1 + j] = AIJTilde[i*n2Tilde + j];
        }
    }

    // // set lower left square to zeros of size n1Tilde x n2
    // for (int i = 0; i < n1Tilde; i++) {
    //     for (int j = 0; j < n2; j++) {
    //         ATilde[(n1 + i)*n2Union + j] = 0;
    //     }
    // }

    // // set lower right square to AITildeJTilde of size n1Tilde x n2Tilde
    // for (int i = 0; i < n1Tilde; i++) {
    //     for (int j = 0; j < n2Tilde; j++) {
    //         ATilde[(n1 + i)*n2Union + n2 + j] = AITildeJTilde[i*n2Tilde + j];
    //     }
    // }

    // print ATilde
    printf("ATilde:\n");
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n2Union; j++) {
            printf("%f ", ATilde[i*n2Union + j]);
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

    // SKAL VI PERMUTERE???? ELLER ER DEN ALLEREDE PÅ DEN RIGTIGE FORM LIGESOM FIGUR 3.3 
    // ATilde = Pr * ABar * Pc
    // Compute Pr * ABar
    // float* ATildeTemp = (float*)malloc(n1Union * n2Union * sizeof(float));
    // for (int i = 0; i < n1Union; i++) {
    //     for (int j = 0; j < n2Union; j++) {
    //         ATildeTemp[i*n2Union + j] = 0;
    //         for (int k = 0; k < n1Union; k++) {
    //             ATildeTemp[i*n2Union + j] += Pr[i*n1Union + k] * ABar[k*n2Union + j];
    //         }
    //     }
    // }



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

    free(Pr);
    free(Pc);
    free(ATilde);
    free(AIJTilde);
    free(ABreve);

    printf("done with updateQR\n");
}




#endif