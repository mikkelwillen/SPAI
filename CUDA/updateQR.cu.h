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
#include "LSProblem.cu.h"

void* updateQR(cublasHandle_t cHandle, CSC* A, float* AHat, float* Q, float* R, int* I, int* J, int* ITilde, int* JTilde, int* IUnion, int* JUnion, int n1, int n2, int n1Tilde, int n2Tilde, int n1Union, int n2Union, float* m_kOut, float* residual, float residualNorm, int k) {
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
            ATilde[i*n2Union + n2 + j] = AIJTilde[i*n2Tilde + j];
        }
    }

    // set lower left square to zeros of size n1Tilde x n2
    for (int i = 0; i < n1Tilde; i++) {
        for (int j = 0; j < n2; j++) {
            ATilde[(n1 + i)*n2Union + j] = 0;
        }
    }

    // set lower right square to AITildeJTilde of size n1Tilde x n2Tilde
    for (int i = 0; i < n1Tilde; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            ATilde[(n1 + i)*n2Union + n2 + j] = AITildeJTilde[i*n2Tilde + j];
        }
    }

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

    // ABreve of size n1 x n2Tilde = Q^T * AIJTilde
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

    // B1 = ABreve[0 : n2, 0 : n2]
    float* B1 = (float*) malloc(n2 * n2Tilde * sizeof(float));
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            B1[i*n2Tilde + j] = ABreve[i*n2Tilde + j];
        }
    }

    // B2 = ABreve[n2 + 1 : n1, 0 : n2Tilde] + AITildeJTilde
    float* B2;
    if (n1 - n2 < 0) {
        B2 = (float*) malloc(n1Tilde * n2Tilde * sizeof(float));
        for (int i = 0; i < n1Tilde; i++) {
            for (int j = 0; j < n2Tilde; j++) {
                B2[i*n2Tilde + j] += AITildeJTilde[i*n2Tilde + j];
            }
        }
    } else {
        B2 = (float*) malloc((n1Union - n2) * n2Tilde * sizeof(float));
        for (int i = 0; i < n1 - n2; i++) {
            for (int j = 0; j < n2Tilde; j++) {
                B2[i*n2Tilde + j] = ABreve[(n1 - n2 + i)*n2Tilde + j];
            }
        }
        for (int i = 0; i < n1Tilde; i++) {
            for (int j = 0; j < n2Tilde; j++) {
                B2[(n1 - n2 + i)*n2Tilde + j] += AITildeJTilde[i*n2Tilde + j];
            }
        }
    }
    

    // print B2
    printf("B2:\n");
    for (int i = 0; i < n1Union - n2; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            printf("%f ", B2[i*n2Tilde + j]);
        }
        printf("\n");
    }

    // compute QR factorization of B2
    float* B2Q = (float*)malloc((n1Union - n2) * (n1Union - n2) * sizeof(float));
    float* B2R = (float*)malloc((n1Union - n2) * n2Tilde * sizeof(float));
    qrBatched(cHandle, B2, n1Union - n2, n2Tilde, B2Q, B2R);
    
    
    // make first matrix with Q in the upper left and identity in the lower right of size n1Union x n1Union
    float* firstMatrix = (float*) malloc(n1Union * n1Union * sizeof(float));
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n1Union; j++) {
            firstMatrix[i*n1Union + j] = 0.0;
        }
    }
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n1; j++) {
            firstMatrix[i*n1Union + j] = Q[i*n1 + j];
        }
    }
    for (int i = 0; i < n1Tilde; i++) {
        firstMatrix[(n1 + i)*n1Union + n1 + i] = 1.0;
    }

    // print firstMatrix
    printf("firstMatrix:\n");
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n1Union; j++) {
            printf("%f ", firstMatrix[i*n1Union + j]);
        }
        printf("\n");
    }


    // make second matrix with identity in the upper left corner and B2Q in the lower right corner of size n1Union x n1Union
    float* secondMatrix = (float*) malloc(n1Union * n1Union * sizeof(float));
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n1Union; j++) {
            secondMatrix[i*n1Union + j] = 0.0;
        }
    }
    for (int i = 0; i < n2; i++) {
        secondMatrix[i*n1Union + i] = 1.0;
    }
    for (int i = 0; i < n1Union - n2; i++) {
        for (int j = 0; j < n1Union - n2; j++) {
            secondMatrix[(n2 + i) * n1Union + n2 + j] = B2Q[i * (n1Union - n2) + j];
        }
    }

    // print secondMatrix
    printf("secondMatrix:\n");
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n1Union; j++) {
            printf("%f ", secondMatrix[i*n1Union + j]);
        }
        printf("\n");
    }

    // compute newQ = firstMatrix * secondMatrix
    float* newQ = (float*) malloc(n1Union * n1Union * sizeof(float));
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n1Union; j++) {
            newQ[i*n1Union + j] = 0.0;
            for (int k = 0; k < n1Union; k++) {
                newQ[i*n1Union + j] += firstMatrix[i*n1Union + k] * secondMatrix[k*n1Union + j];
            }
        }
    }

    // print newQ
    printf("newQ:\n");
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n1Union; j++) {
            printf("%f ", newQ[i*n1Union + j]);
        }
        printf("\n");
    }

    // make newR with R in the top left corner, B1 in the top right corner and B2R under B1 of size n1Union x n2Union
    float* newR = (float*) malloc(n1Union * n2Union * sizeof(float));
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n2Union; j++) {
            newR[i*n2Union + j] = 0.0;
        }
    }

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            newR[i*n2Union + j] = R[i*n2 + j];
        }
    }

    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            newR[i*n2Union + n2 + j] = B1[i*n2Tilde + j];
        }
    }

    for (int i = 0; i < n1Tilde; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            newR[(n2 + i)*n2Union + n2 + j] = B2R[i*n2Tilde + j];
        }
    }

    // print B1
    printf("B1:\n");
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            printf("%f ", B1[i*n2Tilde + j]);
        }
        printf("\n");
    }

    // print newR
    printf("newR:\n");
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n2Union; j++) {
            printf("%f ", newR[i*n2Union + j]);
        }
        printf("\n");
    }

    // compute the new solution m_k for the least squares problem
    float* tempM_k = (float*) malloc(n2Union * sizeof(float));
    LSProblem(cHandle, A, newQ, newR, m_kOut, residual, IUnion, JUnion, n1Union, n2Union, k, residualNorm);

    free(m_kOut);
    m_kOut = (float*) malloc(n2Union * sizeof(float));

    // compute m_KOut = Pc * tempM_k
    for (int i = 0; i < n2Union; i++) {
        m_kOut[i] = 0.0;
        for (int j = 0; j < n2Union; j++) {
            m_kOut[i] += Pc[i * n2Union + j] * tempM_k[j];
        }
    }

    // print m_kOut
    printf("m_kOut:\n");
    for (int i = 0; i < n1Union; i++) {
        printf("%f ", m_kOut[i]);
    }
    printf("\n");



    free(AIJTilde);
    free(AITildeJTilde);
    free(ABreve);
    free(ATilde);
    free(Pr);
    free(Pc);
    free(B2);
    free(B2Q);
    free(B2R);

    printf("done with updateQR\n");
}




#endif