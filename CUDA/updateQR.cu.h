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

// Function for updating the QR decomposition
// cHandle = the cublas handle
// A = the input CSC matrix
// AHat = the submatrix of size n1 x n2
// Q = the Q matrix
// R = the R matrix
// I = the row indices of AHat
// J = the column indices of AHat
// ITilde = the row indices to potentioally add to AHat
// JTilde = the column indices to potentially add to AHat
// IUnion = the union of I and ITilde
// JUnion = the union of J and JTilde
// n1 = the lentgh of I
// n2 = the length of J
// n1Tilde = the length of ITilde
// n2Tilde = the length of JTilde
// n1Union = the length of IUnion
// n2Union = the length of JUnion
// m_kOut = the output of the LS problem
// residual = the residual vector
// residualNorm = the norm of the residual vector
// k = the current iteration
int updateQR(cublasHandle_t cHandle, CSC* A, float** AHat, float** Q, float** R, int** unsortedI, int** unsortedJ, int** sortedI, int** sortedJ, int* ITilde, int* JTilde, int* IUnion, int* JUnion, int n1, int n2, int n1Tilde, int n2Tilde, int n1Union, int n2Union, float** m_kOut, float* residual, float* residualNorm, int k) {
    printf("\n------UPDATE QR------\n");

    // create AIJTilde
    // print JTilde
    printf("JTilde:\n");
    for (int i = 0; i < n2Tilde; i++) {
        printf("%d ", JTilde[i]);
    }
    
    float* AIJTilde = CSCToDense(A, (*sortedI), JTilde, n1, n2Tilde);

    // print AIJTilde
    printf("AIJTilde:\n");
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            printf("%f ", AIJTilde[i * n2Tilde + j]);
        }
        printf("\n");
    }

    // print Q
    printf("Q:\n");
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n1; j++) {
            printf("%f ", (*Q)[i * n1 + j]);
        }
        printf("\n");
    }


    // 13.1) create AITildeJTilde of size n1Tilde x n2Tilde
    float* AITildeJTilde = CSCToDense(A, ITilde, JTilde, n1Tilde, n2Tilde);

    // // create ATilde of size n1Union x n2Union
    // float* ATilde = (float*) malloc(n1Union * n2Union * sizeof(float));
    
    // // set upper left square to AHat of size n1 x n2
    // for (int i = 0; i < n1; i++) {
    //     for (int j = 0; j < n2; j++) {
    //         ATilde[i*n2Union + j] = (*AHat)[i * n2 + j];
    //     }
    // }

    // // set upper right square to AIJTilde of size n1 x n2Tilde
    // for (int i = 0; i < n1; i++) {
    //     for (int j = 0; j < n2Tilde; j++) {
    //         ATilde[i * n2Union + n2 + j] = AIJTilde[i * n2Tilde + j];
    //     }
    // }

    // // set lower left square to zeros of size n1Tilde x n2
    // for (int i = 0; i < n1Tilde; i++) {
    //     for (int j = 0; j < n2; j++) {
    //         ATilde[(n1 + i) * n2Union + j] = 0;
    //     }
    // }

    // // set lower right square to AITildeJTilde of size n1Tilde x n2Tilde
    // for (int i = 0; i < n1Tilde; i++) {
    //     for (int j = 0; j < n2Tilde; j++) {
    //         ATilde[(n1 + i) * n2Union + n2 + j] = AITildeJTilde[i * n2Tilde + j];
    //     }
    // }

    // // print ATilde
    // printf("ATilde:\n");
    // for (int i = 0; i < n1Union; i++) {
    //     for (int j = 0; j < n2Union; j++) {
    //         printf("%f ", ATilde[i * n2Union + j]);
    //     }
    //     printf("\n");
    // }

    // Create permutation matrices Pr and Pc
    float* Pr = (float*)malloc(n1Union * n1Union * sizeof(float));
    float* Pc = (float*)malloc(n2Union * n2Union * sizeof(float));
    createPermutationMatrices(IUnion, JUnion, n1Union, n2Union, Pr, Pc);

    // 13.2) ABreve of size n1 x n2Tilde = Q^T * AIJTilde
    float* ABreve = (float*)malloc(n1 * n2Tilde * sizeof(float));
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            ABreve[i * n2Tilde + j] = 0;
            for (int k = 0; k < n1; k++) {
                ABreve[i * n2Tilde + j] += (*Q)[k * n1 + i] * AIJTilde[k * n2Tilde + j];
            }
        }
    }

    printf("ABreve:\n");
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            printf("%f ", ABreve[i * n2Tilde + j]);
        }
        printf("\n");
    }

    // 13.3) Compute B1 = ABreve[0 : n2, 0 : n2]
    float* B1 = (float*) malloc(n2 * n2Tilde * sizeof(float));
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            B1[i * n2Tilde + j] = ABreve[i * n2Tilde + j];
        }
    }

    // 13.4) Compute B2 = ABreve[n2 + 1 : n1, 0 : n2Tilde] + AITildeJTilde
    float* B2;
    if (n1 - n2 < 0) {
        B2 = (float*) malloc(n1Tilde * n2Tilde * sizeof(float));
        for (int i = 0; i < n1Tilde; i++) {
            for (int j = 0; j < n2Tilde; j++) {
                B2[i * n2Tilde + j] = AITildeJTilde[i * n2Tilde + j];
            }
        }
    } else {
        B2 = (float*) malloc((n1Union - n2) * n2Tilde * sizeof(float));
        for (int i = 0; i < n1 - n2; i++) {
            for (int j = 0; j < n2Tilde; j++) {
                B2[i * n2Tilde + j] = ABreve[(n2 + i) * n2Tilde + j];
            }
        }

        for (int i = 0; i < n1Tilde; i++) {
            for (int j = 0; j < n2Tilde; j++) {
                B2[(n1 - n2 + i) * n2Tilde + j] = AITildeJTilde[i * n2Tilde + j];
            }
        }
    }
    
    // print AITildeJTilde
    printf("AITildeJTilde:\n");
    for (int i = 0; i < n1Tilde; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            printf("%f ", AITildeJTilde[i * n2Tilde + j]);
        }
        printf("\n");
    }

    // print B2
    printf("B2:\n");
    for (int i = 0; i < n1Union - n2; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            printf("%f ", B2[i * n2Tilde + j]);
        }
        printf("\n");
    }

    // 13.5) Do QR factorization of B2
    float* B2Q = (float*)malloc((n1Union - n2) * (n1Union - n2) * sizeof(float));
    float* B2R = (float*)malloc((n1Union - n2) * n2Tilde * sizeof(float));
    int qrSuccess = qrBatched(cHandle, &B2, n1Union - n2, n2Tilde, &B2Q, &B2R);

    // print B2Q
    printf("B2Q:\n");
    for (int i = 0; i < n1Union - n2; i++) {
        for (int j = 0; j < n1Union - n2; j++) {
            printf("%f ", B2Q[i * (n1Union - n2) + j]);
        }
        printf("\n");
    }

    // print B2R
    printf("B2R:\n");
    for (int i = 0; i < n1Union - n2; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            printf("%f ", B2R[i * n2Tilde + j]);
        }
        printf("\n");
    }

    
    // 13.6) Compute Q_B and R_B from algorithm 17
    // make first matrix with Q in the upper left and identity in the lower right of size n1Union x n1Union
    float* firstMatrix = (float*) malloc(n1Union * n1Union * sizeof(float));
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n1Union; j++) {
            firstMatrix[i * n1Union + j] = 0.0;
        }
    }
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n1; j++) {
            firstMatrix[i * n1Union + j] = (*Q)[i*n1 + j];
        }
    }
    for (int i = 0; i < n1Tilde; i++) {
        firstMatrix[(n1 + i) * n1Union + n1 + i] = 1.0;
    }

    // print firstMatrix
    printf("firstMatrix:\n");
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n1Union; j++) {
            printf("%f ", firstMatrix[i * n1Union + j]);
        }
        printf("\n");
    }


    // make second matrix with identity in the upper left corner and B2Q in the lower right corner of size n1Union x n1Union
    float* secondMatrix = (float*) malloc(n1Union * n1Union * sizeof(float));
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n1Union; j++) {
            secondMatrix[i * n1Union + j] = 0.0;
        }
    }
    for (int i = 0; i < n2; i++) {
        secondMatrix[i * n1Union + i] = 1.0;
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
            printf("%f ", secondMatrix[i * n1Union + j]);
        }
        printf("\n");
    }

    // compute unsortedQ = firstMatrix * secondMatrix
    float* unsortedQ = (float*) malloc(n1Union * n1Union * sizeof(float));
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n1Union; j++) {
            unsortedQ[i * n1Union + j] = 0.0;
            for (int k = 0; k < n1Union; k++) {
                unsortedQ[i * n1Union + j] += firstMatrix[i * n1Union + k] * secondMatrix[k * n1Union + j];
            }
        }
    }

    // print unsortedQ
    printf("unsortedQ:\n");
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n1Union; j++) {
            printf("%f ", unsortedQ[i * n1Union + j]);
        }
        printf("\n");
    }

    // make unsortedR with R in the top left corner, B1 in the top right corner and B2R under B1 of size n1Union x n2Union
    float* unsortedR = (float*) malloc(n1Union * n2Union * sizeof(float));
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n2Union; j++) {
            unsortedR[i * n2Union + j] = 0.0;
        }
    }

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            unsortedR[i * n2Union + j] = (*R)[i * n2 + j];
        }
    }

    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            unsortedR[i * n2Union + n2 + j] = B1[i * n2Tilde + j];
        }
    }

    for (int i = 0; i < n1Union - n2; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            unsortedR[(n2 + i) * n2Union + n2 + j] = B2R[i * n2Tilde + j];
        }
    }

    // print B1
    printf("B1:\n");
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n2Tilde; j++) {
            printf("%f ", B1[i * n2Tilde + j]);
        }
        printf("\n");
    }

    // print unsortedR
    printf("unsortedR:\n");
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n2Union; j++) {
            printf("%f ", unsortedR[i * n2Union + j]);
        }
        printf("\n");
    }

    free(*m_kOut);
    (*m_kOut) = (float*) malloc(n2Union * sizeof(float));

    // 13.7) compute the new solution m_k for the least squares problem
    int lsSuccess = LSProblem(cHandle, A, unsortedQ, unsortedR, m_kOut, residual, IUnion, JUnion, n1Union, n2Union, k, residualNorm);

    if (lsSuccess != 0) {
        printf("LSProblem failed\n");
        return 1;
    }
    
    float* tempM_k = (float*) malloc(n2Union * sizeof(float));
    memcpy(tempM_k, (*m_kOut), n2Union * sizeof(float));

    free(*m_kOut);
    (*m_kOut) = (float*) malloc(n2Union * sizeof(float));

    // compute m_KOut = Pc * tempM_k
    for (int i = 0; i < n2Union; i++) {
        (*m_kOut)[i] = 0.0;
        for (int j = 0; j < n2Union; j++) {
            (*m_kOut)[i] += Pc[i * n2Union + j] * tempM_k[j];
        }
    }

    // compute sortedJ = Pc * J
    free(*sortedJ);
    (*sortedJ) = (int*) malloc(n2Union * sizeof(int));
    for (int i = 0; i < n2Union; i++) {
        (*sortedJ)[i] = 0;
        for (int j = 0; j < n2Union; j++) {
            (*sortedJ)[i] += Pc[i * n2Union + j] * (*JUnion)[j];
        }
    }

    // print m_kOut
    printf("m_kOut:\n");
    for (int i = 0; i < n2Union; i++) {
        printf("%f ", (*m_kOut)[i]);
    }
    printf("\n");

    // 14) compute residual = A * mHat_k - e_k
    // malloc space for residual
    // do matrix multiplication
    int* IDense = (int*) malloc(A->m * sizeof(int));
    int* JDense = (int*) malloc(A->n * sizeof(int));
    for (int i = 0; i < A->m; i++) {
        IDense[i] = i;
    }
    for (int j = 0; j < A->n; j++) {
        JDense[j] = j;
    }

    // set sortedI and sortedJ to Pr * IUnion and Pc * JUnion
    free(*sortedI);
    printf("free I\n");
    free(*sortedJ);
    printf("free J\n");
    (*sortedI) = (int*) malloc(n1Union * sizeof(int));
    printf("malloc I\n");
    (*sortedJ) = (int*) malloc(n2Union * sizeof(int));
    printf("malloc J\n");
    
    // compute pr * I
    for (int i = 0; i < n1Union; i++) {
        (*sortedI)[i] = 0;
        for (int j = 0; j < n1Union; j++) {
            (*sortedI)[i] += Pr[i * n1Union + j] * IUnion[j];
        }
    }

    printf("I (after pr * I):\n");
    for (int i = 0; i < n1Union; i++) {
        printf("%d ", (*sortedI)[i]);
    }

    // compute Pc * J
    for (int i = 0; i < n2Union; i++) {
        (*sortedJ)[i] = 0;
        for (int j = 0; j < n2Union; j++) {
            (*sortedJ)[i] += Pc[i * n2Union + j] * JUnion[j];
        }
    }

    // set unsortedI and unsortedJ to IUnion and JUnion
    free(*unsortedI);
    free(*unsortedJ);
    (*unsortedI) = (int*) malloc(n1Union * sizeof(int));
    (*unsortedJ) = (int*) malloc(n2Union * sizeof(int));
    
    for (int i = 0; i < n1Union; i++) {
        (*unsortedI)[i] = IUnion[i];
    }

    for (int i = 0; i < n2Union; i++) {
        (*unsortedJ)[i] = JUnion[i];
    }

    float* ADense = CSCToDense(A, IDense, JDense, A->m, A->n);
    // compute residual
    for (int i = 0; i < A->m; i++) {
        residual[i] = 0.0;
        for (int j = 0; j < A->n; j++) {
            for (int h = 0; h < n2Union; h++) {
                if ((*sortedJ)[h] == j) {
                    residual[i] += ADense[i * A->n + j] * (*m_kOut)[h];
                }
            }
        }
        if (i == k) {
            residual[i] -= 1.0;
        }
    }
    
    printf("residual:\n");
    for (int i = 0; i < A->m; i++) {
        printf("%f ", residual[i]);
    }
    printf("\n");

    // compute the norm of the residual
    *residualNorm = 0.0;
    for (int i = 0; i < A->m; i++) {
        *residualNorm += residual[i] * residual[i];
    }
    *residualNorm = sqrt(*residualNorm);
    
    printf("residual norm: %f\n", *residualNorm);


    // // sort unsortedQ and unsortedR into Q and R
    // free(*Q);
    // (*Q) = (float*) malloc(n1Union * n1Union * sizeof(float));
    // printf("malloc q\n");
    // free(*R);
    // (*R) = (float*) malloc(n1Union * n2Union * sizeof(float));
    // printf("malloc r\n");

    // // compute Pr * unsortedQ
    // for (int i = 0; i < n1Union; i++) {
    //     for (int j = 0; j < n1Union; j++) {
    //         (*Q)[i * n1Union + j] = 0.0;
    //         for (int k = 0; k < n1Union; k++) {
    //             (*Q)[i * n1Union + j] += Pr[i * n1Union + k] * unsortedQ[k * n1Union + j];
    //         }
    //     }
    // }
    // printf("pr * unsortedQ\n");

    // // compute unsortedR * Pc
    // for (int i = 0; i < n1Union; i++) {
    //     for (int j = 0; j < n2Union; j++) {
    //         (*R)[i * n2Union + j] = 0.0;
    //         for (int k = 0; k < n2Union; k++) {
    //             (*R)[i * n2Union + j] += unsortedR[i * n2Union + k] * Pc[k * n2Union + j];
    //         }
    //     }
    // }
    // printf("unsortedR * Pc\n");

    // // set AHat to Q * R
    // free(*AHat);
    // (*AHat) = (float*) malloc(n1Union * n2Union * sizeof(float));
    // for (int i = 0; i < n1Union; i++) {
    //     for (int j = 0; j < n2Union; j++) {
    //         (*AHat)[i * n2Union + j] = 0.0;
    //         for (int k = 0; k < n1Union; k++) {
    //             (*AHat)[i * n2Union + j] += (*Q)[i * n1Union + k] * (*R)[k * n2Union + j];
    //         }
    //     }
    // }
    printf("Q * R\n");

    // set R to unsortedR and Q to unsortedQ
    free(*Q);
    printf("free Q\n");
    free(*R);
    printf("free R\n");
    (*Q) = (float*) malloc(n1Union * n1Union * sizeof(float));
    printf("malloc Q\n");
    (*R) = (float*) malloc(n1Union * n2Union * sizeof(float));
    printf("malloc R\n");
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n1Union; j++) {
            (*Q)[i * n1Union + j] = unsortedQ[i * n1Union + j];
        }
    }
    for (int i = 0; i < n1Union; i++) {
        for (int j = 0; j < n2Union; j++) {
            (*R)[i * n2Union + j] = unsortedR[i * n2Union + j];
        }
    }


    // free memory
    free(AIJTilde);
    printf("freed AIJTilde\n");
    free(AITildeJTilde);
    printf("freed AITildeJTilde\n");
    free(ABreve);
    printf("freed ABreve\n");
    // free(ATilde);
    // printf("freed ATilde\n");
    free(Pr);
    printf("freed Pr\n");
    free(Pc);
    printf("freed Pc\n");
    free(B1);
    printf("freed B1\n");
    free(B2);
    printf("freed B2\n");
    free(B2Q);
    printf("freed B2Q\n");
    free(B2R);
    printf("freed B2R\n");
    free(firstMatrix);
    printf("freed firstMatrix\n");
    // free(secondMatrix);
    // printf("freed secondMatrix\n");
    free(unsortedQ);
    printf("freed unsortedQ\n");
    free(unsortedR);
    printf("freed unsortedR\n");

    free(tempM_k);
    printf("freed tempM_k\n");

    printf("done with updateQR\n");

    return 0;
}

#endif