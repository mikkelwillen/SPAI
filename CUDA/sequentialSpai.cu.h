#ifndef SPAI_H
#define SPAI_H

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "csc.cu.h"
#include "constants.cu.h"
#include "qrBatched.cu.h"
#include "invBatched.cu.h"


// A = matrix we want to compute SPAI on
// m, n = size of array
// tolerance = tolerance
// maxIteration = constraint for the maximal number of iterations
// s = number of rho_j - the most profitable indices
CSC* sequentialSpai(CSC* A, float tolerance, int maxIteration, int s) {
    printCSC(A);

    // initialize cuBLAS
    cublasHandle_t cHandle;
    cublasStatus_t stat;
    stat = cublasCreate(&cHandle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cusolver initialization failed\n");
        printf("cusolver error: %d\n", stat);
    } 

    // initialize M and set to diagonal
    CSC* M = createDiagonalCSC(A->m, A->n);

    // m_k = column in M
    for (int k = 0; k < M->n; k++) {

        printf("\n\n------NEW COLUMN: %d------", k);

        // a) Find the initial sparsity J of m_k
        // malloc space for the indeces from offset[k] to offset[k + 1]
        int n2 = M->offset[k + 1] - M->offset[k];
        int* J = (int*) malloc(sizeof(int) * n2);

        // iterate through row indeces from offset[k] to offset[k + 1] and take all elements from the flatRowIndex
        int h = 0;
        for (int i = M->offset[k]; i < M->offset[k + 1]; i++) {
            J[h] = M->flatRowIndex[i];
            h++;
        }

        // // printJ
        // printf("\nJ: ");
        // for (int i = 0; i < n2; i++) {
        //     printf("%d ", J[i]);
        // }

        // b) Compute the row indices I of the corresponding nonzero entries of A(i, J)
        // We initialize I to -1, and the iterate through all elements of J. Then we iterate through the row indeces of A from the offset J[j] to J[j] + 1. If the row index is already in I, we dont do anything, else we add it to I.
        int* I = (int*) malloc(sizeof(int) * A->m);
        for (int i = 0; i < A->m; i++) {
            I[i] = -1;
        }

        int n1 = 0;
        for (int j = 0; j < n2; j++) {
            for (int i = A->offset[J[j]]; i < A->offset[J[j] + 1]; i++) {
                int keep = 1;
                for (int h = 0; h < A->m; h++) {
                    if (A->flatRowIndex[i] == I[h]) {
                        keep = 0;
                    }
                }
                if (keep == 1) {
                    I[n1] = A->flatRowIndex[i];
                    n1++;
                }
            }
        }

        // print I
        printf("\nI: ");
        for (int i = 0; i < n1; i++) {
            printf("%d ", I[i]);
        }

        // c) Create Â = A(I, J)
        // We initialize AHat to zeros. Then we iterate through all indeces of J, and iterate through all indeces of I. For each of the indices of I and the indices in the flatRowIndex, we check if they match. If they do, we add that element to AHat.
        float* AHat = (float*) calloc(n1 * n2, sizeof(float));
        for(int j = 0; j < n2; j++) {
            for (int i = 0; i < n1; i++) {
                for (int l = A->offset[J[j]]; l < A->offset[J[j] + 1]; l++) {
                    if (I[i] == A->flatRowIndex[l]) {
                        AHat[j * n2 + i] = A->flatData[l];
                    }
                }
            }
        }

        // print AHat
        printf("\nAhat: ");
        for (int i = 0; i < n1 * n2; i++) {
            printf("%f ", AHat[i]);
        }

        // d) do QR decomposition of AHat
        float* Q = (float*) malloc(sizeof(float) * n1 * n1);
        float* R = (float*) malloc(sizeof(float) * n1 * n2);

        qrBatched(AHat, n1, n2, Q, R);

        // e) compute ĉ = Q^T ê_k
        // make e_k and set index k to 1.0
        printf("\nk: %d", k);
        float* e_k = (float*) calloc(n1, sizeof(float));
        for (int i = 0; i < n1; i++) {
            if (k == I[i]) {
                e_k[i] = 1.0;
            }
        }
        printf("\ne_k: ");
        for (int i = 0; i < n1; i++) {
            printf("%f ", e_k[i]);
        }

        // malloc size for cHat and do matrix multiplication
        float* cHat = (float*) calloc(n2, sizeof(float));
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                cHat[j] += Q[j * n1 + i] * e_k[i];
            }
        }
        printf("\ncHat: ");
        for (int i = 0; i < n2; i++) {
            printf("%f ", cHat[i]);
        }

        // f) compute mHat_k = R^-1 cHat
        // Malloc space for mHat_k
        float* mHat_k = (float*) malloc(sizeof(float) * n2);
        
        // Make the inverse
        // Placeholder InvR. R is n2 x n2
        // Malloc space for the inverse of R
        float* invR = (float*) malloc(sizeof(float) * n2 * n2);

        invBatched(R, n2, invR);

        // print invR
        printf("\ninvR: ");
        for (int i = 0; i < n2 * n2; i++) {
            printf("%f ", invR[i]);
        }

        // Matrix multiplication
        for (int i = 0; i < n2; i++) {
            mHat_k[i] = 0.0;
            for (int j = 0; j < n2; j++) {
                mHat_k[i] += invR[i * n2 + j] * cHat[i];
            }
        }
        printf("\nmHat_k: ");
        for (int i = 0; i < n2; i++) {
            printf("%f ", mHat_k[i]);
        }

        // g) set m_k(J) = ^m_k
        // ER DER en grund til at vi opdaterer m_k inden vi er helt færdige med mHat_k. Vi ved jo hvilke index mHat_k svarer til i m_k (pga J).
        M = updateKthColumnCSC(M, mHat_k, k, J, n2);

        // print M
        printCSC(M);
        // h) compute residual
        // residual = A * mHat_k - e_k
        // Malloc space for residual
        float* residual = (float*) calloc(A->m, sizeof(float));

        // Matrix multiplication
        for (int i = 0; i < A->m; i++) {
            for (int j = 0; j < n2; j++) {
                for (int h = A->offset[k]; h < A->offset[k + 1]; h++) {
                    if (i == A->flatRowIndex[h]) {
                       residual[i] += A->flatData[h] * mHat_k[j];
                    }
                }
                if (i == k) {
                    residual[i] -= 1.0;
                }
            }
        }
        printf("\nresidual: ");
        for (int i = 0; i < A->m; i++) {
            printf("%f ", residual[i]);
        }
        
        // compute the norm of the residual
        float residualNorm = 0.0;
        for (int i = 0; i < A->m; i++) {
            residualNorm += residual[i] * residual[i];
        }
        residualNorm = sqrt(residualNorm);
        printf("\nnorm: %f", residualNorm);
        
        int iteration = 0;
        // while norm of residual > tolerance do
        while (residualNorm > tolerance && maxIteration + 1 > iteration) {
            printf("\n\n------Iteration: %d------", iteration);
            iteration++;

            // a) Set L to the set of indices where r(l) != 0
            // count the numbers of nonzeros in residual
            int l = 0;
            for (int i = 0; i < A->m; i++) {
                if (residual[i] != 0.0) {
                    l++;
                }
            }

            // malloc space for L and fill it with the indices
            int* L = (int*) malloc(sizeof(int) * l);
            int index = 0;

            // check if k is in I
            int kNotInI = 1;
            for (int i = 0; i < n1; i++) {
                if (k == I[i]) {
                    kNotInI = 0;
                }
            }

            for (int i = 0; i < A->m; i++) {
                if (residual[i] != 0.0 || (kNotInI && i == k)) {
                    L[index] = i;
                    index++;
                }
            }

            // print L
            printf("\nL: ");
            for (int i = 0; i < l; i++) {
                printf("%d ", L[i]);
            }

            // b) Set JTilde to the set of columns of A corresponding to the indices in L that are not already in J
            // check what indeces we should keep
            int* keepArray = (int*) calloc(A->n, sizeof(int));
            for (int i = 0; i < l; i++) {
                for (int j = 0; j < A->n; j++) {
                    for (int h = A->offset[L[i]]; h < A->offset[L[i] + 1]; h++) {
                        keepArray[h] = 1;
                    }
                }
            }

            // remove the indeces that are already in J
            for (int i = 0; i < n2; i++) {
                keepArray[J[i]] = 0;
            }

            // compute the length of JTilde
            int n2Tilde = 0;
            for (int i = 0; i < A->n; i++) {
                if (keepArray[i] == 1) {
                    n2Tilde++;
                }
            }

            // malloc space for JTilde
            int* JTilde = (int*) malloc(sizeof(int) * n2Tilde);

            // fill JTilde
            index = 0;
            for (int i = 0; i < A->n; i++) {
                if (keepArray[i] == 1) {
                    JTilde[index] = i;
                    index++;
                }
            }

            printf("\nJ: ");
            for(int i = 0; i < n2; i++) {
                printf("%d ", J[i]);
            }
            printf("\nJTilde: ");
            for (int i = 0; i < n2Tilde; i++) {
                printf("%d ", JTilde[i]);
            }

            // c) for each j in JTilde, solve the minimization problem
            // Malloc space for rhoSq
            float* rhoSq = (float*) malloc(sizeof(float) * n2Tilde);
            for (int i = 0; i < n2Tilde; i++) {
                float rTAe_j = 0.0; // r^T * A(.,j)
                for (int j = A->offset[JTilde[i]]; j < A->offset[JTilde[i] + 1]; j++) {
                    rTAe_j += A->flatData[j] * residual[A->flatRowIndex[j]];
                }

                float Ae_jNorm = 0.0;
                for (int j = A->offset[JTilde[i]]; j < A->offset[JTilde[i] + 1]; j++) {
                    Ae_jNorm += A->flatData[j] * A->flatData[j];
                }
                Ae_jNorm = sqrt(Ae_jNorm);

                rhoSq[i] = residualNorm * residualNorm - (rTAe_j * rTAe_j) / (Ae_jNorm * Ae_jNorm);
            }

            // d) find the s indeces of the column with the smallest rhoSq
            int newN2Tilde = MIN(s, n2Tilde);
            int* smallestIndices = (int*) malloc(sizeof(int) * newN2Tilde);

            for (int i = 0; i < newN2Tilde; i++) {
                smallestIndices[i] = -1;
            }

            for (int i = 0; i < n2Tilde; i++) {
                for (int j = 0; j < newN2Tilde; j++) {
                    if (smallestIndices[j] == -1) {
                        smallestIndices[j] = i;
                    } else if (rhoSq[i] < rhoSq[smallestIndices[j]]) {
                        for (int h = newN2Tilde - 1; h > j; h--) {
                            smallestIndices[h] = smallestIndices[h - 1];
                        }
                    }
                }
            }

            printf("\nsmallestIndices: ");
            for (int i = 0; i < newN2Tilde; i++) {
                printf("%d ", smallestIndices[i]);
            }

            int* smallestJTilde = (int*) malloc(sizeof(int) * newN2Tilde);
            for (int i = 0; i < newN2Tilde; i++) {
                smallestJTilde[i] = JTilde[smallestIndices[i]];
            }
            
            printf("\nsmallestJTilde: ");
            for (int i = 0; i < newN2Tilde; i++) {
                printf("%d ", smallestJTilde[i]);
            }

            // e) determine the new indices Î
            // Denote by ITilde teh new rows, which corresponds to the nonzero rows of A(:, J union JTilde) not contained in I yet
            int unionN2 = n2 + newN2Tilde;
            int* unionJ = (int*) malloc(sizeof(int) * unionN2);
            for (int i = 0; i < n2; i++) {
                unionJ[i] = J[i];
            }
            for (int i = 0; i < newN2Tilde; i++) {
                unionJ[n2 + i] = smallestJTilde[i];
            }


            int* ITilde = (int*) malloc(sizeof(int) * A->m);
            for (int i = 0; i < A->m; i++) {
                ITilde[i] = -1;
            }

            int n1Tilde = 0;
            for (int j = 0; j < unionN2; j++) {
                for (int i = A->offset[unionJ[j]]; i < A->offset[unionJ[j] + 1]; i++) {
                    int keep = 1;
                    for (int h = 0; h < A->m; h++) {
                        if (A->flatRowIndex[i] == I[h] || A->flatRowIndex[i] == ITilde[h]) {
                            keep = 0;
                        }
                    }
                    if (keep == 1) {
                        ITilde[n1Tilde] = A->flatRowIndex[i];
                        n1Tilde++;
                    }
                }
            }

            // printf ITilde
            printf("\nITilde: ");
            for (int i = 0; i < n1Tilde; i++) {
                printf("%d ", ITilde[i]);
            }

        }

        // Husk kun at bruge de s mindste residuals. Kig på hvordan man laver L igen
        // vi skal have lavet en ny testmatrice, som har flere ikke nuller, så der er mulighed for at finde flere s indeces
        // vi skal teste om step d) giver det rigtige.


    }

    return M;
}

#endif