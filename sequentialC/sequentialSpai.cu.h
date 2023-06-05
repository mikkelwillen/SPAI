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
#include "updateQR.cu.h"
#include "LSProblem.cu.h"
#include "singular.cu.h"


/* A = matrix we want to compute SPAI on
m = number of rows
n = number of columns
tolerance = tolerance
maxIteration = constraint for the maximal number of iterations
s = number of rho_j - the most profitable indices */
CSC* sequentialSpai(CSC* A, float tolerance, int maxIteration, int s) {
    printf("------SEQUENTIAL SPAI------\n");
    printf("running with parameters: tolerance = %f, maxIteration = %d, s = %d\n", tolerance, maxIteration, s);

    // // check if matrix is singular
    // int checkSingular = checkSingularity(A);
    // if (checkSingular == 1) {
    //     printf("Matrix is singular\n");

    //     return NULL;
    // }

    // Initialize cuBLAS
    cublasHandle_t cHandle;
    cublasStatus_t stat;
    stat = cublasCreate(&cHandle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublas initialization failed\n");
        printf("cublas error: %d\n", stat);

        return NULL;
    } 

    // Initialize M and set to diagonal
    CSC* M = createDiagonalCSC(A->m, A->n);

    // m_k = column in M
    for (int k = 0; k < M->n; k++) {    
        // variables
        int n1 = 0;
        int n2 = 0;
        int iteration = 0;
        float residualNorm = 0.0;

        int* J;
        int* I;
        int* sortedJ = (int*) malloc(sizeof(int) * M->n);
        float* AHat;
        float* Q;
        float* R;
        float* mHat_k;
        float* residual;

        // 1) Find the initial sparsity J of m_k
        // Malloc space for the indeces from offset[k] to offset[k + 1]
        n2 = M->offset[k + 1] - M->offset[k];
        J = (int*) malloc(sizeof(int) * n2);

        // Iterate through row indeces from offset[k] to offset[k + 1] and take all elements from the flatRowIndex
        int h = 0;
        for (int i = M->offset[k]; i < M->offset[k + 1]; i++) {
            J[h] = M->flatRowIndex[i];
            h++;
        }

        // 2) Compute the row indices I of the corresponding nonzero entries of A(i, J)
        // We initialize I to -1, and the iterate through all elements of J. Then we iterate through the row indeces of A from the offset J[j] to J[j] + 1. If the row index is already in I, we dont do anything, else we add it to I.
        I = (int*) malloc(sizeof(int) * A->m);
        for (int i = 0; i < A->m; i++) {
            I[i] = -1;
        }

        n1 = 0;
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

        if (n1 == 0) {
            n2 = 0;
        }

        // 3) Create Â = A(I, J)
        // We initialize AHat to zeros. Then we iterate through all indeces of J, and iterate through all indeces of I. 
        // For each of the indices of I and the indices in the flatRowIndex, we check if they match. If they do, we add that element to AHat.
        AHat = CSCToDense(A, I, J, n1, n2);

        // 4) Do QR decomposition of AHat
        Q = (float*) malloc(sizeof(float) * n1 * n1);
        R = (float*) malloc(sizeof(float) * n1 * n2);

        int qrSuccess = qrBatched(cHandle, &AHat, n1, n2, &Q, &R);

        // Overwrite AHat
        free(AHat);
        AHat = CSCToDense(A, I, J, n1, n2);

        // 5) Compute the solution m_k for the least squares problem
        mHat_k = (float*) malloc(n2 * sizeof(float));
        residual = (float*) malloc(A->m * sizeof(float));        

        int invSuccess = LSProblem(cHandle, A, Q, R, &mHat_k, residual, I, J, n1, n2, k, &residualNorm);

        if (invSuccess != 0) {
            printf("LSProblem failed\n");
            return NULL;
        }

        // 6) Compute residual = A * mHat_k - e_k
        // Malloc space for residual
        // Do matrix multiplication
        int* IDense = (int*) malloc(A->m * sizeof(int));
        int* JDense = (int*) malloc(A->n * sizeof(int));
        for (int i = 0; i < A->m; i++) {
            IDense[i] = i;
        }
        for (int j = 0; j < A->n; j++) {
            JDense[j] = j;
        }

        float* ADense = CSCToDense(A, IDense, JDense, A->m, A->n);
        
        // Compute residual
        for (int i = 0; i < A->m; i++) {
            residual[i] = 0.0;
            for (int j = 0; j < A->n; j++) {
                for (int h = 0; h < n2; h++) {
                    if (J[h] == j) {
                        residual[i] += ADense[i * A->n + j] * mHat_k[h];
                    }
                }
            }
            if (i == k) {
                residual[i] -= 1.0;
            }
        }

        // Compute the norm of the residual
        residualNorm = 0.0;
        for (int i = 0; i < A->m; i++) {
            residualNorm += residual[i] * residual[i];
        }
        residualNorm = sqrt(residualNorm);
    
        int somethingToBeDone = 1;

        // While norm of residual > tolerance do
        while (residualNorm > tolerance && maxIteration > iteration && somethingToBeDone) {
            iteration++;

            // Variables
            int n1Tilde = 0;
            int n2Tilde = 0;
            int n1Union = 0;
            int n2Union = 0;
            int l = 0;
            int kNotInI = 0;
            
            int* L;
            int* keepArray;
            int* JTilde;
            int* ITilde;
            int* IUnion;
            int* JUnion;
            float* rhoSq;
            int* smallestIndices;
            int* smallestJTilde;

            // 7) Set L to the set of indices where r(l) != 0
            // Count the numbers of nonzeros in residual
            for (int i = 0; i < A->m; i++) {
                if (residual[i] != 0.0) {
                    l++;
                } else if (k == i) {
                    kNotInI = 1;
                }
            }

            // Check if k is in I
            for (int i = 0; i < n1; i++) {
                if (k == I[i]) {
                    kNotInI = 0;
                }
            }

            // increment l if k is not in I
            if (kNotInI) {
                l++;
            }
            
            // Malloc space for L and fill it with the indices
            L = (int*) malloc(sizeof(int) * l);

            int index = 0;
            for (int i = 0; i < A->m; i++) {
                if (residual[i] != 0.0 || (kNotInI && i == k)) {
                    L[index] = i;
                    index++;
                }
            }


            // 8) Set JTilde to the set of columns of A corresponding to the indices in L that are not already in J
            // Check what indeces we should keep
            keepArray = (int*) malloc(A->n * sizeof(int));
            // set all to 0
            for (int i = 0; i < A->n; i++) {
                keepArray[i] = 0;
            }

            for (int i = 0; i < A->n; i++) {
                for (int j = 0; j < l; j++) {
                    for (int h = A->offset[i]; h < A->offset[i + 1]; h++) {
                        if (L[j] == A->flatRowIndex[h]) {
                            keepArray[i] = 1;
                        }
                    }
                }
            }

            // Remove the indeces that are already in J
            for (int i = 0; i < n2; i++) {
                keepArray[J[i]] = 0;
            }

            // Compute the length of JTilde
            n2Tilde = 0;
            for (int i = 0; i < A->n; i++) {
                if (keepArray[i] == 1) {
                    n2Tilde++;
                }
            }

            // Malloc space for JTilde
            JTilde = (int*) malloc(sizeof(int) * n2Tilde);

            // Fill JTilde
            index = 0;
            for (int i = 0; i < A->n; i++) {
                if (keepArray[i] == 1) {
                    JTilde[index] = i;
                    index++;
                }
            }

            // 9) For each j in JTilde, solve the minimization problem
            // Malloc space for rhoSq
            rhoSq = (float*) malloc(sizeof(float) * n2Tilde);
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

            // 10) Find the s indeces of the column with the smallest rhoSq
            int newN2Tilde = MIN(s, n2Tilde);
            smallestIndices = (int*) malloc(sizeof(int) * newN2Tilde);

            for (int i = 0; i < newN2Tilde; i++) {
                smallestIndices[i] = -1;
            }
            
            // We iterate through rhoSq and find the smallest indeces.
            // First, we set the first s indeces to the first s indeces of JTilde
            // then if we find a smaller rhoSq, we shift the indeces to the right
            // we insert the index of JTilde with the rhoSq smaller than the current smallest elements
            // smallestIndices then contain the indeces of JTIlde corresponding to the smallest values of rhoSq
            for (int i = 0; i < n2Tilde; i++) {
                for (int j = 0; j < newN2Tilde; j++) {
                    if (smallestIndices[j] == -1) {
                        smallestIndices[j] = i;
                        break;
                    } else if (rhoSq[i] < rhoSq[smallestIndices[j]]) {
                        for (int h = newN2Tilde - 1; h > j; h--) {
                            smallestIndices[h] = smallestIndices[h - 1];
                        }

                        smallestIndices[j] = i;
                        break;
                    }
                }
            }

            smallestJTilde = (int*) malloc(sizeof(int) * newN2Tilde);
            for (int i = 0; i < newN2Tilde; i++) {
                smallestJTilde[i] = JTilde[smallestIndices[i]];
            }

            free(JTilde);
            JTilde = (int*) malloc(sizeof(int) * newN2Tilde);
            for (int i = 0; i < newN2Tilde; i++) {
                JTilde[i] = smallestJTilde[i];
            }

            // 11) Determine the new indices Î
            // Denote by ITilde the new rows, which corresponds to the nonzero rows of A(:, J union JTilde) not contained in I yet
            n2Tilde = newN2Tilde;
            n2Union = n2 + n2Tilde;
            JUnion = (int*) malloc(sizeof(int) * n2Union);
            for (int i = 0; i < n2; i++) {
                JUnion[i] = J[i];
            }
            for (int i = 0; i < n2Tilde; i++) {
                JUnion[n2 + i] = JTilde[i];
            }

            ITilde = (int*) malloc(sizeof(int) * A->m);
            for (int i = 0; i < A->m; i++) {
                ITilde[i] = -1;
            }

            n1Tilde = 0;
            for (int j = 0; j < n2Union; j++) {
                for (int i = A->offset[JUnion[j]]; i < A->offset[JUnion[j] + 1]; i++) {
                    int keep = 1;
                    for (int h = 0; h < n1; h++) {
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

            // 12) Make I U ITilde and J U JTilde
            // Make union of I and ITilde
            n1Union = n1 + n1Tilde;
            IUnion = (int*) malloc(sizeof(int) * (n1 + n1Tilde));
            for (int i = 0; i < n1; i++) {
                IUnion[i] = I[i];
            }
            for (int i = 0; i < n1Tilde; i++) {
                IUnion[n1 + i] = ITilde[i];
            }

            // 13) Update the QR factorization of A(IUnion, JUnion)
            int updateSuccess = updateQR(cHandle, A, &AHat, &Q, &R, &I, &J, &sortedJ, ITilde, JTilde, IUnion, JUnion, n1, n2, n1Tilde, n2Tilde, n1Union, n2Union, &mHat_k, residual, &residualNorm, k);

            if (updateSuccess != 0) {
                printf("update failed\n");
                
                return NULL;
            }

            n1 = n1Union;
            n2 = n2Union;
            
            // free memory
            free(L);
            free(IUnion);
            free(JUnion);
            free(ITilde);
            free(JTilde);
            free(smallestJTilde);
            free(rhoSq);
            free(keepArray);
        }

        // 16) Set m_k(J) = mHat_k
        // Update kth column of M
        M = updateKthColumnCSC(M, mHat_k, k, sortedJ, n2);

        // Free memory
        free(I);
        free(J);
        free(sortedJ);
        free(AHat);
        free(Q);
        free(R);
        free(mHat_k);
        free(residual);
    }

    cublasDestroy(cHandle);

    return M;
}

#endif