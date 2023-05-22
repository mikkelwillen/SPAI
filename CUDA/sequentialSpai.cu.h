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
        printf("cublas initialization failed\n");
        printf("cublas error: %d\n", stat);

        return NULL;
    } 

    // initialize M and set to diagonal
    CSC* M = createDiagonalCSC(A->m, A->n);

    // m_k = column in M
    for (int k = 0; k < M->n; k++) {
        printf("\n\n------NEW COLUMN: %d------", k);
        
        // variables
        int n1 = 0;
        int n2 = 0;
        int iteration = 0;
        float residualNorm = 0.0;

        int* J;
        int* I;
        float* AHat;
        float* Q;
        float* R;
        float* mHat_k;
        float* residual;

        // a) Find the initial sparsity J of m_k
        // malloc space for the indeces from offset[k] to offset[k + 1]
        n2 = M->offset[k + 1] - M->offset[k];
        J = (int*) malloc(sizeof(int) * n2);

        // iterate through row indeces from offset[k] to offset[k + 1] and take all elements from the flatRowIndex
        int h = 0;
        for (int i = M->offset[k]; i < M->offset[k + 1]; i++) {
            J[h] = M->flatRowIndex[i];
            h++;
        }

        // print J
        printf("\nJ: ");
        for (int i = 0; i < n2; i++) {
            printf("%d ", J[i]);
        }

        // // printJ
        // printf("\nJ: ");
        // for (int i = 0; i < n2; i++) {
        //     printf("%d ", J[i]);
        // }

        // b) Compute the row indices I of the corresponding nonzero entries of A(i, J)
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

        // print I
        printf("\nI: ");
        for (int i = 0; i < n1; i++) {
            printf("%d ", I[i]);
        }

        // c) Create Â = A(I, J)
        // We initialize AHat to zeros. Then we iterate through all indeces of J, and iterate through all indeces of I. 
        // For each of the indices of I and the indices in the flatRowIndex, we check if they match. If they do, we add that element to AHat.
        AHat = CSCToDense(A, I, J, n1, n2);

        // print AHat
        printf("\nAhat:\n");
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                printf("%f ", AHat[i * n2 + j]);
            }
            printf("\n");
        }
        printf("\n");

        // d) do QR decomposition of AHat
        Q = (float*) malloc(sizeof(float) * n1 * n1);
        R = (float*) malloc(sizeof(float) * n1 * n2);

        int qrSuccess = qrBatched(cHandle, &AHat, n1, n2, &Q, &R);
        printf("after qrBatched\n");

        // overwrite AHat
        free(AHat);
        AHat = CSCToDense(A, I, J, n1, n2);

        // e) Compute the solution m_k for the least squares problem
        mHat_k = (float*) malloc(n2 * sizeof(float));
        residual = (float*) malloc(A->m * sizeof(float));        

        int invSuccess = LSProblem(cHandle, A, Q, R, &mHat_k, residual, I, J, n1, n2, k, &residualNorm);

        if (invSuccess != 0) {
            printf("LSProblem failed\n");
            return NULL;
        }

        printf("\nnorm: %f", residualNorm);
        printf("\n");

        int somethingToBeDone = 1;

        // while norm of residual > tolerance do
        while (residualNorm > tolerance && maxIteration > iteration && somethingToBeDone) {
            printf("\n\n------Iteration: %d------\n", iteration);
            iteration++;

            // variables
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

            // a) Set L to the set of indices where r(l) != 0
            // count the numbers of nonzeros in residual
            for (int i = 0; i < A->m; i++) {
                if (residual[i] != 0.0) {
                    l++;
                } else if (k == i) {
                    kNotInI = 1;
                }
            }

            // check if k is in I
            for (int i = 0; i < n1; i++) {
                if (k == I[i]) {
                    kNotInI = 0;
                }
            }

            // increment l if k is not in I
            if (kNotInI) {
                l++;
            }
            
            // malloc space for L and fill it with the indices
            L = (int*) malloc(sizeof(int) * l);

            int index = 0;
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
            printf("\n");

            // b) Set JTilde to the set of columns of A corresponding to the indices in L that are not already in J
            // check what indeces we should keep
            keepArray = (int*) malloc(A->n * sizeof(int));
            printf("after malloc\n");
            // set all to 0
            for (int i = 0; i < A->n; i++) {
                keepArray[i] = 0;
            }

            // index fra 0 til A->n i keep array er en boolean for, om vi skal tilføje den til JTilde
            // kig på repræsentationen af A
            printf("after set to 0\n");
            for (int i = 0; i < A->n; i++) {
                for (int j = 0; j < l; j++) {
                    for (int h = A->offset[i]; h < A->offset[i + 1]; h++) {
                        if (L[j] == A->flatRowIndex[h]) {
                            keepArray[i] = 1;
                        }
                    }
                }
            }
            printf("after keepArray\n");


            // remove the indeces that are already in J
            for (int i = 0; i < n2; i++) {
                keepArray[J[i]] = 0;
            }
            printf("after remove\n");

            // print keepArray
            printf("\nkeepArray: ");
            for (int i = 0; i < A->n; i++) {
                printf("%d ", keepArray[i]);
            }
            printf("\n");

            // compute the length of JTilde
            n2Tilde = 0;
            for (int i = 0; i < A->n; i++) {
                if (keepArray[i] == 1) {
                    n2Tilde++;
                }
            }

            // malloc space for JTilde
            JTilde = (int*) malloc(sizeof(int) * n2Tilde);

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
            printf("\n");

            // c) for each j in JTilde, solve the minimization problem
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

            //print rhoSq
            printf("\nrhoSq: ");
            for (int i = 0; i < n2Tilde; i++) {
                printf("%f ", rhoSq[i]);
            }
            printf("\n");

            // d) find the s indeces of the column with the smallest rhoSq
            int newN2Tilde = MIN(s, n2Tilde);
            smallestIndices = (int*) malloc(sizeof(int) * newN2Tilde);
            printf("\nnewN2Tilde: %d", newN2Tilde);

            for (int i = 0; i < newN2Tilde; i++) {
                smallestIndices[i] = -1;
            }
            
            // we iterate through rhoSq and find the smallest indeces
            // first, we set the first s indeces to the first s indeces of JTilde
            // then if we find a smaller rhoSq, we shift the indeces to the right
            // we insert the index of JTilde with the rhoSq smaller than the current smallest elements
            // smallestIndices then contain the indeces of JTIlde corresponding to the smallest values of rhoSq
            for (int i = 0; i < n2Tilde; i++) {
                for (int j = 0; j < newN2Tilde; j++) {
                    if (smallestIndices[j] == -1) {
                        smallestIndices[j] = i;
                    } else if (rhoSq[i] < rhoSq[smallestIndices[j]]) {
                        for (int h = newN2Tilde - 1; h > j; h--) {
                            smallestIndices[h] = smallestIndices[h - 1];
                        }

                        smallestIndices[j] = i;
                    }
                }
            }

            printf("\nsmallestIndices: ");
            for (int i = 0; i < newN2Tilde; i++) {
                printf("%d ", smallestIndices[i]);
            }
            printf("\n");

            smallestJTilde = (int*) malloc(sizeof(int) * newN2Tilde);
            for (int i = 0; i < newN2Tilde; i++) {
                smallestJTilde[i] = JTilde[smallestIndices[i]];
            }
            
            printf("\nsmallestJTilde: ");
            for (int i = 0; i < newN2Tilde; i++) {
                printf("%d ", smallestJTilde[i]);
            }
            printf("\n");

            // e) determine the new indices Î
            // Denote by ITilde the new rows, which corresponds to the nonzero rows of A(:, J union JTilde) not contained in I yet
            n2Tilde = newN2Tilde;
            n2Union = n2 + n2Tilde;
            printf("n2Union: %d\n", n2Union);
            JUnion = (int*) malloc(sizeof(int) * n2Union);
            printf("after JUnion\n");
            for (int i = 0; i < n2; i++) {
                JUnion[i] = J[i];
            }
            for (int i = 0; i < n2Tilde; i++) {
                JUnion[n2 + i] = smallestJTilde[i];
            }

            // print JUnion
            printf("\nJUnion: ");
            for (int i = 0; i < n2Union; i++) {
                printf("%d ", JUnion[i]);
            }
            printf("\n");

            ITilde = (int*) malloc(sizeof(int) * A->m);
            for (int i = 0; i < A->m; i++) {
                ITilde[i] = -1;
            }

            n1Tilde = 0;
            for (int j = 0; j < n2Union; j++) {
                for (int i = A->offset[JUnion[j]]; i < A->offset[JUnion[j] + 1]; i++) {
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

            if (n1Tilde == 0) {
                n2Tilde = 0;
                n2Union = n2;
            }

            // print I
            printf("\nI: ");
            for (int i = 0; i < n1; i++) {
                printf("%d ", I[i]);
            }

            // printf ITilde
            printf("\nITilde: ");
            for (int i = 0; i < n1Tilde; i++) {
                printf("%d ", ITilde[i]);
            }

            // f) set IUnion and JUnion
            // make union of I and ITilde
            n1Union = n1 + n1Tilde;
            IUnion = (int*) malloc(sizeof(int) * (n1 + n1Tilde));
            for (int i = 0; i < n1; i++) {
                IUnion[i] = I[i];
            }
            for (int i = 0; i < n1Tilde; i++) {
                IUnion[n1 + i] = ITilde[i];
            }

            // print IUnion
            printf("\nIUnion: ");
            for (int i = 0; i < n1Union; i++) {
                printf("%d ", IUnion[i]);
            }

            // print Q
            printf("\nQ before update: ");
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < n1; j++) {
                    printf("%f ", Q[i * n1 + j]);
                }
                printf("\n");
            }

            // g) Update the QR factorization of A(IUnion, JUnion)
            int updateSuccess = updateQR(cHandle, A, &AHat, &Q, &R, I, J, ITilde, JTilde, IUnion, JUnion, n1, n2, n1Tilde, n2Tilde, n1Union, n2Union, &mHat_k, residual, &residualNorm, k);

            if (updateSuccess != 0) {
                printf("update failed\n");
                return NULL;
            }

            // print Q
            printf("\nQ when we return from update: ");
            for (int i = 0; i < n1Union; i++) {
                for (int j = 0; j < n1Union; j++) {
                    printf("%f ", Q[i * n1Union + j]);
                }
                printf("\n");
            }
            
            // free memory
            free(L);
            printf("L freed\n");
            free(IUnion);
            printf("IUnion freed\n");
            free(JUnion);
            printf("JUnion freed\n");
            free(ITilde);
            printf("ITilde freed\n");
            free(smallestIndices);
            printf("smallestIndices freed\n");
            free(smallestJTilde);
            printf("smallestJTilde freed\n");
            free(rhoSq);
            printf("rhoSq freed\n");
            free(JTilde);
            printf("JTilde freed\n");
            free(keepArray);
            printf("keepArray freed\n");
        }

        // update kth column of M
        // print mHat_k
        printf("\nmHat_k: ");
        for (int i = 0; i < n2; i++) {
            printf("%f ", mHat_k[i]);
        }
        M = updateKthColumnCSC(M, mHat_k, k, J, n2);
        printCSC(M);
        printf("\nM after updateKthColumnCSC:\n");

        // free memory
        free(I);
        printf("I freed\n");
        free(J);
        printf("J freed\n");
        free(AHat);
        printf("AHat freed\n");
        free(Q);
        printf("Q freed\n");
        free(R);
        printf("R freed\n");
        free(mHat_k);
        printf("mHat_k freed\n");
        free(residual);
        printf("all freed\n");

    }
    printCSC(M);

    printf("vi er done\n");
    cublasDestroy(cHandle);

    return M;
}

// idéer:
// - få batched funktionerne til at returnere 0 ved succes og 1 ved fejl
// - brug denne retur værdi til at styre control flowet i resten af koden
// - vi skal muligvis have sat en masse variabler op i toppen af funktionerne, så vi har adgang til dem i hele funktionen, og så opdatere dem, hvis vi er i det rigtige control flow
// - måske skal vi også have lavet noget mere af vores kode om til funktioner, som vi også kan have returnere 0 ved succes og 1 ved fejl

#endif