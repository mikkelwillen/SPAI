#ifndef SPAI_H
#define SPAI_H

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"


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
        printf("cuBLAS initialization failed\n");
        printf("cuBLAS error: %d\n", stat);
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

        // // print I
        // printf("\nI: ");
        // for (int i = 0; i < n1; i++) {
        //     printf("%d ", I[i]);
        // }

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

        // // d) do QR decomposition of AHat
        // // set variables
        // int lda = n1;
        // int ltau = MAX(1, MIN(n1, n2));
        // float* d_AHat;
        // float* d_Tau;
        // int info;

        // // qr initialization
        // cudaMalloc((void**) &d_AHat, n1 * n2 * BATCHSIZE * sizeof(float));
        // cudaMalloc((void**) &d_Tau, ltau * BATCHSIZE * sizeof(float));
        // cudaMemcpy(d_AHat, AHat, sizeof(AHat), cudaMemcpyHostToDevice);
        // stat = cublasSgeqrfBatched(cHandle,
        //                            n1,
        //                            n2,
        //                            &d_AHat,
        //                            lda,
        //                            &d_Tau,
        //                            &info,
        //                            BATCHSIZE);
                                   
        // if (stat != CUBLAS_STATUS_SUCCESS) {
        //     printf("\ncublasSgeqrfBatched failed");
        // }

        // Placeholder Q. Q is n1 x n2 (and Q1 is n2 x n2)
        float* Q = (float*) malloc(sizeof(float) * n1 * n2);

        if (k == 0) {
            Q[0] = 1.0;
        }
        else if (k == 1) {
            Q[0] = 0.6;
            Q[1] = 0.8;
        }
        else if (k == 2) {
            Q[0] = 1;
        }
        printf("\nQ: ");
        for (int i = 0; i < n1; i++) {
            printf("%f ", Q[i]);
        }

        // e) compute ĉ = Q^T ê_k
        // make e_k and set index k to 1.0
        printf("\nk: %d", k);
        float* e_k = (float*) calloc(n1, sizeof(float));
        for (int i = 0; i < n1; i++) {
            if (k == I[i]) {
                e_k[i] = 1.0;
            } else {
                e_k[i] = 0.0;
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

        // Placeholder R. R is n2 x n2
        float* R1 = (float*) malloc(sizeof(float) * n2 * n2);

        if (k == 0) {
            R1[0] = 20.0;
        }
        else if (k == 1) {
            R1[0] = 50.0;
        }
        else if (k == 2) {
            R1[0] = 10.0;
        }
        printf("\nR1: ");
        for (int i = 0; i < n2; i++) {
            printf("%f ", R1[i]);
        }

        // f) compute mHat_k = R^-1 cHat
        // Malloc space for mHat_k
        float* mHat_k = (float*) malloc(sizeof(float) * n2);
        
        // Make the inverse
        // Placeholder InvR. R is n2 x n2
        // Malloc space for the inverse of R
        float* invR1 = (float*) malloc(sizeof(float) * n2 * n2);

        if (k == 0) {
            invR1[0] = 0.05;
        }
        else if (k == 1) {
            invR1[0] = 0.02;
        }
        else if (k == 2) {
            invR1[0] = 0.1;
        }
        printf("\ninvR: ");
        for (int i = 0; i < n2; i++) {
            printf("%f ", invR1[i]);
        }

        // Matrix multiplication
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n2; j++) {
                mHat_k[i] = invR1[i * n2 + j] * cHat[i];
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
        float norm = 0.0;
        for (int i = 0; i < A->m; i++) {
            norm += residual[i] * residual[i];
        }
        norm = sqrt(norm);
        printf("\nnorm: %f", norm);
        
        int iteration = 0;
        // while norm of residual > tolerance do
        while (norm > tolerance && maxIteration + 1 > iteration) {
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
            for (int i = 0; i < A->m; i++) {
                if (residual[i] != 0.0) {
                    L[index] = i;
                    index++;
                }
            }

            // print L
            for (int i = 0; i < l; i++) {
                printf("\nL[%d]: %d", i, L[i]);
            }

            // b) Set JTilde to the set of columns of A corresponding to the indices in L that are not already in J
            // compute the length of JTilde
            int n2Tilde = 0;
            for (int i = 0; i < l; i++) {
                for (int j = 0; j < A->n; j++) {
                    // skal lave noget logik for at indeksere i A
                }
            }
        }

        // Husk kun at bruge de s mindste residuals. Kig på hvordan man laver L igen


    }
    
    return M;
}

#endif