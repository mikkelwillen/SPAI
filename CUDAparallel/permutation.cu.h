#ifndef PERMUTATION_H
#define PERMUTATION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "constants.cu.h"

// kører parallelt batched
/* function for creating the row permutation matrix
d_PointerPr = pointer to the row permutation matrix
d_PointerI = pointer to the row permutation vector
d_n1Union = pointer to the number of rows in the original matrix
maxn1 = maximum number of rows in the batch
batchsize = number of matrices in the batch */
__global__ void createPr(float** d_PointerPr, int** d_PointerI, int* d_n1Union, int maxn1, int batchsize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < batchsize) {
        float* d_Pr = d_PointerPr[tid];
        int* d_I = d_PointerI[tid];
        int n1 = d_n1Union[tid];

        // create normalized index of I
        int* IIndex = (int*) malloc(n1 * sizeof(int));
        int prevLowest = -1;
        for (int i = 0; i < n1; i++) {
            int currentLowest = INT_MAX;
            for (int j = 0; j < n1; j++) {
                if (d_I[j] > prevLowest && d_I[j] < currentLowest) {
                    currentLowest = d_I[j];
                    IIndex[j] = i;
                }
            }

            prevLowest = currentLowest;
        }

        // create row permutation matrix of size n1 x n1
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n1; j++) {
                if (IIndex[j] == i) {
                    d_Pr[i * maxn1 + j] = 1;
                }
                else {
                    d_Pr[i * maxn1 + j] = 0;
                }
            }
        }
    }
}

// kører parallelt batched
/* function for creating the column permutation matrix
d_PointerPc = pointer to the column permutation matrix
d_PointerJ = pointer to the column permutation vector
d_n2Union = pointer to the number of columns in the original matrix
maxn2 = maximum number of columns in the batch
batchsize = number of matrices in the batch */
__global__ void createPc(float** d_PointerPc, int** d_PointerJ, int* d_n2Union, int maxn2, int batchsize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < batchsize) {
        float* d_Pc = d_PointerPc[tid];
        int* d_J = d_PointerJ[tid];
        int n2 = d_n2Union[tid];

        // create normalized index of J
        int* JIndex = (int*) malloc(n2 * sizeof(int));
        int prevLowest = -1;
        for (int i = 0; i < n2; i++) {
            int currentLowest = INT_MAX;
            for (int j = 0; j < n2; j++) {
                if (d_J[j] > prevLowest && d_J[j] < currentLowest) {
                    currentLowest = d_J[j];
                    JIndex[j] = i;
                }
            }

            prevLowest = currentLowest;
        }

        // create column permutation matrix of size n2 x n2
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n2; j++) {
                if (JIndex[j] == i) {
                    d_Pc[i * maxn2 + j] = 1;
                }
                else {
                    d_Pc[i * maxn2 + j] = 0;
                }
            }
        }
    }
}

/* function for creating the permutation matrices
d_PointerPr = pointer to the row permutation matrix (NULL if not needed)
d_PointerPc = pointer to the column permutation matrix (NULL if not needed)
d_PointerIUnion = pointer to the row permutation vector
d_PointerJUnion = pointer to the column permutation vector
d_n1Union = pointer to the number of rows in the original matrix
d_n2Union = pointer to the number of columns in the original matrix
maxn1 = maximum number of rows in the batch
maxn2 = maximum number of columns in the batch
batchsize = number of matrices in the batch */
void* createPermutationMatrices(float** d_PointerPr, float** d_PointerPc, int** d_PointerIUnion, int** d_PointerJUnion, int* d_n1Union, int* d_n2Union, int maxn1, int maxn2, int batchsize) {
    int numBlocks;

    if (d_PointerPr != NULL) {
        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        createPr<<<numBlocks, BLOCKSIZE>>>(d_PointerPr, d_PointerIUnion, d_n1Union, maxn1, batchsize);
    }

    if (d_PointerPc != NULL) {
        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        createPc<<<numBlocks, BLOCKSIZE>>>(d_PointerPc, d_PointerJUnion, d_n2Union, maxn2, batchsize);
    }
}

#endif