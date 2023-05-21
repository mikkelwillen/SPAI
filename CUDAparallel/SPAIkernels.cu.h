#ifndef SPAI_KERNELS_CU_H
#define SPAI_KERNELS_CU_H

#include <stdio.h>
#include <cuda.h>
#include "csc.cu.h"

// kernel for computing I, J n1 and n2
// d_A          = device pointer to A
// d_M          = device pointer to M
// d_PointerI   = device pointer pointer to I
// d_PointerJ   = device pointer pointer to J
// d_n1         = device pointer to n1
// d_n2         = device pointer to n2
// currentBatch = the current batch
// batchsize    = the size of the batch
__global__ void computeIandJ(CSC* d_A, CSC* d_M, int** d_PointerI, int** d_PointerJ, int* d_n1, int* d_n2, int currentBatch, int batchsize, int maxN2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize) {
        int index = currentBatch * batchsize + tid;
        if (index < maxN2) {
            int n2 = d_M->offset[index + 1] - d_M->offset[index];
            int* J = (int*) malloc(n2 * sizeof(int));

            // iterate through the row indeces from offset[k] to offset[k+1] and take all elements from the flatRowIndex
            int h = 0;
            for (int i = d_M->offset[index]; i < d_M->offset[index + 1]; i++) {
                J[h] = d_M->flatRowIndex[i];
                h++;
            }

            // We initialize I to -1, and the iterate through all elements of J. Then we iterate through the row indeces of A from the offset J[j] to J[j] + 1. If the row index is already in I, we dont do anything, else we add it to I.
            int* I = (int*) malloc(d_A->m * sizeof(int));
            for (int i = 0; i < d_A->m; i++) {
                I[i] = -1;
            }

            int n1 = 0;
            for (int j = 0; j < n2; j++) {
                for (int i = d_A->offset[J[j]]; i < d_A->offset[J[j] + 1]; i++) {
                    int keep = 1;
                    for (int k = 0; k < d_A->m; k++) {
                        if (I[k] == d_A->flatRowIndex[i]) {
                            keep = 0;
                            break;
                        }
                    }
                    if (keep) {
                        I[n1] = d_A->flatRowIndex[i];
                        n1++;
                    }
                }
            }

            // set device values
            // giver det mening at parallelisere dette?
            d_PointerI[tid] = &I[0];
            d_PointerJ[tid] = &J[0];
            d_n1[tid] = n1;
            d_n2[tid] = n2;
        } else {
            d_PointerI[tid] = NULL;
            d_PointerJ[tid] = NULL;
            d_n1[tid] = 0;
            d_n2[tid] = 0;
        }
    }
}

// kernel for computing Ahat
// d_A          = device pointer to A
// d_AHat       = device pointer pointer to AHat
// d_PointerI   = device pointer pointer to I
// d_PointerJ   = device pointer pointer to J
// d_n1         = device pointer to n1
// d_n2         = device pointer to n2
// maxn1        = the maximum value of n1
// maxn2        = the maximum value of n2
// maxOffset    = the maximum value of offset
// currentBatch = the current batch
// batchsize    = the size of the batch
__global__ void computeAHat(CSC* d_A, float** d_AHat, int** d_PointerI, int** d_PointerJ, int* d_n1, int* d_n2, int maxn1, int maxn2, int maxOffset, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize * maxn1 * maxn2 * maxOffset) {
        int b = (tid / (maxn1 * maxn2 * maxOffset));
        int i = (tid % (maxn1 * maxn2 * maxOffset)) / (maxn2 * maxOffset);
        int j = ((tid % (maxn1 * maxn2 * maxOffset)) % (maxn2 * maxOffset)) / maxOffset;
        int l = ((tid % (maxn1 * maxn2 * maxOffset)) % (maxn2 * maxOffset)) % maxOffset;

        int n1 = d_n1[b];
        int n2 = d_n2[b];

        int* I = d_PointerI[b];
        int* J = d_PointerJ[b];
        if (tid == 0) {
            // print I
            printf("I: ");
            for (int k = 0; k < d_n1[b]; k++) {
                printf("%d ", I[k]);
            } 
            printf("\n");

            // print J
            printf("J: ");
            for (int k = 0; k < d_n2[b]; k++) {
                printf("%d ", J[k]);
            }
            printf("\n");
        }

        float* AHat = d_AHat[b];

        if (l == 0) {
            AHat[i * maxn2 + j] = 0.0;
        }
        __syncthreads();

        if (i < d_n1[b] && j < d_n2[b]) {
            int offset = d_A->offset[J[j]];
            int offsetDiff = d_A->offset[J[j] + 1] - offset;
            if (tid == 0) {
                printf("offsetDiff: %d\n", offsetDiff);
            }
            if (l < offsetDiff) {
                if (I[i] == d_A->flatRowIndex[l + offset]) {
                    AHat[i * d_n2[b] + j] += d_A->flatData[l + offset];
                }
            }
        }
    }
}

// kernel for freeing I and J
// d_PointerI = device pointer pointer to I
// d_PointerJ = device pointer pointer to J
// batchsize  = the size of the batch
__global__ void freeIJ(int** d_PointerI, int** d_PointerJ, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize) {
        if (d_PointerI[tid] != NULL) {
            free(d_PointerI[tid]);
        }
        if (d_PointerJ[tid] != NULL) {
            free(d_PointerJ[tid]);
        }
    }
}

// kernel for finding the length of L
// d_l               = device pointer to l
// d_PointerResidual = device pointer pointer to residual
// d_PointerI        = device pointer pointer to I
// m                 = the number of rows in A
// n1                = the number of rows in AHat
// currentBatch      = the current batch
// batchsize         = the size of the batch
__global__ void computeLengthOfL(int* d_l, float** d_PointerResidual, int** d_PointerI, int** d_PointerL, int m, int* d_n1, int currentBatch, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize) {
        int k = currentBatch * batchsize + tid;
        int kNotInI = 0;

        float* d_residual = d_PointerResidual[tid];
        int* d_I = d_PointerI[tid];

        int l = 0;
        for (int i = 0; i < m; i++) {
            if (d_residual[i] != 0.0) {
                l++;
            } else if (k == i) {
                kNotInI = 1;
            }
        }

        // check if k is in I
        for (int i = 0; i < d_n1[tid]; i++) {
            if (k == d_I[i]) {
                kNotInI = 0;
                break;
            }
        }

        if (kNotInI) {
            l++;
        }

        d_l[tid] = l;

        // malloc space for L and fill it
        int* d_L = (int*) malloc(l * sizeof(int));

        int index = 0;
        for (int i = 0; i < m; i++) {
            if (d_residual[i] != 0.0) {
                d_L[index] = i;
                index++;
            } else if (k == i && kNotInI) {
                d_L[index] = i;
                index++;
            }
        }

        d_PointerL[tid] = d_L;
    }
}

__global__ void computeKeepArray(CSC* d_A, int** d_PointerKeepArray, int** d_PointerL, int** d_PointerJ, int* d_n2, int* d_l, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = d_A->n;
    if (tid < n * batchsize) {
        int b = tid / n;
        int i = tid % n;

        int* d_KeepArray = d_PointerKeepArray[b];
        int* d_L = d_PointerL[b];
        int* d_J = d_PointerJ[b];
        int n2 = d_n2[b];

        d_KeepArray[i] = 0;

        for (int j = 0; j < d_l[b]; j++) {
            for (int h = d_A->offset[i]; h < d_A->offset[i + 1]; h++) {
                if (d_A->flatRowIndex[h] == d_L[j]) {
                    d_KeepArray[i] = 1;
                }
            }
        }

        // remove the indeces that are already in J
        for (int i = 0; i < n2; i++) {
            d_KeepArray[d_J[i]] = 0;
        }
    }
}

// kernel for finding the length of n2Tilde
// d_PointerKeepArray = device pointer pointer to keepArray
// d_n2Tilde          = device pointer to n2Tilde
// batchsize          = the size of the batch
__global__ void computeN2Tilde(int** d_PointerKeepArray, int* d_n2Tilde, int n, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize) {
        int* d_KeepArray = d_PointerKeepArray[tid];

        int n2Tilde = 0;
        for (int i = 0; i < n; i++) {
            if (d_KeepArray[i]) {
                n2Tilde++;
            }
        }

        d_n2Tilde[tid] = n2Tilde;
    }
}

// kernel for setting JTilde
// d_PointerKeepArray = device pointer pointer to keepArray
// d_PointerJTilde    = device pointer pointer to JTilde
// n                  = the number of columns in A
// batchsize          = the size of the batch
__global__ void computeJTilde(int** d_PointerKeepArray, int** d_PointerJTilde, int* d_N2Tilde, int n, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize) {
        int* d_KeepArray = d_PointerKeepArray[tid];
        int* d_JTilde = d_PointerJTilde[tid];
        int n2Tilde = d_N2Tilde[tid];

        d_JTilde = (int*) malloc(n2Tilde * sizeof(int));

        int index = 0;
        for (int i = 0; i < n; i++) {
            if (d_KeepArray[i]) {
                d_JTilde[index] = i;
                index++;
            }
        }
    }
}