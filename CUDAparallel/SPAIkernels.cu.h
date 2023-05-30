#ifndef SPAI_KERNELS_CU_H
#define SPAI_KERNELS_CU_H

#include <stdio.h>
#include <cuda.h>
#include <math.h>
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
__global__ void computeIandJ(CSC* d_A, CSC* d_M, int** d_PointerI, int** d_PointerJ, int** d_PointerSortedJ, int* d_n1, int* d_n2, int currentBatch, int batchsize, int maxN2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize) {
        int index = currentBatch * batchsize + tid;
        if (index < maxN2) {
            int n2 = d_M->offset[index + 1] - d_M->offset[index];
            int* d_J = (int*) malloc(n2 * sizeof(int));

            // iterate through the row indeces from offset[k] to offset[k+1] and take all elements from the flatRowIndex
            int h = 0;
            for (int i = d_M->offset[index]; i < d_M->offset[index + 1]; i++) {
                d_J[h] = d_M->flatRowIndex[i];
                h++;
            }

            // We initialize I to -1, and the iterate through all elements of J. Then we iterate through the row indeces of A from the offset J[j] to J[j] + 1. If the row index is already in I, we dont do anything, else we add it to I.
            int* d_I = (int*) malloc(d_A->m * sizeof(int));
            for (int i = 0; i < d_A->m; i++) {
                d_I[i] = -1;
            }

            int n1 = 0;
            for (int j = 0; j < n2; j++) {
                for (int i = d_A->offset[d_J[j]]; i < d_A->offset[d_J[j] + 1]; i++) {
                    int keep = 1;
                    for (int k = 0; k < d_A->m; k++) {
                        if (d_I[k] == d_A->flatRowIndex[i]) {
                            keep = 0;
                            break;
                        }
                    }
                    if (keep) {
                        d_I[n1] = d_A->flatRowIndex[i];
                        n1++;
                    }
                }
            }

            // set device values
            // giver det mening at parallelisere dette?
            d_PointerI[tid] = d_I;
            d_PointerJ[tid] = d_J;
            d_PointerSortedJ[tid] = d_J;
            d_n1[tid] = n1;
            d_n2[tid] = n2;
        } else {
            d_PointerI[tid] = NULL;
            d_PointerJ[tid] = NULL;
            d_PointerSortedJ[tid] = NULL;
            d_n1[tid] = 0;
            d_n2[tid] = 0;
        }
    }
}

// kernel for setting the dense matrices padded with zeros to make the uniform size
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
__global__ void CSCToBatchedDenseMatrices(CSC* d_A, float** d_AHat, int** d_PointerI, int** d_PointerJ, int* d_n1, int* d_n2, int maxn1, int maxn2, int maxOffset, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize * maxn1 * maxn2 * maxOffset) {
        int b = (tid / (maxn1 * maxn2 * maxOffset));
        int i = (tid % (maxn1 * maxn2 * maxOffset)) / (maxn2 * maxOffset);
        int j = ((tid % (maxn1 * maxn2 * maxOffset)) % (maxn2 * maxOffset)) / maxOffset;
        int l = ((tid % (maxn1 * maxn2 * maxOffset)) % (maxn2 * maxOffset)) % maxOffset;

        int n1 = d_n1[b];
        int n2 = d_n2[b];

        if (tid == 0) {
            for (int h = 0; h < batchsize; h++) {
                int* I = d_PointerI[h];
                int* J = d_PointerJ[h];
                printf("batch: %d\n", h);
                // print I
                printf("I: ");
                for (int k = 0; k < d_n1[h]; k++) {
                    printf("%d ", I[k]);
                } 
                printf("\n");

                // print J
                printf("J: ");
                for (int k = 0; k < d_n2[h]; k++) {
                    printf("%d ", J[k]);
                }
                printf("\n");
            }
        }
        __syncthreads();

        int* I = d_PointerI[b];
        int* J = d_PointerJ[b];

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
        int n2Tilde = d_N2Tilde[tid];

        int* d_JTilde = (int*) malloc(n2Tilde * sizeof(int));

        int index = 0;
        for (int i = 0; i < n; i++) {
            if (d_KeepArray[i]) {
                d_JTilde[index] = i;
                index++;
            }
        }

        d_PointerJTilde[tid] = d_JTilde;
    }
}

// kernel for computing rho squared
// d_A                 = device pointer to A
// d_PointerRhoSquared = device pointer pointer to rhoSquared
// d_PointerResidual   = device pointer pointer to residual
// d_PointerJTilde     = device pointer pointer to JTilde
// d_n2Tilde           = device pointer to n2Tilde
// maxn2Tilde          = the maximum length of n2Tilde
// batchsize           = the size of the batch
__global__ void computeRhoSquared(CSC* d_A, float** d_PointerRhoSquared, float** d_PointerResidual, int** d_PointerJTilde, float* d_residualNorm, int* d_n2Tilde, int maxn2Tilde, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize * maxn2Tilde) {
        int b = tid / maxn2Tilde;
        int i = tid % maxn2Tilde;

        float* d_rhoSquared = d_PointerRhoSquared[b];
        float* d_residual = d_PointerResidual[b];
        int* d_JTilde = d_PointerJTilde[b];

        if (i < d_n2Tilde[b]) {
            float rTAe_j = 0.0;
            for (int j = d_A->offset[d_JTilde[i]]; j < d_A->offset[d_JTilde[i] + 1]; j++) {
                rTAe_j += d_A->flatData[j] * d_residual[d_A->flatRowIndex[j]];
            }

            float Ae_jNorm = 0.0;
            for (int j = d_A->offset[d_JTilde[i]]; j < d_A->offset[d_JTilde[i] + 1]; j++) {
                Ae_jNorm += d_A->flatData[j] * d_A->flatData[j];
            }
            Ae_jNorm = sqrt(Ae_jNorm);

            d_rhoSquared[i] = d_residualNorm[b] * d_residualNorm[b] - (rTAe_j * rTAe_j) / (Ae_jNorm * Ae_jNorm);
        } else {
            d_rhoSquared[i] = 0.0;
        }
    }
}

/* kernel for finding the index of the maximum rho squared
d_PointerRhoSquared      = device pointer pointer to rhoSquared
d_PointerSmallestIndices = device pointer pointer to smallestIndices
d_PointerSmallestJTilde  = device pointer pointer to smallestJTilde
d_newN2Tilde             = device pointer to newN2Tilde
d_n2Tilde                = device pointer to n2Tilde
s                        = the number of indices to keep
batchsize                = the size of the batch */
__global__ void computeSmallestIndices(float** d_PointerRhoSquared, int** d_PointerSmallestIndices, int** d_PointerSmallestJTilde, int** d_PointerJTilde, int* d_newN2Tilde, int* d_n2Tilde, int s, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize) {
        float* d_rhoSquared = d_PointerRhoSquared[tid];
        int* d_smallestIndices = d_PointerSmallestIndices[tid];
        int* d_smallestJTilde = d_PointerSmallestJTilde[tid];
        int* d_JTilde = d_PointerJTilde[tid];

        d_newN2Tilde[tid] = MIN(s, d_n2Tilde[tid]);

        for (int i = 0; i < d_newN2Tilde[tid]; i++) {
            d_smallestIndices[i] = -1;
        }

        for (int i = 0; i < d_n2Tilde[tid]; i++) {
            for (int j = 0; j < d_newN2Tilde[tid]; j++) {
                if (d_smallestIndices[j] == -1) {
                    d_smallestIndices[j] = i;
                    break;
                } else if (d_rhoSquared[i] < d_rhoSquared[d_smallestIndices[j]]) {
                    for (int k = d_newN2Tilde[tid] - 1; k > j; k--) {
                        d_smallestIndices[k] = d_smallestIndices[k - 1];
                    }
                    d_smallestIndices[j] = i;
                    break;
                }
            }
        }

        for (int i = 0; i < d_newN2Tilde[tid]; i++) {
            d_smallestJTilde[i] = d_JTilde[d_smallestIndices[i]];
        }
    }
}

/* kernel for finding ITilde and setting IUnion and JUnion
d_A                 = device pointer to A
d_PointerI          = device pointer pointer to I
d_PointerJ          = device pointer pointer to J
d_PointerITilde     = device pointer pointer to ITilde
d_PointerJTilde     = device pointer pointer to JTilde
d_PointerIUnion     = device pointer pointer to IUnion
d_PointerJUnion     = device pointer pointer to JUnion
d_n1                = device pointer to n1
d_n2                = device pointer to n2
d_n1Tilde           = device pointer to n1Tilde
d_n2Tilde           = device pointer to n2Tilde
d_n1Union           = device pointer to n1Union
d_n2Union           = device pointer to n2Union
batchsize           = the size of the batch */
__global__ void computeITilde(CSC* d_A, int** d_PointerI, int** d_PointerJ, int** d_PointerITilde, int** d_PointerJTilde, int** d_PointerIUnion, int** d_PointerJUnion, int* d_n1, int* d_n2, int* d_n1Tilde, int* d_n2Tilde, int* d_n1Union, int* d_n2Union, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize) {
        int n1 = d_n1[tid];
        int n2 = d_n2[tid];
        int n2Tilde = d_n2Tilde[tid];

        int* d_I = d_PointerI[tid];
        int* d_J = d_PointerJ[tid];
        int* d_JTilde = d_PointerJTilde[tid];
        
        // set d_n2Union and make JUnion
        int n2Union = n2 + n2Tilde;
        int* d_JUnion = (int*) malloc(n2Union* sizeof(int));
        for (int i = 0; i < n2; i++) {
            d_JUnion[i] = d_J[i];
        }

        for (int i = 0; i < n2Tilde; i++) {
            d_JUnion[n2 + i] = d_JTilde[i];
        }

        // find ITilde 
        int* d_ITilde = (int*) malloc(d_A->m * sizeof(int));
        for (int i = 0; i < d_A->m; i++) {
            d_ITilde[i] = -1;
        }

        int n1Tilde = 0;
        for (int j = 0; j < n2Union; j++) {
            for (int i = d_A->offset[d_JUnion[j]]; i < d_A->offset[d_JUnion[j] + 1]; i++) {
                int keep = 1;
                for (int h = 0; h < n1; h++) {
                    if (d_A->flatRowIndex[i] == d_I[h] || d_A->flatRowIndex[i] == d_ITilde[h]) {
                        keep = 0;
                        break;
                    }
                }
                if (keep) {
                    d_ITilde[n1Tilde] = d_A->flatRowIndex[i];
                    n1Tilde++;
                }
            }
        }

        // set d_n1Tilde and make IUnion
        int n1Union = n1 + n1Tilde;
        int* d_IUnion = (int*) malloc(n1Union * sizeof(int));
        for (int i = 0; i < n1; i++) {
            d_IUnion[i] = d_I[i];
        }

        for (int i = 0; i < n1Tilde; i++) {
            d_IUnion[n1 + i] = d_ITilde[i];
        }

        // copy the values to array
        d_n1Tilde[tid] = n1Tilde;
        d_n1Union[tid] = n1Union;
        d_n2Union[tid] = n2Union;

        d_PointerITilde[tid] = d_ITilde;
        d_PointerJUnion[tid] = d_JUnion;
        d_PointerIUnion[tid] = d_IUnion;
    }
}

// kører parallelt i batchsize * maxn1 * maxn2Tilde tråde
/* kernel for computing ABreve = Q^T * A(I, JTilde)
d_PointerQ        = device pointer pointer to Q
d_PointerAIJTilde = device pointer pointer to AIJTilde
d_PointerABreve   = device pointer pointer to ABreve
d_n1              = device pointer to n1
d_n2Tilde         = device pointer to n2Tilde
maxn1             = the maximum value of n1 in the batch
maxn2Tilde        = the maximum value of n2Tilde in the batch 
batchsize         = the size of the batch */
__global__ void computeABreve(float** d_PointerQ, float** d_PointerAIJTilde, float** d_PointerABreve, int* d_n1, int* d_n2Tilde, int maxn1, int maxn2Tilde, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize * maxn1 * maxn2Tilde) {
        int b = tid / (maxn1 * maxn2Tilde);
        int i = (tid % (maxn1 * maxn2Tilde)) / maxn2Tilde;
        int j = (tid % (maxn1 * maxn2Tilde)) % maxn2Tilde;

        int n1 = d_n1[b];
        int n2Tilde = d_n2Tilde[b];
        
        float* d_Q = d_PointerQ[b];
        float* d_AIJTilde = d_PointerAIJTilde[b];
        float* d_ABreve = d_PointerABreve[b];

        if (i < n1 && j < n2Tilde) {
            d_ABreve[i * maxn2Tilde + j] = 0.0;
            for (int k = 0; k < n1; k++) {
                d_ABreve[i * maxn2Tilde + j] += d_Q[k * n1 + i] * d_AIJTilde[k * maxn2Tilde + j];
                printf("%f = %f * %f\n", d_ABreve[i * maxn2Tilde + j], d_Q[k * maxn1 + i], d_AIJTilde[k * maxn2Tilde + j]);
            }
        }
    }
}

// kører parallelt i batchsize * maxn2 * maxn2Tilde tråde
/* kernel for setting B1 = ABreve[0:n2, 0:n2Tilde]
d_PointerABreve = device pointer pointer to ABreve
d_PointerB1     = device pointer pointer to B1
d_n2            = device pointer to n2
d_n2Tilde       = device pointer to n2Tilde
maxn2           = the maximum value of n2 in the batch
maxn2Tilde      = the maximum value of n2Tilde in the batch
batchsize       = the size of the batch */
__global__ void setB1(float** d_PointerABreve, float** d_PointerB1, int* d_n2, int* d_n2Tilde, int maxn2, int maxn2Tilde, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize * maxn2 * maxn2Tilde) {
        int b = tid / (maxn2 * maxn2Tilde);
        int i = (tid % (maxn2 * maxn2Tilde)) / maxn2;
        int j = (tid % (maxn2 * maxn2Tilde)) % maxn2;

        int n2 = d_n2[b];
        int n2Tilde = d_n2Tilde[b];

        float* d_ABreve = d_PointerABreve[b];
        float* d_B1 = d_PointerB1[b];

        if (i < n2 && j < n2Tilde) {
            d_B1[i * maxn2Tilde + j] = d_ABreve[i * maxn2Tilde + j];
        }
    }
}

// kører parallelt i batchsize * maxn1Union * maxn2Tilde tråde
/* kernel for setting B2 = ABreve[n2 + 1:n1, 0:n2Tilde] + A(ITilde, JTilde)
d_PointerABreve        = device pointer pointer to ABreve
d_PointerAITildeJTilde = device pointer pointer to AITildeJTilde
d_PointerB2            = device pointer pointer to B2
d_n1Union              = device pointer to n1Union
d_n1                   = device pointer to n1
d_n2                   = device pointer to n2
d_n2Tilde              = device pointer to n2Tilde
maxn1Union             = the maximum value of n1Union in the batch
maxn2Tilde             = the maximum value of n2 in the batch
batchsize              = the size of the batch */
__global__ void setB2(float** d_PointerABreve, float** d_PointerAITildeJTilde, float** d_PointerB2, int* d_n1, int* d_n1Union, int* d_n2, int* d_n2Tilde, int maxn1Union, int maxn2Tilde, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize * maxn1Union * maxn2Tilde) {
        int b = tid / (maxn1Union * maxn2Tilde);
        int i = (tid % (maxn1Union * maxn2Tilde)) / maxn2Tilde;
        int j = (tid % (maxn1Union * maxn2Tilde)) % maxn2Tilde;

        int n1 = d_n1[b];
        int n1Union = d_n1Union[b];
        int n2 = d_n2[b];
        int n2Tilde = d_n2Tilde[b];

        float* d_ABreve = d_PointerABreve[b];
        float* d_AITildeJTilde = d_PointerAITildeJTilde[b];
        float* d_B2 = d_PointerB2[b];

        if (i < maxn1Union && j < maxn2Tilde) {
            if (i < n1 - n2 && j < n2Tilde) {
                d_B2[i * maxn2Tilde + j] = d_ABreve[(n2 + i) * maxn2Tilde + j];
            } else if (i < n1Union && j < n2Tilde){
                d_B2[i * maxn2Tilde + j] = d_AITildeJTilde[(i - (n1 - n2)) * maxn2Tilde + j];
            } else {
                d_B2[i * maxn2Tilde + j] = 0.0;
            }
        }
    }
}

// parallelt i batchsize * maxn1Union * maxn1Union tråde
/* kernel for setting the firstMatrix with Q in the upper left corner and identity in the lower right corner
d_PointerFirstMatrix = device pointer pointer to firstMatrix
d_PointerQ           = device pointer pointer to Q
d_n1                 = device pointer to n1
d_n1Union            = device pointer to n1Union
maxn1Union           = the maximum value of n1Union in the batch
batchsize            = the size of the batch */
__global__ void setFirstMatrix(float** d_PointerFirstMatrix, float** d_PointerQ, int* d_n1, int* d_n1Union, int maxn1Union, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize * maxn1Union * maxn1Union) {
        int b = tid / (maxn1Union * maxn1Union);
        int i = (tid % (maxn1Union * maxn1Union)) / maxn1Union;
        int j = (tid % (maxn1Union * maxn1Union)) % maxn1Union;

        int n1 = d_n1[b];
        int n1Union = d_n1Union[b];

        float* d_FirstMatrix = d_PointerFirstMatrix[b];
        float* d_Q = d_PointerQ[b];

        if (i < n1Union && j < n1Union) {
            if (i < n1 && j < n1) {
                d_FirstMatrix[i * n1Union + j] = d_Q[i * n1 + j];
            } else if (i == j) {
                d_FirstMatrix[i * n1Union + j] = 1.0;
            } else {
                d_FirstMatrix[i * n1Union + j] = 0.0;
            }
        } 
    }
}

// parallelt i batchsize * maxn1Union * maxn1Union tråde
/* kernel for setting the secondMatrix with identity in the upper left corner and QB in the lower right corner
d_PointerSecondMatrix = device pointer pointer to secondMatrix
d_PointerQB           = device pointer pointer to QB
d_n1Tilde             = device pointer to n1
d_n1Union             = device pointer to n1Union
maxn1Union            = the maximum value of n1Union in the batch
batchsize             = the size of the batch */
__global__ void setSecondMatrix(float** d_PointerSecondMatrix, float** d_PointerB2Q, int* d_n1Tilde, int* d_n1Union, int* d_n2, int maxn1Union, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize * maxn1Union * maxn1Union) {
        int b = tid / (maxn1Union * maxn1Union);
        int i = (tid % (maxn1Union * maxn1Union)) / maxn1Union;
        int j = (tid % (maxn1Union * maxn1Union)) % maxn1Union;

        int n1Tilde = d_n1Tilde[b];
        int n1Union = d_n1Union[b];
        int n2 = d_n2[b];

        float* d_SecondMatrix = d_PointerSecondMatrix[b];
        float* d_B2Q = d_PointerB2Q[b];

        if (i < n1Union && j < n1Union) {
            d_SecondMatrix[i * n1Union + j] = 0.0;
            if (i < n1Union - n2 && j < n1Union - n2) {
                d_SecondMatrix[(i + n2) * n1Union + (j + n2)] = d_B2Q[i * (n1Union - n2) + j];
            } else if (i == j) {
                d_SecondMatrix[(i - (n1Union - n2)) * n1Union + j - (n1Union - n2)] = 1.0;
            } 
        }
    }
}

// parallelt i batchsize * maxn1Union * maxn1Union tråde
/* kernel for computing matrix multiplication
d_PointerA = device pointer pointer to matrix A
d_PointerB = device pointer pointer to matrix B
d_PointerC = device pointer pointer to the result matrix C
d_dim1     = device pointer to the first dimension of A (NULL if dim1 is 1)
d_dim2     = device pointer to the second dimension of A and the first dimension of B (NULL if dim2 is 1)
d_dim3     = device pointer to the second dimension of B (NULL if dim3 is 1)
maxdim1    = the maximum value of the first dimension of A in the batch 
maxdim2    = the maximum value of the second dimension of A and the first dimension of B in the batch
maxdim3    = the maximum value of the second dimension of B in the batch
batchsize  = the size of the batch */
__global__ void matrixMultiplication (float** d_PointerA, float** d_PointerB, float** d_PointerC, int* d_dim1, int* d_dim2, int* d_dim3, int maxdim1, int maxdim2, int maxdim3, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize * maxdim1 * maxdim3) {
        int b = tid / (maxdim1 * maxdim3);
        int i = (tid % (maxdim1 * maxdim3)) / maxdim1;
        int j = (tid % (maxdim1 * maxdim3)) % maxdim1;
 
        int dim1;
        int dim2;
        int dim3;

        if (d_dim1 == NULL) {
            dim1 = 1;
        } else {
            dim1 = d_dim1[b];
        }

        if (d_dim2 == NULL) {
            dim2 = 1;
        } else {
            dim2 = d_dim2[b];
        }

        if (d_dim3 == NULL) {
            dim3 = 1;
        } else {
            dim3 = d_dim3[b];
        }

        float* d_A = d_PointerA[b];
        float* d_B = d_PointerB[b];
        float* d_C = d_PointerC[b];

        if (i < dim1 && j < dim3) {
            float sum = 0.0;
            for (int k = 0; k < dim2; k++) {
                sum += d_A[i * dim2 + k] * d_B[k * dim3 + j];
            }
            d_C[i * dim3 + j] = sum;
        }
    }
}

// parallelt i batchsize * maxn1Union * maxn2Union tråde
/* kernel for setting unsorted R with R in the upper left corner, B1 in the upper right corner, B2R below B1 and zeros the rest
d_PointerUnsortedR = device pointer pointer to unsorted R
d_PointerR         = device pointer pointer to R
d_PointerB1        = device pointer pointer to B1
d_PointerB2R       = device pointer pointer to B2R
d_n1               = device pointer to n1
d_n1Union          = device pointer to n1Union
d_n2               = device pointer to n2
d_n2Union          = device pointer to n2Union
d_n2Tilde          = device pointer to n2Tilde
maxn1Union         = the maximum value of n1Union in the batch
maxn2Union         = the maximum value of n2Union in the batch
batchsize          = the size of the batch */
__global__ void setUnsortedR(float** d_PointerUnsortedR, float** d_PointerR, float** d_PointerB1, float** d_PointerB2R, int* d_n1, int* d_n1Union, int* d_n2, int* d_n2Union, int* d_n2Tilde, int maxn1Union, int maxn2Union, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize * maxn1Union * maxn2Union) {
        int b = tid / (maxn1Union * maxn2Union);
        int i = (tid % (maxn1Union * maxn2Union)) / maxn2Union;
        int j = (tid % (maxn1Union * maxn2Union)) % maxn2Union;

        int n1 = d_n1[b];
        int n1Union = d_n1Union[b];
        int n2 = d_n2[b];
        int n2Union = d_n2Union[b];
        int n2Tilde = d_n2Tilde[b];

        float* d_UnsortedR = d_PointerUnsortedR[b];
        float* d_R = d_PointerR[b];
        float* d_B1 = d_PointerB1[b];
        float* d_B2R = d_PointerB2R[b];

        if (i < n1Union && j < n2Union) {
            if (i < n2 && j < n2) {
                d_UnsortedR[i * n2Union + j] = d_R[i * n2 + j];
            } else if (i < n2 && j < n2Union && j > n2 - 1) {
                d_UnsortedR[i * n2Union + j] = d_B1[i * n2Tilde + j - n2];
            } else if (i < n1Union && j < n2Union && j > n2 - 1) {
                d_UnsortedR[i * n2Union + j] = d_B2R[(i - n2) * n2Tilde + j - n2];
            } else {
                d_UnsortedR[i * n2Union + j] = 0.0;
            }
        }
    }
}

// parallelt i batchsize * maxn2Union
/* kernel for permuting J
d_PointerSortedJ = device pointer pointer to sorted J
d_PointerJ       = device pointer pointer to J
d_PointerPc      = device pointer pointer to the permutation matrix
d_n2             = device pointer to n2
maxn2            = the maximum value of n2 in the batch
batchsize        = the size of the batch */
__global__ void permuteJ(int** d_PointerSortedJ, int** d_PointerJ, float** d_PointerPc, int* d_n2, int maxn2, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize * maxn2) {
        int b = tid / maxn2;
        int i = tid % maxn2;

        int n2 = d_n2[b];

        int* d_SortedJ = d_PointerSortedJ[b];
        int* d_J = d_PointerJ[b];
        float* d_Pc = d_PointerPc[b];

        if (i < n2) {
            d_SortedJ[i] = 0;
            for (int j = 0; j < n2; j++) {
                d_SortedJ[i] += (int) (d_Pc[i * n2 + j] + 0.05) * d_J[j];
            }
        }
    }
}

// parallelt i batchsize 
/* kernel for copying IUnion to I and JUnion to J
d_PointerI      = device pointer pointer to I
d_PointerJ      = device pointer pointer to J
d_PointerIUnion = device pointer pointer to IUnion
d_PointerJUnion = device pointer pointer to JUnion
d_n1Union       = device pointer to n1Union
d_n2Union       = device pointer to n2Union
batchsize       = the size of the batch */
__global__ void copyIandJ(int** d_PointerI, int** d_PointerJ, int** d_PointerIUnion, int** d_PointerJUnion, int* d_n1Union, int* d_n2Union, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize) {
        int n1Union = d_n1Union[tid];
        int n2Union = d_n2Union[tid];

        int* d_I = d_PointerI[tid];
        int* d_J = d_PointerJ[tid];
        int* d_IUnion = d_PointerIUnion[tid];
        int* d_JUnion = d_PointerJUnion[tid];

        free(d_I);
        free(d_J);
        d_I = (int*) malloc(n1Union * sizeof(int));
        d_J = (int*) malloc(n2Union * sizeof(int));
        for (int i = 0; i < n1Union; i++) {
            d_I[i] = d_IUnion[i];
        }
        for (int i = 0; i < n2Union; i++) {
            d_J[i] = d_JUnion[i];
        }

        d_PointerI[tid] = d_I;
        d_PointerJ[tid] = d_J;
    }
}

#endif