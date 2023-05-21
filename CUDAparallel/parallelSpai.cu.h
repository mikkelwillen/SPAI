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
#include "helperKernels.cu.h"

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
        d_L = (int*) malloc(l * sizeof(int));

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

        // // remove the indeces that are already in J
        // for (int i = 0; i < n2; i++) {
        //     d_KeepArray[d_J[i]] = 0;
        // }
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

// A            = matrix we want to compute SPAI on
// m, n         = size of array
// tolerance    = tolerance
// maxIteration = constraint for the maximal number of iterations
// s            = number of rho_j - the most profitable indices
// batchsize    = number of matrices to be processed in parallel
CSC* parallelSpai(CSC* A, float tolerance, int maxIterations, int s, const int batchsize) {
    printCSC(A);

    // initialize cublas
    cublasHandle_t cHandle;
    cublasStatus_t stat;
    stat = cublasCreate(&cHandle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        printf("CUBLAS error: %d\n", stat);

        return NULL;
    }

    int numBlocks;
    // initialize M and set to diagonal
    CSC* M = createDiagonalCSC(A->m, A->n);
    printf("after m\n");
    // copy A to device
    // de her skal lige fixes
    CSC* d_A = copyCSCFromHostToDevice(A);
    printf("after d_A\n");
    CSC* d_M = copyCSCFromHostToDevice(M);
    printf("after d_M\n");
    
    // compute the number of batches
    int numberOfBatches = (A->n + batchsize - 1) / batchsize;

    for (int i = 0; i < numberOfBatches; i++) {
        printf("---------BATCH: %d---------\n", i);
        int iteration = 0;
        int** d_PointerI;
        int** d_PointerJ;
        int* d_n1;
        int* d_n2;

        // malloc space
        gpuAssert(
            cudaMalloc((void**) &d_PointerI, batchsize * sizeof(int*)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerJ, batchsize * sizeof(int*)));
        gpuAssert(
            cudaMalloc((void**) &d_n1, batchsize * sizeof(int)));
        gpuAssert(
            cudaMalloc((void**) &d_n2, batchsize * sizeof(int)));
        
        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        computeIandJ<<<numBlocks, BLOCKSIZE>>>(d_A, d_M, d_PointerI, d_PointerJ, d_n1, d_n2, i, batchsize, A->n);

        // find the max value of n1 and n2
        int* n1 = (int*) malloc(batchsize * sizeof(float));
        int* n2 = (int*) malloc(batchsize * sizeof(float));

        gpuAssert(
            cudaMemcpy(n1, d_n1, batchsize * sizeof(float), cudaMemcpyDeviceToHost));
        gpuAssert(
            cudaMemcpy(n2, d_n2, batchsize * sizeof(float), cudaMemcpyDeviceToHost));

        int maxn1 = 0;
        int maxn2 = 0;
        for (int j = 0; j < batchsize; j++) {
            if (n1[j] > maxn1) {
                maxn1 = n1[j];
            }
            if (n2[j] > maxn2) {
                maxn2 = n2[j];
            }
        }

        // create d_AHat
        float* d_AHat;
        float** d_PointerAHat;

        gpuAssert(
            cudaMalloc((void**) &d_AHat, batchsize * maxn1 * maxn2 * sizeof(float)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerAHat, batchsize * sizeof(float*)));

        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        deviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerAHat, d_AHat, batchsize, maxn1 * maxn2);

        numBlocks = (batchsize * maxn1 * maxn2 * A->m + BLOCKSIZE - 1) / BLOCKSIZE;
        computeAHat<<<numBlocks, BLOCKSIZE>>>(d_A, d_PointerAHat, d_PointerI, d_PointerJ, d_n1, d_n2, maxn1, maxn2, A->m, batchsize);
        
        float* h_AHat = (float*) malloc(batchsize * maxn1 * maxn2 * sizeof(float));
        gpuAssert(
            cudaMemcpy(h_AHat, d_AHat, batchsize * maxn1 * maxn2 * sizeof(float), cudaMemcpyDeviceToHost));

        printf("--printing h_AHat--\n");
        for (int b = 0; b < batchsize; b++) {
            printf("b: %d\n", b);
            for (int j = 0; j < maxn1; j++) {
                for (int k = 0; k < maxn2; k++) {
                    printf("%f ", h_AHat[b * maxn1 * maxn2 + j * maxn2 + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("maxn1: %d\n", maxn1);
        printf("maxn2: %d\n", maxn2);
        // initialize d_Q and d_R
        float* d_Q;
        float* d_R;
        float** d_PointerQ;
        float** d_PointerR;

        gpuAssert(
            cudaMalloc((void**) &d_Q, batchsize * maxn1 * maxn1 * sizeof(float)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerQ, batchsize * sizeof(float*)));
        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        deviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerQ, d_Q, batchsize, maxn1 * maxn1);


        gpuAssert(
            cudaMalloc((void**) &d_R, batchsize * maxn1 * maxn2 * sizeof(float)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerR, batchsize * sizeof(float*)));
        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        deviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerR, d_R, batchsize, maxn1 * maxn2);

        qrBatched(cHandle, d_PointerAHat, d_PointerQ, d_PointerR, batchsize, maxn1, maxn2);

        // overwrite AHat, since qr overwrote it previously
        numBlocks = (batchsize * maxn1 * maxn2 * A->m + BLOCKSIZE - 1) / BLOCKSIZE;
        computeAHat<<<numBlocks, BLOCKSIZE>>>(d_A, d_PointerAHat, d_PointerI, d_PointerJ, d_n1, d_n2, maxn1, maxn2, A->m, batchsize);


        // initialize mHat_k, residual, residualNorm
        float* d_mHat_k;
        float** d_PointerMHat_k;

        float* d_residual;
        float** d_PointerResidual;

        float* d_residualNorm;

        gpuAssert(
            cudaMalloc((void**) &d_mHat_k, batchsize * maxn2 * sizeof(float)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerMHat_k, batchsize * sizeof(float*)));
        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        deviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerMHat_k, d_mHat_k, batchsize, maxn2);

        gpuAssert(
            cudaMalloc((void**) &d_residual, batchsize * A->m * sizeof(float)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerResidual, batchsize * sizeof(float*)));
        deviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerResidual, d_residual, batchsize, A->m);

        gpuAssert(
            cudaMalloc((void**) &d_residualNorm, batchsize * sizeof(float)));

        LSProblem(cHandle, d_A, A, d_PointerQ, d_PointerR, d_PointerMHat_k, d_PointerResidual, d_PointerI, d_PointerJ, d_n1, d_n2, maxn1, maxn2, i, d_residualNorm, batchsize);
        
        // check if the tolerance is met
        int toleranceNotMet = 0;
        float* h_residualNorm = (float*) malloc(batchsize * sizeof(float));
        gpuAssert(
            cudaMemcpy(h_residualNorm, d_residualNorm, batchsize * sizeof(float), cudaMemcpyDeviceToHost));
        
        printf("--printing h_residualNorm--\n");
        for (int b = 0; b < batchsize; b++) {
            printf("%f ", h_residualNorm[b]);
        }
        
        for (int b = 0; b < batchsize; b++) {
            if (h_residualNorm[b] > tolerance) {
                toleranceNotMet = 1;
            }
        }
        printf("toleranceNotMet: %d\n", toleranceNotMet);

        // while the tolerance is not met, continue the loop
        while (toleranceNotMet && maxIterations > iteration) {
            printf("\n-------Iteration: %d-------\n", iteration);
            iteration++;

            // compute the length of L and set L
            int* d_l;
            int** d_PointerL;

            gpuAssert(
                cudaMalloc((void**) &d_l, batchsize * sizeof(int)));
            gpuAssert(
                cudaMalloc((void**) &d_PointerL, batchsize * sizeof(int*)));

            numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            computeLengthOfL<<< numBlocks, BLOCKSIZE >>>(d_l, d_PointerResidual, d_PointerI, d_PointerL, A->m, d_n1, i, batchsize);

            // check what indeces to keep
            int* d_KeepArray;
            int** d_PointerKeepArray;

            gpuAssert(
                cudaMalloc((void**) &d_KeepArray, batchsize * A->n * sizeof(int)));
            gpuAssert(
                cudaMalloc((void**) &d_PointerKeepArray, batchsize * sizeof(int*)));
            
            numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            iDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerKeepArray, d_KeepArray, batchsize, A->n);

            numBlocks = (batchsize * A->n + BLOCKSIZE - 1) / BLOCKSIZE;
            computeKeepArray<<<numBlocks, BLOCKSIZE>>>(d_A, d_PointerKeepArray, d_PointerL, d_PointerJ, d_n2, d_l, batchsize);

            // int* d_n2Tilde;
            // gpuAssert(
            //     cudaMalloc((void**) &d_n2Tilde, batchsize * sizeof(int)));
            
            // numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            // computeN2Tilde<<<numBlocks, BLOCKSIZE>>>(d_PointerKeepArray, d_n2Tilde, A->n, batchsize);

            // // fill JTilde
            // int** d_PointerJTilde;

            // gpuAssert(
            //     cudaMalloc((void**) &d_PointerJTilde, batchsize * sizeof(int*)));
            
            // numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            // computeJTilde<<<numBlocks, BLOCKSIZE>>>(d_PointerKeepArray, d_PointerJTilde, d_n2Tilde, A->n, batchsize);


            int* h_l = (int*) malloc(batchsize * sizeof(int));
            gpuAssert(
                cudaMemcpy(h_l, d_l, batchsize * sizeof(int), cudaMemcpyDeviceToHost));
            printf("--printing h_l--\n");
            for (int b = 0; b < batchsize; b++) {
                printf("b: %d, l: %d\n", b, h_l[b]);
            }

            int* h_Jtilde = (int*) malloc(batchsize * A->n * sizeof(int));
        }

        float* h_Q = (float*) malloc(batchsize * maxn1 * maxn1 * sizeof(float));
        float* h_R = (float*) malloc(batchsize * maxn1 * maxn2 * sizeof(float));
        gpuAssert(
            cudaMemcpy(h_Q, d_Q, batchsize * maxn1 * maxn1 * sizeof(float), cudaMemcpyDeviceToHost));
        gpuAssert(
            cudaMemcpy(h_R, d_R, batchsize * maxn1 * maxn2 * sizeof(float), cudaMemcpyDeviceToHost));
        
        printf("--printing h_Q--\n");
        for (int b = 0; b < batchsize; b++) {
            printf("b: %d\n", b);
            for (int j = 0; j < maxn1; j++) {
                for (int k = 0; k < maxn1; k++) {
                    printf("%f ", h_Q[b * maxn1 * maxn1 + j * maxn1 + k]);
                }
                printf("\n");
            }
            printf("\n");
        }

        printf("--printing h_R--\n");
        for (int b = 0; b < batchsize; b++) {
            printf("b: %d\n", b);
            for (int j = 0; j < maxn1; j++) {
                for (int k = 0; k < maxn2; k++) {
                    printf("%f ", h_R[b * maxn1 * maxn2 + j * maxn2 + k]);
                }
                printf("\n");
            }
            printf("\n");
        }

        float* h_mHat_k = (float*) malloc(batchsize * maxn2 * sizeof(float));
        gpuAssert(
            cudaMemcpy(h_mHat_k, d_mHat_k, batchsize * maxn2 * sizeof(float), cudaMemcpyDeviceToHost));

        printf("--printing h_mHat_k--\n");
        for (int b = 0; b < batchsize; b++) {
            printf("b: %d\n", b);
            for (int j = 0; j < maxn2; j++) {
                printf("%f ", h_mHat_k[b * maxn2 + j]);
            }
            printf("\n");
        }

        float* h_residual = (float*) malloc(batchsize * A->m * sizeof(float));
        gpuAssert(
            cudaMemcpy(h_residual, d_residual, batchsize * A->m * sizeof(float), cudaMemcpyDeviceToHost));
        
        printf("--printing h_residual--\n");
        for (int b = 0; b < batchsize; b++) {
            printf("b: %d\n", b);
            for (int j = 0; j < A->m; j++) {
                printf("%f ", h_residual[b * A->m + j]);
            }
            printf("\n");
        }


        // free memory
        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        freeIJ<<<numBlocks, BLOCKSIZE>>>(d_PointerI, d_PointerJ, batchsize);
        gpuAssert(
            cudaFree(d_PointerI));
        gpuAssert(
            cudaFree(d_PointerJ));
        gpuAssert(
            cudaFree(d_n1));
        gpuAssert(
            cudaFree(d_n2));
        gpuAssert(
            cudaFree(d_AHat));
        gpuAssert(
            cudaFree(d_PointerAHat));
        gpuAssert(
            cudaFree(d_Q));
        gpuAssert(
            cudaFree(d_PointerQ));
        gpuAssert(
            cudaFree(d_R));
        gpuAssert(
            cudaFree(d_PointerR));
        
    }

    // free memory
    freeDeviceCSC(d_A);
    freeDeviceCSC(d_M);
    freeCSC(M);

    cublasDestroy(cHandle);
}

#endif