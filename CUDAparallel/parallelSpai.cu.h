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

// kernel for computing I, J n1 and n2
// d_A = device pointer to A
// d_M = device pointer to M
// d_I = device pointer pointer to I
// d_J = device pointer pointer to J
// d_n1 = device pointer to n1
// d_n2 = device pointer to n2
// batchnumber = the current batchnumber
// batchsize = the size of the batch
__global__ void computeIandJ(CSC* d_A, CSC* d_M, int** d_I, int** d_J, int* d_n1, int* d_n2, int batchnumber, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize) {
        int index = batchnumber * batchsize + tid;
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
        d_I[tid] = &I[0];
        d_J[tid] = &J[0];
        d_n1[tid] = n1;
        d_n2[tid] = n2;
    }
}

// kernel for computing Ahat
__global__ void computeAHat(CSC* d_A, float** d_AHat, int** d_I, int** d_J, int* d_n1, int* d_n2, int maxn1, int maxn2, int cscOffset, int batchnumber, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize * maxn1 * maxn2 * cscOffset) {
        int b = (tid / (maxn1 * maxn2 * cscOffset));
        int i = (tid % (maxn1 * maxn2 * cscOffset)) / (maxn2 * cscOffset);
        int j = ((tid % (maxn1 * maxn2 * cscOffset)) % (maxn2 * cscOffset)) / cscOffset;
        int l = ((tid % (maxn1 * maxn2 * cscOffset)) % (maxn2 * cscOffset)) % cscOffset;

        int n1 = d_n1[b];
        int n2 = d_n2[b];

        int* I = d_I[b];
        int* J = d_J[b];
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

__global__ void deviceToDevicePointerKernel(float** d_PointerAHat, float* d_AHat, int batchsize, int maxn1, int maxn2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize) {
        d_PointerAHat[tid] = &d_AHat[tid * maxn1 * maxn2];
    }
}

// A = matrix we want to compute SPAI on
// m, n = size of array
// tolerance = tolerance
// maxIteration = constraint for the maximal number of iterations
// s = number of rho_j - the most profitable indices
// batchsize = number of matrices to be processed in parallel
CSC* parallelSpai(CSC* A, float tolerance, int maxIterations, int s, int batchsize) {
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

    // initialize M and set to diagonal
    CSC* M = createDiagonalCSC(A->m, A->n);
    printf("after m\n");
    // copy A to device
    // de her skal lige fixes
    CSC* d_A = copyCSCFromHostToDevice(A);
    printf("after d_A\n");
    CSC* d_M = copyCSCFromHostToDevice(M);
    printf("after d_M\n");
    
    // compute the batchnumber 
    int batchnumber = (A->n + batchsize - 1) / batchsize;

    for (int i = 0; i < batchnumber; i++) {
        int** d_I;
        int** d_J;
        int* d_n1;
        int* d_n2;

        // malloc space
        gpuAssert(
            cudaMalloc((void**) &d_I, batchsize * sizeof(int*)));
        gpuAssert(
            cudaMalloc((void**) &d_J, batchsize * sizeof(int*)));
        gpuAssert(
            cudaMalloc((void**) &d_n1, batchsize * sizeof(int)));
        gpuAssert(
            cudaMalloc((void**) &d_n2, batchsize * sizeof(int)));
        
        computeIandJ<<<1, batchsize>>>(d_A, d_M, d_I, d_J, d_n1, d_n2, i, batchsize);

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

        deviceToDevicePointerKernel<<<1, batchsize>>>(d_PointerAHat, d_AHat, batchsize, maxn1, maxn2);

        computeAHat<<<1, batchsize * maxn1 * maxn2 * A->n>>>(d_A, d_PointerAHat, d_I, d_J, d_n1, d_n2, maxn1, maxn2, A->n, i, batchsize);
        
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

        // initialize d_Q and d_R
        float** d_Q;
        float** d_R;

        gpuAssert(
            cudaMalloc((void**) &d_Q, batchsize * sizeof(float)));
        gpuAssert(
            cudaMalloc((void**) &d_R, batchsize * sizeof(float)));
        


        // print n2
        printf("n2: ");
        for (int j = 0; j < batchsize; j++) {
            printf("%d ", n2[j]);
        }
        printf("\n");

    }
}

#endif