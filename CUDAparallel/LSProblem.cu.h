#ifndef LS_PROBLEM_H
#define LS_PROBLEM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "csc.cu.h"
#include "constants.cu.h"
#include "parallelSpai.cu.h"
#include "invBatched.cu.h"
#include "helperKernels.cu.h"


// Function for setting the cHat vector, which is the k'th vector of the Q matrix
// d_PointerCHat = the pointer to the cHat vector
// d_PointerQ    = the pointer to the Q matrix
// d_PointerI    = the pointer to the I vector
// d_n1          = the number of rows in A
// currentBatch  = the current batch
// batchsize     = the batchsize for the cublas handle
// maxn1         = the maximum number of rows in A
__global__ void setCHat(float** d_PointerCHat, float** d_PointerQ, int** d_PointerI, int* d_n1, int currentBatch, int batchsize, int maxn1) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < maxn1 * maxn1 * batchsize) {
        int b = tid / (maxn1 * maxn1);
        int i = (tid % (maxn1 * maxn1)) / maxn1;
        int j = (tid % (maxn1 * maxn1)) % maxn1;
        int k = currentBatch * batchsize + b;

        float* d_cHat = d_PointerCHat[b];
        float* d_Q = d_PointerQ[b];
        int* d_I = d_PointerI[b];

        if (j == 0) {
            d_cHat[i] = 0.0;
        }
        __syncthreads();

        if (i < d_n1[b]) {
            if (k == d_I[i]) {
                d_cHat[j] = d_Q[i * d_n1[b] + j];
            }
        }
    }
}

// Function for computing the least squares problem
// cHandle = the cublas handle
// A = the sparse matrix
// d_PointerQ = the pointer to the Q matrix
// d_PointerR = the pointer to the R matrix
// d_mHat_k = the pointer to the mHat_k vector
// d_PointerResidual = the pointer to the residual vector
// d_PointerI = the pointer to the I vector
// d_PointerJ = the pointer to the J vector
// d_n1 = the number of rows in A
// d_n2 = the number of columns in A
// k = the index of the column to be added
// residualNorm = the norm of the residual
// batchsize = the batchsize for the cublas handle
int LSProblem(cublasHandle_t cHandle, CSC* A, float** d_PointerQ, float** d_PointerR, float** d_mHat_k, float** d_PointerResidual, int** d_PointerI, int** d_PointerJ, int* d_n1, int* d_n2, int maxn1, int maxn2,int currentBatch, float* residualNorm, int batchsize) {
    int numBlocks;
    float* d_cHat;
    float** d_PointerCHat;

    gpuAssert(
        cudaMalloc((void**) &d_cHat, maxn1 * batchsize * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerCHat, batchsize * sizeof(float*)));
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    deviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerCHat, d_cHat, batchsize, maxn1);
    
    numBlocks = (maxn1 * maxn1 * batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    setCHat<<<numBlocks, BLOCKSIZE>>>(d_PointerCHat, d_PointerQ, d_PointerI, d_n1, currentBatch, batchsize, maxn1);

    float* h_cHat = (float*) malloc(maxn1 * batchsize * sizeof(float));
    gpuAssert(
        cudaMemcpy(h_cHat, d_cHat, maxn1 * batchsize * sizeof(float), cudaMemcpyDeviceToHost));
    printf("--printing cHat--\n");
    for (int b = 0; b < batchsize; b++) {
        printf("batch %d\n", b);
        for (int i = 0; i < maxn1; i++) {
            printf("%f ", h_cHat[b * maxn1 + i]);
        }
        printf("\n");
    }

    // float* d_invR;
    // float** d_PointerInvR;

    // gpuAssert(
    //     cudaMalloc((void**) &d_invR, maxn2 * maxn2 * batchsize * sizeof(float)));
    // gpuAssert(
    //     cudaMalloc((void***) &d_PointerInvR, batchsize * sizeof(float*)));
    // numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    // deviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerInvR, d_invR, batchsize, maxn2 * maxn2);
    
    // invBatched(cHandle, d_PointerR, d_PointerInvR, maxn2, batchsize);


    return 0;
}

#endif