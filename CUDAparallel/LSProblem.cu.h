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
#include "SPAIkernels.cu.h"


// Function for setting the cHat vector, which is the k'th vector of the Q matrix
// d_PointerCHat = the pointer to the cHat vector
// d_PointerQ    = the pointer to the Q matrix
// d_PointerI    = the pointer to the I vector
// d_n1          = the number of rows in A
// currentBatch  = the current batch
// batchsize     = the batchsize for the cublas handle
// maxn1         = the maximum number of rows in A
__global__ void setCHat(float** d_PointerCHat, float** d_PointerQ, int** d_PointerI, int* d_n1, int* d_n2, int maxn1, int maxn2, int currentBatch, int batchsize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < maxn1 * maxn2 * batchsize) {
        int b = tid / (maxn1 * maxn2);
        int i = (tid % (maxn1 * maxn2)) / maxn2;
        int j = (tid % (maxn1 * maxn2)) % maxn2;
        int k = currentBatch * batchsize + b;

        float* d_cHat = d_PointerCHat[b];
        float* d_Q = d_PointerQ[b];
        int* d_I = d_PointerI[b];

        if (i == 0) {
            d_cHat[j] = 0.0;
        }
        __syncthreads();

        if (i < d_n1[b]) {
            printf("j in setCHat: %d\n", j);
            if (k == d_I[i]) {
                d_cHat[j] = d_Q[i * maxn1 + j];
            }
        }
    }
}

// function for computing mHat_k
// d_PointerMHat_k = the pointer to the mHat_k vector
// d_PointerInvR   = the pointer to the invR matrix
// d_PointerCHat   = the pointer to the cHat vector
// maxn2           = the maximum number of columns in A
// batchsize       = the batchsize for the cublas handle
__global__ void computeMHat_k(float** d_PointerMHat_k, float** d_PointerInvR, float** d_PointerCHat, int maxn2, int batchsize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < maxn2 * batchsize) {
        int b = tid / maxn2;
        int i = tid % maxn2;

        float* d_mHat_k = d_PointerMHat_k[b];
        float* d_invR = d_PointerInvR[b];
        float* d_cHat = d_PointerCHat[b];

        d_mHat_k[i] = 0.0;
        for (int j = 0; j < maxn2; j++) {
            d_mHat_k[i] += d_invR[i * maxn2 + j] * d_cHat[j];
        }
    }
}

// function for computing the residual
// d_A               = the sparse matrix
// d_PointerResidual = the pointer to the residual vector
// d_PointerMHat_k   = the pointer to the mHat_k vector
// maxn2             = the maximum number of columns in A
// currentBatch      = the current batch
// batchsize         = the batchsize
__global__ void computeResidual(float* d_ADense, float** d_PointerResidual, float** d_PointerMHat_k, int** d_PointerJ, int* d_n2, int m, int n, int currentBatch, int batchsize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < m * n * batchsize) {
        int b = tid / (m * n);
        int i = (tid % (m * n)) / n;
        int j = (tid % (m * n)) % n;
        int k = currentBatch * batchsize + b;

        int n2 = d_n2[b];

        int* d_J = d_PointerJ[b];
        float* d_residual = d_PointerResidual[b];
        float* d_mHat_k = d_PointerMHat_k[b];

        d_residual[i] = 0.0;

        for (int h = 0; h < n2; h++) {
            if (j == d_J[h]) {
                d_residual[i] += d_ADense[i * n + j] * d_mHat_k[h];
                if (b == 0) {
                    printf("%f = %f * %f\n", d_residual[i], d_ADense[i * n + j], d_mHat_k[h]);
                }
            }
        }

        if (i == k && j == 0) {
            d_residual[i] -= 1.0;
        }
    }
}

// function for computing the norm of the residual
// d_PointerResidual = the pointer to the residual vector
// d_residualNorm    = the pointer to the residual norm value
// batchsize         = the batchsize
// m                 = the number of rows in A
__global__ void computeNorm(float** d_PointerResidual, float* d_residualNorm, int batchsize, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize) {

        float* d_residual = d_PointerResidual[tid];

        d_residualNorm[tid] = 0.0;
        for (int i = 0; i < m; i++) {
            d_residualNorm[tid] += d_residual[i] * d_residual[i];
        }

        d_residualNorm[tid] = sqrt(d_residualNorm[tid]);
    }
}

// Function for computing the least squares problem
// cHandle           = the cublas handle
// A                 = the sparse matrix
// d_PointerQ        = the pointer to the Q matrix
// d_PointerR        = the pointer to the R matrix
// d_mHat_k          = the pointer to the mHat_k vector
// d_PointerPc       = the pointer to the column permutation matrix (null if not used)
// d_PointerResidual = the pointer to the residual vector
// d_PointerI        = the pointer to the I vector
// d_PointerJ        = the pointer to the J vector
// d_n1              = the number of rows in A
// d_n2              = the number of columns in A
// k                 = the index of the column to be added
// residualNorm      = the norm of the residual
// batchsize         = the batchsize for the cublas handle
int LSProblem(cublasHandle_t cHandle, CSC* d_A, CSC* A, float* d_ADense, float** d_PointerQ, float** d_PointerR, float** d_PointerMHat_k, float** d_PointerPc, float** d_PointerResidual, int** d_PointerI, int** d_PointerJ, int* d_n1, int* d_n2, int maxn1, int maxn2,int currentBatch, float* d_residualNorm, int batchsize) {
    // define the number of blocks
    int numBlocks;

    // create the cHat vector
    float* d_cHat;
    float** d_PointerCHat;

    // allocate device memory for the cHat vector
    gpuAssert(
        cudaMalloc((void**) &d_cHat, maxn2 * batchsize * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerCHat, batchsize * sizeof(float*)));
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerCHat, d_cHat, batchsize, maxn2);
    
    // set the cHat vector
    numBlocks = (maxn1 * maxn2 * batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    setCHat<<<numBlocks, BLOCKSIZE>>>(d_PointerCHat, d_PointerQ, d_PointerI, d_n1, d_n2, maxn1, maxn2, currentBatch, batchsize);

    // print the cHat vector
    float* h_cHat = (float*) malloc(maxn2 * batchsize * sizeof(float));
    gpuAssert(
        cudaMemcpy(h_cHat, d_cHat, maxn2 * batchsize * sizeof(float), cudaMemcpyDeviceToHost));
    printf("--printing cHat--\n");
    for (int b = 0; b < batchsize; b++) {
        printf("batch %d\n", b);
        for (int i = 0; i < maxn2; i++) {
            printf("%f ", h_cHat[b * maxn2 + i]);
        }
        printf("\n");
    }

    // create the invR matrices
    float* d_invR;
    float** d_PointerInvR;

    // allocate device memory for the invR matrices
    gpuAssert(
        cudaMalloc((void**) &d_invR, maxn2 * maxn2 * batchsize * sizeof(float)));
    gpuAssert(
        cudaMalloc((void***) &d_PointerInvR, batchsize * sizeof(float*)));
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerInvR, d_invR, batchsize, maxn2 * maxn2);
    
    // compute the invR matrices
    invBatched(cHandle, d_PointerR, d_PointerInvR, maxn2, batchsize);


    // print the invR matrices
    float* h_invR = (float*) malloc(maxn2 * maxn2 * batchsize * sizeof(float));
    gpuAssert(
        cudaMemcpy(h_invR, d_invR, maxn2 * maxn2 * batchsize * sizeof(float), cudaMemcpyDeviceToHost));
    printf("--printing invR--\n");
    for (int b = 0; b < batchsize; b++) {
        printf("batch %d\n", b);
        for (int i = 0; i < maxn2; i++) {
            for (int j = 0; j < maxn2; j++) {
                printf("%f ", h_invR[b * maxn2 * maxn2 + i * maxn2 + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // compute the mHat_k vectors
    numBlocks = (maxn2 * batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    computeMHat_k<<<numBlocks, BLOCKSIZE>>>(d_PointerMHat_k, d_PointerInvR, d_PointerCHat, maxn2, batchsize);
    printf("after mHat_k\n");

    // compute residual vectors
    numBlocks = (A->m * A->n * batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    computeResidual<<<numBlocks, BLOCKSIZE>>>(d_ADense, d_PointerResidual, d_PointerMHat_k, d_PointerJ, d_n2, A->m, A->n, currentBatch, batchsize);
    printf("after residual\n");
    
    // permute the mHat_k vectors, if necessary
    if (d_PointerPc != NULL) {
        // permute the vector and save it in a temporary vector
        float* d_tempMHat_k;
        float** d_PointerTempMHat_k;
        gpuAssert(
            cudaMalloc((void**) &d_tempMHat_k, maxn2 * batchsize * sizeof(float)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerTempMHat_k, batchsize * sizeof(float*)));
        
        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerTempMHat_k, d_tempMHat_k, batchsize, maxn2);

        numBlocks = (maxn2 * batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        matrixMultiplication<<<numBlocks, BLOCKSIZE>>>( d_PointerPc, d_PointerMHat_k, d_PointerTempMHat_k, d_n2, d_n2, NULL, maxn2, maxn2, 1, batchsize);

        float* h_tempMHat_k = (float*) malloc(maxn2 * batchsize * sizeof(float));
        gpuAssert(
            cudaMemcpy(h_tempMHat_k, d_tempMHat_k, maxn2 * batchsize * sizeof(float), cudaMemcpyDeviceToHost));
        printf("--printing tempMHat_k--\n");
        for (int b = 0; b < batchsize; b++) {
            printf("batch %d\n", b);
            for (int i = 0; i < maxn2; i++) {
                printf("%f ", h_tempMHat_k[b * maxn2 + i]);
            }
            printf("\n");
        }

        // copy the temporary mHat_k to the mHat_k vector
        gpuAssert(
            cudaMemcpy(d_PointerMHat_k, d_PointerTempMHat_k, batchsize * sizeof(float*), cudaMemcpyDeviceToDevice));
    }

    // compute the norm of the residual
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    computeNorm<<<numBlocks, BLOCKSIZE>>>(d_PointerResidual, d_residualNorm, batchsize, A->m);

    return 0;
}

#endif