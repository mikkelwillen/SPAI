#ifndef INV_BATCHED_H
#define INV_BATCHED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"

// Kernel to copy d_A to d_PointerA
// d_A is an array of batch matrices
// d_PointerA is an array of pointers to the start of each matrix in d_A
// batch is the BATCHSIZE
// n is the number of rows and columns in each matrix
__global__ void deviceToDevicePointerKernel(float** d_PointerA, float* d_A, int batch, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < BATCHSIZE) {
        d_PointerA[tid] = &d_A[tid * n * n];
    }
}

// Function to do inversion of batch matrices
// A is an array of batch matrices
// n is the max number of rows and columns of the matrices
// AInv is an array of batch inverse matrices
int invBatched(cublasHandle_t cHandle, cublasStatus_t stat, float* A, int n, float* AInv) {
    printf("\nDo inversion of A\n");

    // Set constants
    int lda = n;
    int ldc = n;
    const size_t AMemSize = n * n * BATCHSIZE * sizeof(float);
    const size_t APointerMemSize = BATCHSIZE * sizeof(float*);

    // create input and output arrays
    float* d_A;
    float* d_AInv;
    float** d_PointerA;
    float** d_PointerAInv;
    int* h_info = (int*) malloc(BATCHSIZE * sizeof(int));
    int* d_info;

    // malloc space and copy data for A
    gpuAssert(
        cudaMalloc((void**) &d_A, AMemSize));
    gpuAssert(
        cudaMemcpy(d_A, A, AMemSize, cudaMemcpyHostToDevice));
    gpuAssert(
        cudaMalloc((void**) &d_PointerA, APointerMemSize));
    deviceToDevicePointerKernel <<< 1, BATCHSIZE >>> (d_PointerA, d_A, BATCHSIZE, n);

    // malloc space for AInv
    gpuAssert(
        cudaMalloc((void**) &d_AInv, AMemSize));
    gpuAssert(
        cudaMalloc((void**) &d_PointerAInv, APointerMemSize));
    deviceToDevicePointerKernel <<< 1, BATCHSIZE >>> (d_PointerAInv, d_AInv, BATCHSIZE, n);

    // malloc space for info
    gpuAssert(
        cudaMalloc((void**) &d_info, BATCHSIZE * sizeof(int)));

    // run batched inversion from cublas
    // cublas docs: https://docs.nvidia.com/cuda/cublas/
    stat = cublasSgetriBatched(cHandle,
                               n,
                               d_PointerA,
                               lda,
                               NULL,
                               d_PointerAInv,
                               ldc,
                               d_info,
                               BATCHSIZE);
    
    // error handling
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("\ncublasSgetriBatched failed");
        printf("\ncublas error: %d\n", stat);
    }

    gpuAssert(
        cudaMemcpy(h_info, d_info, BATCHSIZE * sizeof(int), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < BATCHSIZE; i++) {
        if (h_info[i] != 0) {
            printf("\nError: Matrix %d is singular\n", i);
        }
    }

    // copy result back to host
    gpuAssert(
        cudaMemcpy(AInv, d_AInv, AMemSize, cudaMemcpyDeviceToHost));

    // free memory
    gpuAssert(
        cudaFree(d_A));
    gpuAssert(
        cudaFree(d_AInv));
    gpuAssert(
        cudaFree(d_PointerA));
    gpuAssert(
        cudaFree(d_PointerAInv));
    gpuAssert(
        cudaFree(d_info));
    free(h_info);

    // destroy cublas handle
    cublasDestroy(cHandle);

    return 0;
}

#endif