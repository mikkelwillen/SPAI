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
// cHandle is the cublas handle
// A is an array of batch matrices
// n is the max number of rows and columns of the matrices
// AInv is an array of batch inverse matrices
// returns 0 if succesful, 1 if not
int invBatched(cublasHandle_t cHandle, float** A, int n, float** AInv) {
    // Set constants
    cublasStatus_t stat;
    int lda = n;
    int ldc = n;
    const size_t AMemSize = n * n * BATCHSIZE * sizeof(float);
    const size_t APointerMemSize = BATCHSIZE * sizeof(float*);

    // Create input and output arrays
    float* d_A;
    float* d_AInv;
    int* d_PivotArray;
    float** d_PointerA;
    float** d_PointerAInv;
    int* h_info = (int*) malloc(BATCHSIZE * sizeof(int));
    int* d_info;

    // Malloc space and copy data for A
    gpuAssert(
        cudaMalloc((void**) &d_A, AMemSize));
    gpuAssert(
        cudaMemcpy(d_A, (*A), AMemSize, cudaMemcpyHostToDevice));
    gpuAssert(
        cudaMalloc((void**) &d_PointerA, APointerMemSize));
    deviceToDevicePointerKernel <<< 1, BATCHSIZE >>> (d_PointerA, d_A, BATCHSIZE, n);

    // Malloc space for AInv
    gpuAssert(
        cudaMalloc((void**) &d_AInv, AMemSize));
    gpuAssert(
        cudaMalloc((void**) &d_PointerAInv, APointerMemSize));
    deviceToDevicePointerKernel <<< 1, BATCHSIZE >>> (d_PointerAInv, d_AInv, BATCHSIZE, n);

    // Malloc space for pivot array
    gpuAssert(
        cudaMalloc((void**) &d_PivotArray, n * BATCHSIZE * sizeof(float)));

    // Malloc space for info
    gpuAssert(
        cudaMalloc((void**) &d_info, BATCHSIZE * sizeof(int)));

    // Run batched LU factorization from cublas
    // Cublas docs: https://docs.nvidia.com/cuda/cublas/
    stat = cublasSgetrfBatched(cHandle,
                               n,
                               d_PointerA,
                               lda,
                               d_PivotArray,
                               d_info,
                               BATCHSIZE);

    // Error handling
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("\ncublasSgetrfBatched failed");
        printf("\ncublas error: %d\n", stat);
        
        return stat;
    }

    gpuAssert(
        cudaMemcpy(h_info, d_info, BATCHSIZE * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < BATCHSIZE; i++) {
        if (h_info[i] != 0) {
            printf("\nError in LU: Matrix %d is singular\n", i);

            return 1;
        }
    }

    // Run batched inversion from cublas
    // Cublas docs: https://docs.nvidia.com/cuda/cublas/
    stat = cublasSgetriBatched(cHandle,
                               n,
                               d_PointerA,
                               lda,
                               d_PivotArray,
                               d_PointerAInv,
                               ldc,
                               d_info,
                               BATCHSIZE);
    
    // Error handling
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("\ncublasSgetriBatched failed");
        printf("\ncublas error: %d\n", stat);
        
        return stat;
    }

    gpuAssert(
        cudaMemcpy(h_info, d_info, BATCHSIZE * sizeof(int), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < BATCHSIZE; i++) {
        if (h_info[i] != 0) {
            printf("\nError in INV: Matrix %d is singular\n", i);

            return 1;
        }
    }

    // Copy result back to host
    gpuAssert(
        cudaMemcpy((*AInv), d_AInv, AMemSize, cudaMemcpyDeviceToHost));

    // Free memory
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
    gpuAssert(
        cudaFree(d_PivotArray));
    free(h_info);

    return 0;
}

#endif