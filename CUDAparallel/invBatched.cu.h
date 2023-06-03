#ifndef INV_BATCHED_H
#define INV_BATCHED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"
#include "helperKernels.cu.h"


// Function to do inversion of batch matrices
// A is an array of batch matrices
// n is the max number of rows and columns of the matrices
// AInv is an array of batch inverse matrices
int invBatched(cublasHandle_t cHandle, float** d_PointerR, float** d_PointerInvR, int maxn2, int batchsize) {
    printf("\nDo inversion of R\n");

    // Set constants
    cublasStatus_t stat;
    int lda = maxn2;
    int ldc = maxn2;

    // create device info array
    int* h_info = (int*) malloc(batchsize * sizeof(int));
    int* d_info;
    int* d_PivotArray;

    // malloc space for info
    gpuAssert(
        cudaMalloc((void**) &d_info, batchsize * sizeof(int)));
    printf("malloc info\n");
    // malloc space for pivot array
    gpuAssert(
        cudaMalloc((void**) &d_PivotArray, maxn2 * batchsize * sizeof(int)));
    // printf("malloc pivot array\n");

    // run batched LU factorization from cublas
    // cublas docs: https://docs.nvidia.com/cuda/cublas/
    stat = cublasSgetrfBatched(cHandle,
                               maxn2,
                               d_PointerR,
                               lda,
                               d_PivotArray,
                               d_info,
                               batchsize);
    
    // error handling
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("\ncublasSgetrfBatched failed");
        printf("\ncublas error: %d\n", stat);
        
        return stat;
    }

    printf("before info to host\n");
    // copy info back to host
    gpuAssert(
        cudaMemcpy(h_info, d_info, BATCHSIZE * sizeof(int), cudaMemcpyDeviceToHost));
    printf("copy info back to host\n");
    
    for (int i = 0; i < BATCHSIZE; i++) {
        if (h_info[i] != 0) {
            printf("\nError in LU: Matrix %d is singular\n", i);
        }
    }

    int* h_PivotArray = (int*) malloc(maxn2 * batchsize * sizeof(int));
    gpuAssert(
        cudaMemcpy(h_PivotArray, d_PivotArray, maxn2 * batchsize * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < BATCHSIZE; i++) {
        printf("Pivot array for matrix %d:\n", i);
        for (int j = 0; j < maxn2; j++) {
            printf("%d ", h_PivotArray[i * maxn2 + j]);
        }
        printf("\n");
    }

    // run batched inversion from cublas
    // cublas docs: https://docs.nvidia.com/cuda/cublas/
    stat = cublasSgetriBatched(cHandle,
                               maxn2,
                               d_PointerR,
                               lda,
                               d_PivotArray,
                               d_PointerInvR,
                               ldc,
                               d_info,
                               batchsize);
    
    // error handling
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("\ncublasSgetriBatched failed");
        printf("\ncublas error: %d\n", stat);
        
        return stat;
    }

    // copy info back to host
    gpuAssert(
        cudaMemcpy(h_info, d_info, BATCHSIZE * sizeof(int), cudaMemcpyDeviceToHost));
    
    // check for singular matrices
    for (int i = 0; i < BATCHSIZE; i++) {
        if (h_info[i] != 0) {
            printf("\nError in INV: Matrix %d is singular\n", i);
        }
    }

    // free memory
    gpuAssert(
        cudaFree(d_info));
    free(h_info);

    return 0;
}

#endif