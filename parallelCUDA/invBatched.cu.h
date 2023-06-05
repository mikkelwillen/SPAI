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
#include "SPAIkernels.cu.h"


// Function to do inversion of batch matrices
// A is an array of batch matrices
// n is the max number of rows and columns of the matrices
// AInv is an array of batch inverse matrices
int invBatched(cublasHandle_t cHandle, float* d_R, float** d_PointerR, float** d_PointerInvR, int maxn2, int batchsize) {
    printf("\nDo inversion of R\n");

    // Set constants
    cublasStatus_t stat;
    int lda = MAX(1, maxn2);
    int ldc = MAX(1, maxn2);

    // create device info array
    int* h_info = (int*) malloc(batchsize * sizeof(int));
    int* d_info;

    float* h_R = (float*) malloc(20 * maxn2 * batchsize * sizeof(float));

    // copy R to host
    gpuAssert(
        cudaMemcpy(h_R, d_R, 20 * maxn2 * batchsize * sizeof(float), cudaMemcpyDeviceToHost));
    
    // print R
    printf("\nR:\n");
    for (int i = 0; i < batchsize; i++) {
        printf("\nMatrix %d:\n", i);
        for (int j = 0; j < 20; j++) {
            for (int k = 0; k < maxn2; k++) {
                printf("%f ", h_R[i * 20 * maxn2 + j * 20 + k]);
            }
            printf("\n");
        }
    }

    printf("maxn2 in invBatched: %d\n", maxn2);
    printf("batchsize in invBatched: %d\n", batchsize);

    // malloc space for info
    gpuAssert(
        cudaMalloc((void**) &d_info, batchsize * sizeof(int)));
    printf("malloc info\n");

    // run batched LU factorization from cublas
    // cublas docs: https://docs.nvidia.com/cuda/cublas/
    stat = cublasSgetrfBatched(cHandle,
                               maxn2,
                               d_PointerR,
                               lda,
                               NULL,
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
        cudaMemcpy(h_info, d_info, batchsize * sizeof(int), cudaMemcpyDeviceToHost));
    printf("copy info back to host\n");
    
    // for (int i = 0; i < BATCHSIZE; i++) {
    //     if (h_info[i] != 0) {
    //         printf("\nError in LU: Matrix %d is singular\n", i);
    //     }
    // }

    // run batched inversion from cublas
    // cublas docs: https://docs.nvidia.com/cuda/cublas/
    stat = cublasSgetriBatched(cHandle,
                               maxn2,
                               d_PointerR,
                               lda,
                               NULL,
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
        cudaMemcpy(h_info, d_info, batchsize * sizeof(int), cudaMemcpyDeviceToHost));
    
    // // check for singular matrices
    // for (int i = 0; i < BATCHSIZE; i++) {
    //     if (h_info[i] != 0) {
    //         printf("\nError in INV: Matrix %d is singular\n", i);
    //     }
    // }

    // free memory
    gpuAssert(
        cudaFree(d_info));
    free(h_info);

    return 0;
}

#endif