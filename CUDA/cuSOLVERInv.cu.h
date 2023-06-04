#ifndef CUSOLVER_INV_H
#define CUSOLVER_INV_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "cusolverDn.h"
#include "csc.cu.h"
#include "constants.cu.h"


__global__ void getLAndU(float* A, float* L, float* U, int m, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < m * n) {
        int i = tid / m;
        int j = tid % m;

        if (i == j) {
            L[i * n + j] = 1.0;
            U[i * n + j] = A[i * n + j];
        } else if (i < j) {
            L[i * n + j] = 0.0;
            U[i * n + j] = A[i * n + j];
        } else {
            L[i * n + j] = A[i * n + j];
            U[i * n + j] = 0.0;
        }
    }
}

__global__ void matrixMultiplication (float* d_A, float* d_B, float* d_C, int d_dim1, int d_dim2, int d_dim3) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < d_dim1 * d_dim3) {
        int i = tid / d_dim3;
        int j = tid % d_dim3;

        if (i < d_dim1 && j < d_dim3) {
            float sum = 0.0;
            for (int k = 0; k < d_dim2; k++) {
                sum += d_A[i * d_dim2 + k] * d_B[k * d_dim3 + j];
            }
            d_C[i * d_dim3 + j] = sum;
        }
    }
}

float* cuSOLVERInversion(float* A, int m, int n) {
    // Create cusolver handle
    cusolverDnHandle_t handle;
    cusolverStatus_t stat;
    stat = cusolverDnCreate(&handle);
    if (stat != CUSOLVER_STATUS_SUCCESS) {
        printf("cuSolver initialization failed\n");
        printf("cuSolver error: %d\n", stat);

        return NULL;
    } 

    // Variables
    float* d_A;
    int lda = MAX(1, m);
    int Lwork;
    float* workspace;
    int* devIpiv;
    int devSize = MIN(m, n);
    int* d_info;

    gpuAssert(
        cudaMalloc((void**) &d_A, m * n * sizeof(float)));

    gpuAssert(
        cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // find the work buffer size
    stat = cusolverDnSgetrf_bufferSize(handle,
                                       m, 
                                       n, 
                                       d_A, 
                                       lda,
                                       &Lwork);
    
    // error handling
    if (stat != CUSOLVER_STATUS_SUCCESS) {
        printf("CUSOLVER get buffer size failed\n");

        return NULL;
    }

    // allocate the work buffer
    gpuAssert(
        cudaMalloc((void**) &workspace, Lwork * sizeof(float)));

    // allocate the pivot array
    gpuAssert(
        cudaMalloc((void**) &devIpiv, devSize * sizeof(int)));

    // allocate the info array
    gpuAssert(
        cudaMalloc((void**) &d_info, sizeof(int)));

    // perform LU factorization from cusolver
    // https://docs.nvidia.com/cuda/cusolver
    stat = cusolverDnSgetrf(handle,
                            m,
                            n,
                            d_A,
                            lda,
                            workspace,
                            devIpiv,
                            d_info);

    // error handling
    int h_info;
    gpuAssert(
        cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        
    if (h_info < 0) {
        printf("The %d'th paramter is wrong\n", -h_info);

    }

    if (stat != CUSOLVER_STATUS_SUCCESS) {
        printf("CUSOLVER LU factorization failed with status: %d\n", stat);

        return NULL;
    } 

    // Set L and U
    float* d_L;
    float* d_U;

    gpuAssert(
        cudaMalloc((void**) &d_L, m * n * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_U, m * n * sizeof(float)));
    
    int numBlocks = (m * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    getLAndU<<<numBlocks, BLOCK_SIZE>>>(d_A, d_L, d_U, m, n);

    // float* h_L = (float*) malloc(m * n * sizeof(float));
    // float* h_U = (float*) malloc(m * n * sizeof(float));
    // gpuAssert(
    //     cudaMemcpy(h_L, d_L, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    // gpuAssert(
    //     cudaMemcpy(h_U, d_U, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // // print h_L
    // printf("L:\n");
    // for (int i = 0; i < m; i++) {
    //     printf("[");
    //     for (int j = 0; j < n; j++) {
    //         printf("%f ", h_L[i * n + j]);
    //     }
    //     printf("]\n");
    // }

    // // print h_U
    // printf("U:\n");
    // for (int i = 0; i < m; i++) {
    //     printf("[");
    //     for (int j = 0; j < n; j++) {
    //         printf("%f ", h_U[i * n + j]);
    //     }
    //     printf("]\n");
    // }


    // Variables
    size_t workspaceInBytesOnDevice;
    size_t workspaceInBytesOnHost;

    // Do inversion of L and U

    // Compute buffersize of L
    stat = cusolverDnXtrtri_bufferSize(handle,
                                       CUBLAS_FILL_MODE_LOWER,
                                       CUBLAS_DIAG_NON_UNIT,
                                       n,
                                       CUDA_R_32F,
                                       d_L,
                                       lda,
                                       &workspaceInBytesOnDevice,
                                       &workspaceInBytesOnHost);

    void* bufferOnDevice;
    gpuAssert(
        cudaMalloc((void**) &bufferOnDevice, workspaceInBytesOnDevice));
    void* bufferOnHost;
    gpuAssert(
        cudaMallocHost((void**) &bufferOnHost, workspaceInBytesOnHost));

    // Do inversion of L
    stat = cusolverDnXtrtri(handle,
                            CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_DIAG_NON_UNIT,
                            n,
                            CUDA_R_32F,
                            d_L,
                            lda,
                            bufferOnDevice,
                            workspaceInBytesOnDevice,
                            bufferOnHost,
                            workspaceInBytesOnHost,
                            d_info);

    // error handling
    gpuAssert(
        cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (h_info < 0) {
        printf("The %d'th paramter is wrong\n", -h_info);

        return NULL;
    }

    if (stat != CUSOLVER_STATUS_SUCCESS) {
        printf("CUSOLVER inversion of L failed with status: %d\n", stat);

        return NULL;
    }

    // Compute buffersize of U
    stat = cusolverDnXtrtri_bufferSize(handle,
                                       CUBLAS_FILL_MODE_UPPER,
                                       CUBLAS_DIAG_NON_UNIT,
                                       n,
                                       CUDA_R_32F,
                                       d_U,
                                       lda,
                                       &workspaceInBytesOnDevice,
                                       &workspaceInBytesOnHost);
    
    gpuAssert(
        cudaFree(bufferOnDevice));
    gpuAssert(
        cudaFree(bufferOnHost));

    gpuAssert(
        cudaMalloc((void**) &bufferOnDevice, workspaceInBytesOnDevice));
    gpuAssert(
        cudaMallocHost((void**) &bufferOnHost, workspaceInBytesOnHost));

    // Do inversion of U
    stat = cusolverDnXtrtri(handle,
                            CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_DIAG_NON_UNIT,
                            n,
                            CUDA_R_32F,
                            d_U,
                            lda,
                            bufferOnDevice,
                            workspaceInBytesOnDevice,
                            bufferOnHost,
                            workspaceInBytesOnHost,
                            d_info);

    // error handling
    gpuAssert(
        cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (h_info < 0) {
        printf("The %d'th paramter is wrong\n", -h_info);

        return NULL;
    }

    if (stat != CUSOLVER_STATUS_SUCCESS) {
        printf("CUSOLVER inversion of U failed with status: %d\n", stat);

        return NULL;
    }

    float* AInv;
    gpuAssert(
        cudaMalloc((void**) &AInv, m * n * sizeof(float)));

    // Compute AInv = U^-1 * L^-1
    numBlocks = (m * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matrixMultiplication<<<numBlocks, BLOCK_SIZE>>>(d_L, d_U, AInv, m, n, n);

    // Destroy handles
    cusolverDnDestroy(handle);
    if (stat != CUSOLVER_STATUS_SUCCESS) {
        printf("CUSOLVER handle destruction failed with status: %d\n", stat);

        return NULL;
    }

    float* h_AInv = (float*) malloc(m * n * sizeof(float));
    gpuAssert(
        cudaMemcpy(h_AInv, AInv, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // free memory
    gpuAssert(
        cudaFree(d_A));
    gpuAssert(
        cudaFree(d_L));
    gpuAssert(
        cudaFree(d_U));
    gpuAssert(
        cudaFree(AInv));
    gpuAssert(
        cudaFree(bufferOnDevice));
    gpuAssert(
        cudaFree(bufferOnHost));
    gpuAssert(
        cudaFree(d_info));
    gpuAssert(
        cudaFree(devIpiv));
    gpuAssert(
        cudaFree(workspace));

    
    // return AInv
    return h_AInv;

}

#endif