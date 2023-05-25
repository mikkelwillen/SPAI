#ifndef DETERMINANT_CU_H
#define DETERMINANT_CU_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "csc.cu.h"
#include "cusolverDn.h"
#include "constants.cu.h"

// Function to check if a matrix is singular
// A = the matrix to check for singularity
// m = the number of rows in A
// n = the number of columns in A
// returns 1 if A is singular, 0 otherwise
int checkSingularity(CSC* cscA) {
    // initialize cusolver
    cusolverDnHandle_t cHandle;
    cusolverStatus_t stat;
    stat = cusolverDnCreate(&cHandle);
    if (stat != CUSOLVER_STATUS_SUCCESS) {
        printf("CUSOLVER initialization failed\n");

        return 1;
    }

    int m = cscA->m;
    int n = cscA->n;

    int* I = (int*) malloc(m * sizeof(int));
    int* J = (int*) malloc(n * sizeof(int));

    for (int i = 0; i < m; i++) {
        I[i] = i;
    }

    for (int j = 0; j < n; j++) {
        J[j] = j;
    }

    // densify A
    float* A = CSCToDense(cscA, I, J, m, n);

    // print A
    printf("A:\n");
    for (int i = 0; i < m; i++) {
        printf("[");
        for (int j = 0; j < n; j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("]\n");
    }
    
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
    stat = cusolverDnSgetrf_bufferSize(cHandle,
                                       m, 
                                       n, 
                                       d_A, 
                                       lda,
                                       &Lwork);
    
    // error handling
    if (stat != CUSOLVER_STATUS_SUCCESS) {
        printf("CUSOLVER get buffer size failed\n");

        return 1;
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
    stat = cusolverDnSgetrf(cHandle,
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

        return 1;
    } 

    float* h_A = (float*) malloc(lda * n * sizeof(float));
    gpuAssert(
        cudaMemcpy(h_A, d_A, lda * n * sizeof(float), cudaMemcpyDeviceToHost));
    printf("LU factorization:\n");
    for (int i = 0; i < m; i++) {
        printf("[");
        for (int j = 0; j < n; j++) {
            printf("%f ", h_A[i * n + j]);
        }
        printf("]\n");
    }

    if (h_info > 0) {
        printf("The input matrix is singular: %d\n", h_info);

        return 1;
    }

    // free memory
    free(A);
    free(I);
    free(J);
    free(h_A);
    gpuAssert(
        cudaFree(d_A));
    gpuAssert(
        cudaFree(workspace));
    gpuAssert(
        cudaFree(devIpiv));
    gpuAssert(
        cudaFree(d_info));

    cusolverDnDestroy(cHandle);

    return 0;
}

#endif