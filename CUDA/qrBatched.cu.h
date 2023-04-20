#ifndef QR_BATCHED_H
#define QR_BATCHED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"

__global__ void printKernel(float** D, int length) {
    printf("inside printKernel\n");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < length) {
        printf("D[%d]: %f \n", tid, D[tid]);
    }
}

int qrBatched(float* AHat, int n1, int n2, float* Q, float* R) {
    printf("\nDo QR decomposition of AHat\n");
    // create cublas handle
    cublasHandle_t cHandle;
    cublasStatus_t stat = cublasCreate(&cHandle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("\ncublasCreate failed");
        printf("\ncublas error: %d\n", stat);
    }
    int lda = n1;
    int min = MIN(n1, n2);
    int ltau = MAX(1, min);
    const size_t tauMemSize = ltau * ltau * BATCHSIZE * sizeof(float);
    const size_t AHatMemSize = n1 * n2 * BATCHSIZE * sizeof(float);
    float* tau = (float*) malloc(tauMemSize);
    float* h_AHat[BATCHSIZE];
    float* h_Tau[BATCHSIZE];
    float** d_AHat;
    float** d_Tau;
    int info;

    // set h_AHat and h_Tau
    for (int i = 0; i < BATCHSIZE; i++) {
        h_AHat[i] = AHat + i * n1 * n2;
        h_Tau[i] = tau + i * ltau;
    }

    printf("ltau: %d\n", ltau);

    // qr initialization
    gpuAssert(
        cudaMalloc((void**) &d_AHat, AHatMemSize));
    printf("malloc d_AHat\n");
    gpuAssert(
        cudaMalloc((void**) &d_Tau, tauMemSize));
    printf("malloc d_Tau\n");

    gpuAssert(
        cudaMemcpy(d_AHat, h_AHat, AHatMemSize, cudaMemcpyHostToDevice));
    printf("copy AHat to d_AHat\n");
    gpuAssert(
        cudaMemcpy(d_Tau, h_Tau, tauMemSize, cudaMemcpyHostToDevice));
    printf("memset d_Tau\n");

    stat = cublasSgeqrfBatched(cHandle,
                                n1,
                                n2,
                                d_AHat,
                                lda,
                                d_Tau,
                                &info,
                                BATCHSIZE);
    
    if (info != 0) {
        printf("\nparameters are invalid\n");
    }
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("\ncublasSgeqrfBatched failed");
        printf("\ncublas error: %d\n", stat);
    }

    // for (int i = 0; i < ltau * ltau; i++) {
    //     gpuAssert(
    //         cudaMemcpy(h_Tau + i, d_Tau[i], sizeof(float), cudaMemcpyDeviceToHost));
    // }
    printf("after cublasSgeqrfBatched\n");
    printKernel <<< 1, 1 >>> (d_Tau, ltau);
    gpuAssert(
        cudaDeviceSynchronize());
    printf("after printKernel\n");
    gpuAssert(
        cudaMemcpy(h_Tau, d_Tau, tauMemSize, cudaMemcpyDeviceToHost));
    printf("copy d_Tau to h_Tau\n");

    printf("\nh_Tau: ");
    for (int i = 0; i < ltau * BATCHSIZE; i++) {
        printf("\nith vector: ");
        for (int j = 0; j < ltau; j++) {
            printf("%f ", h_Tau[i * ltau + j]);
        }
    }
    printf("\nh_Tau: %f", h_Tau[0]);


    // free and destroy
    gpuAssert(
        cudaFree(d_AHat));
    gpuAssert(
        cudaFree(d_Tau));
    free(h_Tau);
    cublasDestroy(cHandle);
    return 0;
}

#endif
