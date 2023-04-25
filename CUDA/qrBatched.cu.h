#ifndef QR_BATCHED_H
#define QR_BATCHED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"

// vi skal have kigget på nogle forskellige størrelser
// fx skal vi finde ud af, om vi vil kopiere alt data til GPU'en med det samme, eller om vi skal kopiere hver aHat separat og så lave en batch af dem

__global__ void printKernel(int length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        printf("tid %d: \n", tid);
    }
}

__global__ void printDeviceArrayKernel(float* h_AHat, int length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        printf("tid %d: %f\n", tid, h_AHat[tid]);
    }
}

__global__ void deviceToDevicePointerKernel(float** d_AHat, float* h_AHat, int batch, int n1, int n2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < BATCHSIZE) {
        d_AHat[tid] = &h_AHat[tid * n1 * n2];
    }
}

__global__ void devicePointerToDeviceKernel(float** d_tau, float* h_tau, int batch, int ltau) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < BATCHSIZE * ltau * ltau) {
        int i = tid / ltau;
        int j = tid % ltau;
        h_tau[tid] = d_tau[i][j];
    }
}


// virker sgu ikke rigtigt
__global__ void printDeviceArrayPointerKernel(float** d_AHat, int length, int batch) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("hallo?");
    if (tid < length) {
        printf("tid %d: %f\n", tid, d_AHat[batch][tid]);
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
    float* h_AHat;
    float* h_Tau;
    float** d_AHat;
    float** d_Tau;
    int info;

    gpuAssert(
        cudaMalloc((void**) &h_AHat, AHatMemSize));
    gpuAssert(
        cudaMemcpy(h_AHat, AHat, n1 * n2 * sizeof(float), cudaMemcpyHostToDevice));
    gpuAssert(
        cudaMalloc((void**) &d_AHat, BATCHSIZE * sizeof(float*)));
    
    deviceToDevicePointerKernel <<< 1, BATCHSIZE >>> (d_AHat, h_AHat, BATCHSIZE, n1, n2);
    
    gpuAssert(
        cudaMalloc((void**) &h_Tau, tauMemSize));
    gpuAssert(
        cudaMalloc((void**) &d_Tau, BATCHSIZE * ltau * sizeof(float*)));

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

    printf("after cublasSgeqrfBatched\n");
    // copy d_Tau to h_Tau
    devicePointerToDeviceKernel <<< 1, BATCHSIZE * ltau * ltau >>> (d_Tau, h_Tau, BATCHSIZE, ltau);
    printf("after devicePointerToDeviceKernel\n");
    printDeviceArrayKernel <<< 1, BATCHSIZE * ltau * ltau >>> (h_Tau, BATCHSIZE * ltau * ltau);
    printf("after printDeviceArrayKernel\n");
    gpuAssert(
        cudaMemcpy(tau, h_Tau, tauMemSize, cudaMemcpyDeviceToHost));
    printf("after cudaMemcpy\n");
    
    // print tau
    printf("\nTau: ");
    for (int i = 0; i < ltau * BATCHSIZE; i++) {
        printf("\nith vector: ");
        for (int j = 0; j < ltau; j++) {
            printf("%f ", tau[i * ltau + j]);
        }
    }
    printf("after tau print\n");

    // // for (int i = 0; i < ltau * ltau; i++) {
    // //     gpuAssert(
    // //         cudaMemcpy(h_Tau + i, d_Tau[i], sizeof(float), cudaMemcpyDeviceToHost));
    // // }
    // printf("after cublasSgeqrfBatched\n");
    // int numberOfBlocks = (n1 * n2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // printKernel <<< numberOfBlocks,  BLOCK_SIZE >>> (n1 * n2);
    // gpuAssert(
    //     cudaDeviceSynchronize());
    // printf("after printKernel\n");
    // gpuAssert(
    //     cudaMemcpy(h_Tau, d_Tau, tauMemSize, cudaMemcpyDeviceToHost));
    // printf("copy d_Tau to h_Tau\n");

    // printf("\nh_Tau: ");
    // for (int i = 0; i < ltau * BATCHSIZE; i++) {
    //     printf("\nith vector: ");
    //     for (int j = 0; j < ltau; j++) {
    //         printf("%f ", h_Tau[i * ltau + j]);
    //     }
    // }
    // printf("\nh_Tau: %f", h_Tau[0]);


    // // free and destroy
    // gpuAssert(
    //     cudaFree(d_AHat));
    // gpuAssert(
    //     cudaFree(d_Tau));
    // free(h_Tau);
    // cublasDestroy(cHandle);
    return 0;
}

#endif
