#ifndef QR_BATCHED_H
#define QR_BATCHED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"

// Kernel to copy d_AHat to d_PointerAHat
// d_AHat is an array of batch matrices
// d_PointerAHat is an array of pointers to the start of each matrix in d_AHat
// batch is the BATCHSIZE
// n1 is the number of rows in each matrix
// n2 is the number of columns in each matrix
__global__ void AHatDeviceToDevicePointerKernel(float** d_AHat, float* h_AHat, int batch, int n1, int n2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < BATCHSIZE) {
        d_AHat[tid] = &h_AHat[tid * n1 * n2];
    }
}

// Kernel to copy d_tau to d_PointerTau
// d_tau is an array of batch tau vectors
// d_PointerTau is an array of pointers to the start of each tau vector in d_tau
// batch is the BATCHSIZE
// ltau is the length of each tau vector
__global__ void tauDeviceToDevicePointerKernel(float** d_Tau, float* h_Tau, int batch, int ltau) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < BATCHSIZE * ltau) {
        d_Tau[tid] = &h_Tau[tid * ltau];
    }
}

// Function to do QR decomposition of batch AHat matrices
// cHandle is the cublas handle
// AHat is an array of pointers to batch matrices
// n1 is the max number of rows of the matrices
// n2 is the max number of columns of the matrices
// Q is an array of pointers to batch Q matrices
// R is an array of pointers batch R matrices
// returns 0 if succesful, 1 if not
int qrBatched(cublasHandle_t cHandle, float** AHat, int n1, int n2, float** Q, float** R) {
    // Set constants
    cublasStatus_t stat;
    int lda = n1;
    int min = MIN(n1, n2);
    int ltau = MAX(1, min);
    const size_t tauMemSize = ltau * BATCHSIZE * sizeof(float);
    const size_t tauPointerMemSize = BATCHSIZE * sizeof(float*);
    const size_t AHatMemSize = n1 * n2 * BATCHSIZE * sizeof(float);
    const size_t AHatPointerMemSize = BATCHSIZE * sizeof(float*);

    // Create input and output arrays
    float* h_tau = (float*) malloc(tauMemSize);
    float* d_AHat;
    float* d_tau;
    float** d_PointerAHat;
    float** d_PointerTau;
    int info;

    // Malloc space and copy data for AHat
    gpuAssert(
        cudaMalloc((void**) &d_AHat, AHatMemSize));
    gpuAssert(
        cudaMemcpy(d_AHat, (*AHat), AHatMemSize, cudaMemcpyHostToDevice));
    gpuAssert(
        cudaMalloc((void**) &d_PointerAHat, AHatPointerMemSize));
    AHatDeviceToDevicePointerKernel <<< 1, BATCHSIZE >>> (d_PointerAHat, d_AHat, BATCHSIZE, n1, n2);
    
    // Malloc space for tau
    gpuAssert(
        cudaMalloc((void**) &d_tau, tauMemSize));
    gpuAssert(
        cudaMalloc((void**) &d_PointerTau, tauPointerMemSize));
    tauDeviceToDevicePointerKernel <<< 1, BATCHSIZE * ltau >>> (d_PointerTau, d_tau, BATCHSIZE, ltau);

    // Run QR decomposition from cublas
    // Cublas docs: https://docs.nvidia.com/cuda/cublas
    stat = cublasSgeqrfBatched(cHandle,
                               n1,
                               n2,
                               d_PointerAHat,
                               lda,
                               d_PointerTau,
                               &info,
                               BATCHSIZE);
    
    // Error handling
    if (info != 0) {
        printf("Parameter error %d in QR decomposition\n", info);
    }
    
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgeqrfBatched failed\n");
        printf("cublas error: %d\n", stat);

        return stat;
    }

    // Copy AHat and tau back to host
    gpuAssert(
        cudaMemcpy((*AHat), d_AHat, AHatMemSize, cudaMemcpyDeviceToHost));
    gpuAssert(
        cudaMemcpy(h_tau, d_tau, tauMemSize, cudaMemcpyDeviceToHost));
    
    // Copy R from AHat
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            if (i <= j) {
                (*R)[i * n2 + j] = (*AHat)[i * n2 + j];
            } else {
                (*R)[i * n2 + j] = 0.0;
            }
        }
    }

    // Make Q with Algorithm 1 from Kerr Campbell Richards QRD on GPUs
    // Set Q to I
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n1; j++) {
            (*Q)[i * n1 + j] = 0.0;
        }
        (*Q)[i * n1 + i] = 1.0;
    }

    // Do for loop
    for (int k = 0; k < n2; k++) {
        // Make v
        float* v = (float*) malloc(n1 * sizeof(float));
        for (int i = 0; i < n1; i++) {
            if (k > i) {
                v[i] = 0.0;
            } else if (k == i) {
                v[i] = 1.0;
            } else {
                v[i] = (*AHat)[k * n1 + i];
            }
        }

        // Compute Q * v
        float* Qv = (float*) malloc(n1 * sizeof(float));
        for (int i = 0; i < n1; i++) {
            Qv[i] = 0.0;
            for (int j = 0; j < n1; j++) {
                Qv[i] += (*Q)[i * n1 + j] * v[j];
            }
        }

        // Compute Qv * v^T
        float* Qvvt = (float*) malloc(n1 * n1 * sizeof(float));
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n1; j++) {
                Qvvt[i * n1 + j] = h_tau[k] * Qv[i] * v[j];
            }
        }
        // Compute Q - Qv * v^T
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n1; j++) {
                (*Q)[i * n1 + j] -= Qvvt[i * n1 + j];
            }
        }

        // Free memory
        free(v);
        free(Qv);
        free(Qvvt);
    }

    // Free arrays and destroy cHandle
    free(h_tau);

    gpuAssert(
        cudaFree(d_AHat));
    gpuAssert(
        cudaFree(d_tau));
    gpuAssert(
        cudaFree(d_PointerAHat));
    gpuAssert(
        cudaFree(d_PointerTau));
    
    return 0;
}

#endif
