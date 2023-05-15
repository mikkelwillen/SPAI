#ifndef QR_BATCHED_H
#define QR_BATCHED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"

// kernel for copying R from AHat
// d_PointerAHat = an array of pointers to the start of each AHat matrix in d_AHat
// d_R = an array of pointers to the start of each R matrix in d_R
// n1 = the max number of rows of the matrices
// n2 = the max number of columns of the matrices
// batchsize = the number of matrices in the batch
__global__ void copyRFromAHat(float** d_PointerAHat, float** d_R, int n1, int n2, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n1 * n2 * batchsize) {
        int b = tid / (n1 * n2);
        int i = (tid % (n1 * n2)) / n1;
        int j = (tid % (n1 * n2)) % n1;

        float* d_AHat = d_PointerAHat[b];

        if (i >= j) {
            (*d_R)[i * n1 + j] = d_AHat[i * n1 + j];
        } else {
            (*d_R)[tid] = 0.0;
        }
    }
}

// kernel for setting Q to I
// d_Q = an array of pointers to the start of each Q matrix in d_Q
// n1 = the max number of rows of the matrices
// batchsize = the number of matrices in the batch
__global__ void setQToIdentity(float** d_PointerQ, int n1, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n1 * n1 * batchsize) {
        int b = tid / (n1 * n1);
        int i = (tid % (n1 * n1)) / n1;
        int j = (tid % (n1 * n1)) % n1;

        float* d_Q = d_PointerQ[b];

        if (i == j) {
            d_Q[i * n1 + j] = 1.0;
        } else {
            d_Q[i * n1 + j] = 0.0;
        }
    }
}

// vi skal lave kernel/kernels for Kerr Campell algorithm 1




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
// AHat is an array of batch matrices
// n1 is the max number of rows of the matrices
// n2 is the max number of columns of the matrices
// Q is an array of batch Q matrices
// R is an array of batch R matrices
int qrBatched(cublasHandle_t cHandle, float** d_PointerAHat, float** d_Q, float** d_R, int batchsize, int n1, int n2) {d
    printf("\nDo QR decomposition of AHat\n");

    // Set constants
    cublasStatus_t stat;
    int lda = n1;
    int min = MIN(n1, n2);
    int ltau = MAX(1, min);
    const size_t tauMemSize = ltau * batchsize * sizeof(float);
    const size_t tauPointerMemSize = batchsize * sizeof(float*);

    // create input and output arrays
    float* h_tau = (float*) malloc(tauMemSize);
    float* d_tau;
    float** d_PointerTau;
    int info;
    printf("after creating input and output arrays\n");
    
    // malloc space for tau
    gpuAssert(
        cudaMalloc((void**) &d_tau, tauMemSize));
    gpuAssert(
        cudaMalloc((void**) &d_PointerTau, tauPointerMemSize));
    tauDeviceToDevicePointerKernel <<< 1, batchsize * ltau >>> (d_PointerTau, d_tau, batchsize, ltau);

    // run QR decomposition from cublas
    // cublas docs: https://docs.nvidia.com/cuda/cublas
    stat = cublasSgeqrfBatched(cHandle,
                               n1,
                               n2,
                               d_PointerAHat,
                               lda,
                               d_PointerTau,
                               &info,
                               batchsize);
    
    // error handling
    if (info != 0) {
        printf("Parameter error %d in QR decomposition\n", info);
    }
    
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgeqrfBatched failed\n");
        printf("cublas error: %d\n", stat);

        return stat;
    }

    // copy AHat and tau back to host
    gpuAssert(
        cudaMemcpy((*AHat), d_AHat, AHatMemSize, cudaMemcpyDeviceToHost));
    gpuAssert(
        cudaMemcpy(h_tau, d_tau, tauMemSize, cudaMemcpyDeviceToHost));
    
    // copy R from AHat
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n1; j++) {
            if (i >= j) {
                (*R)[i * n1 + j] = (*AHat)[i * n1 + j];
            } else {
                (*R)[i * n1 + j] = 0;
            }
        }
    }

    // make Q with Algorithm 1 from Kerr Campbell Richards QRD on GPUs
    // set Q to I
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n1; j++) {
            (*Q)[i * n1 + j] = 0;
        }
        (*Q)[i * n1 + i] = 1;
    }

    // do for loop
    for (int k = 0; k < n2; k++) {
        // make v
        float* v = (float*) malloc(n1 * sizeof(float));
        for (int i = 0; i < n1; i++) {
            if (k > i) {
                v[i] = 0;
            } else if (k == i) {
                v[i] = 1;
            } else {
                v[i] = (*AHat)[k * n1 + i];
            }
        }

        // compute Q * v
        float* Qv = (float*) malloc(n1 * sizeof(float));
        for (int i = 0; i < n1; i++) {
            Qv[i] = 0;
            for (int j = 0; j < n1; j++) {
                Qv[i] += (*Q)[i * n1 + j] * v[j];
            }
        }

        // compute Qv * v^T
        float* Qvvt = (float*) malloc(n1 * n1 * sizeof(float));
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n1; j++) {
                Qvvt[i * n1 + j] = h_tau[k] * Qv[i] * v[j];
            }
        }
        // compute Q - Qv * v^T
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n1; j++) {
                (*Q)[i * n1 + j] -= Qvvt[i * n1 + j];
            }
        }
    }

    // print various matrices
    {
        printf("R: \n");
        for (int j = 0; j < n1; j++) {
            for (int i = 0; i < n2; i++) {
                printf("%f ", (*R)[i * n1 + j]);
            }
            printf("\n");
        }
        

        printf("Q: \n");
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n1; j++) {
                printf("%f ", (*Q)[i * n1 + j]);
            }
            printf("\n");
        }

        // print tau
        for (int i = 0; i < BATCHSIZE; i++) {
            printf("tau %d:", i);
            for (int k = 0; k < ltau; k++) {
                printf("%f ", h_tau[i * ltau + k]);
            }
            printf("\n");
        }
        
        // print AHat
        for (int i = 0; i < BATCHSIZE; i++) {
            printf("AHat %d:\n", i);
            for (int j = 0; j < n1; j++) {
                for (int k = 0; k < n2; k++) {
                    printf("%f ", (*AHat)[i * n1 * n2 + j * n2 + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    // free arrays and destroy cHandle
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
