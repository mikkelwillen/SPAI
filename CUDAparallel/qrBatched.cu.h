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
__global__ void copyRFromAHat(float** d_PointerAHat, float** d_PointerR, int n1, int n2, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n1 * n2 * batchsize) {
        int b = tid / (n1 * n2);
        int i = (tid % (n1 * n2)) / n1;
        int j = (tid % (n1 * n2)) % n1;

        float* d_AHat = d_PointerAHat[b];
        float* d_R = d_PointerR[b];
        if (i >= j) {
            d_R[i * n1 + j] = d_AHat[i * n1 + j];
        } else {
            d_R[tid] = 0.0;
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

// kernel for setting v
// d_PointerAHat = an array of pointers to the start of each AHat matrix in d_AHat
// d_PointerV = an array of pointers to the start of each V matrix in d_V
// n1 = the max number of rows of the matrices
// n2 = the max number of columns of the matrices
// batchsize = the number of matrices in the batch
__global__ void makeV(float** d_PointerAHat, float** d_PointerV, int n1, int n2, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n1 * n2 * batchsize) {
        int b = tid / (n1 * n2);
        int k = (tid % (n1 * n2)) / n1;
        int i = (tid % (n1 * n2)) % n1;

        float* d_AHat = d_PointerAHat[b];
        float* d_v = d_PointerV[b];

        if (k > i) {
            d_v[k * n1 + i] = 0.0;
        } else if (k == i) {
            d_v[k * n1 + i] = 1.0;
        } else {
            d_v[k * n1 + i] = d_AHat[k * n1 + i];
        }
    }
}

// kernel for computing Q * v
// d_PointerQ = an array of pointers to the start of each Q matrix in d_Q
// d_PointerV = an array of pointers to the start of each V matrix in d_V
// d_PointerQv = an array of pointers to the start of each Qv matrix in d_Qv
// n1 = the max number of rows of the matrices
// n2 = the max number of columns of the matrices
// batchsize = the number of matrices in the batch
__global__ void computeQtimesV(float** d_PointerQ, float** d_PointerV, float** d_PointerQv, int n1, int n2, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n1 * n1 * n2 * batchsize) {
        int b = tid / (n1 * n1 * n2);
        int k = (tid % (n1 * n1 * n2)) / (n1 * n1);
        int i = ((tid % (n1 * n1 * n2)) % (n1 * n1)) / n1;
        int j = ((tid % (n1 * n1 * n2)) % (n1 * n1)) % n1;

        float* d_v = d_PointerV[b];
        float* d_Q = d_PointerQ[b];
        float* d_Qv = d_PointerQv[b];

        if (j == 0) {
            d_Qv[k * n1 + i] = 0.0;
        }
        __syncthreads();
        
        d_Qv[k * n1 + i] += d_Q[i * n1 + j] * d_v[k * n1 + j];
    }
}

// kernel for computing Qv * v^T
// d_PointerQv = an array of pointers to the start of each Qv matrix in d_Qv
// d_PointerV = an array of pointers to the start of each V matrix in d_V
// d_PointerQvvt = an array of pointers to the start of each Qvvt matrix in d_Qvvt
// d_PointerTau = an array of pointers to the start of each tau vector in d_tau
// n1 = the max number of rows of the matrices
// n2 = the max number of columns of the matrices
// batchsize = the number of matrices in the batch
__global__ void computeQvTimesVtransposed(float** d_PointerQv, float** d_PointerV, float** d_PointerQvvt, float** d_PointerTau, int n1, int n2, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n1 * n1 * n2 * batchsize) {
        int b = tid / (n1 * n1 * n2);
        int k = (tid % (n1 * n1 * n2)) / (n1 * n1);
        int i = ((tid % (n1 * n1 * n2)) % (n1 * n1)) / n1;
        int j = ((tid % (n1 * n1 * n2)) % (n1 * n1)) % n1;

        float* d_tau = d_PointerTau[b];
        float* d_v = d_PointerV[b];
        float* d_Qv = d_PointerQv[b];
        float* d_Qvvt = d_PointerQvvt[b];
        
        d_Qvvt[i * n1 + j] = d_tau[k] * d_Qv[k * n1 + i] * d_v[k * n1 + j];
    }
}

// kernel for computing Q - Qvvt
// d_PointerQ = an array of pointers to the start of each Q matrix in d_Q
// d_PointerQvvt = an array of pointers to the start of each Qvvt matrix in d_Qvvt
// n1 = the max number of rows of the matrices
// n2 = the max number of columns of the matrices
// batchsize = the number of matrices in the batch
__global__ void computeQminusQvvt(float** d_PointerQ, float** d_PointerQvvt, int n1, int n2, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n1 * n1 * batchsize) {
        int b = tid / (n1 * n1);
        int i = (tid % (n1 * n1)) / n1;
        int j = (tid % (n1 * n1)) % n1;

        float* d_Q = d_PointerQ[b];
        float* d_Qvvt = d_PointerQvvt[b];
        
        d_Q[i * n1 + j] -= d_Qvvt[i * n1 + j];
    }
}

// kernel to copy d_data to d_Pointer
// d_Pointer = an array of pointers to the start of each matrix in d_data
// d_data = an array of batch matrices
// pointerArraySize = the size of d_Pointer
// dataSize = the size of each matrix in d_data
__global__ void deviceToDevicePointerKernel(float** d_Pointer, float* d_data, int pointerArraySize, int dataSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pointerArraySize) {
        d_Pointer[tid] = &d_data[tid * dataSize];
    }
}

// Function to do QR decomposition of batch AHat matrices
// AHat = an array of batch matrices
// n1 = the max number of rows of the matrices
// n2 = the max number of columns of the matrices
// Q = an array of batch Q matrices
// R = an array of batch R matrices
int qrBatched(cublasHandle_t cHandle, float** d_PointerAHat, float** d_PointerQ, float** d_PointerR, int batchsize, int n1, int n2) {
    printf("\nDo QR decomposition of AHat\n");

    // Set constants
    cublasStatus_t stat;
    int lda = n1;
    int min = MIN(n1, n2);
    int ltau = MAX(1, min);
    const size_t tauMemSize = ltau * batchsize * sizeof(float);
    const size_t tauPointerMemSize = batchsize * sizeof(float*);

    // create input and output arrays
    float* d_tau;
    float** d_PointerTau;
    int info;
    printf("after creating input and output arrays\n");
    
    // malloc space for tau
    gpuAssert(
        cudaMalloc((void**) &d_tau, tauMemSize));
    gpuAssert(
        cudaMalloc((void**) &d_PointerTau, tauPointerMemSize));
    deviceToDevicePointerKernel <<< 1, batchsize * ltau >>> (d_PointerTau, d_tau, batchsize * ltau, ltau);

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

    // malloc space for arrays and copy array start pointers to device
    float* d_v;
    float** d_PointerV;

    float* d_Qv;
    float** d_PointerQv;

    float* d_Qvvt;
    float** d_PointerQvvt;

    gpuAssert(
        cudaMalloc((void**) &d_v, n1 * n2 * batchsize * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerV, batchsize * sizeof(float*)));
    deviceToDevicePointerKernel <<< 1, batchsize * n1 * n2 >>> (d_PointerV, d_v, batchsize, n1 * n2);

    gpuAssert(
        cudaMalloc((void**) &d_Qv, n1 * n2 * batchsize * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerQv, batchsize * sizeof(float*)));
    deviceToDevicePointerKernel <<< 1, batchsize * n1 * n2 >>> (d_PointerQv, d_Qv, batchsize, n1 * n2);

    gpuAssert(
        cudaMalloc((void**) &d_Qvvt, n1 * n1 * batchsize * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerQvvt, batchsize * sizeof(float*)));
    deviceToDevicePointerKernel <<<1, batchsize * n1 * n1 >>> (d_PointerQvvt, d_Qvvt, batchsize, n1 * n1);

    // copy R from AHat
    copyRFromAHat <<<1, n1 * n2 * batchsize >>> (d_PointerAHat, d_PointerR, n1, n2, batchsize);

    // set Q to I
    setQToIdentity <<<1, n1 * n1 * batchsize>>>(d_PointerQ, n1, batchsize);

    // compute Q * v
    
    computeQtimesV <<<1, n1 * n1 * n2 * batchsize>>>(d_PointerQ, d_PointerAHat, d_PointerV, n1, n2, batchsize);

    // compute Qv * v^T
    computeQvTimesVtransposed <<<1, n1 * n1 * n2 * batchsize>>>(d_PointerQv, d_PointerV, d_PointerQvvt, d_PointerTau, n1, n2, batchsize);

    // compute Q - Qvvt
    computeQminusQvvt <<<1, n1 * n1 * batchsize>>>(d_PointerQ, d_PointerQvvt, n1, n2, batchsize);

    // free arrays and destroy cHandle
    cudaFree(d_tau);
    cudaFree(d_PointerTau);
    cudaFree(d_v);
    cudaFree(d_PointerV);
    cudaFree(d_Qv);
    cudaFree(d_PointerQv);
    cudaFree(d_Qvvt);
    cudaFree(d_PointerQvvt);
    
    return 0;
}

#endif
