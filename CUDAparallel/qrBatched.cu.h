#ifndef QR_BATCHED_H
#define QR_BATCHED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"
#include "helperKernels.cu.h"

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
__global__ void makeV(float** d_PointerAHat, float** d_PointerV, int n1, int k, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n1 * batchsize) {
        int b = tid / n1;
        int i = (tid % n1);

        float* d_AHat = d_PointerAHat[b];
        float* d_v = d_PointerV[b];

        if (k > i) {
            d_v[i] = 0.0;
        } else if (k == i) {
            d_v[i] = 1.0;
        } else {
            d_v[i] = d_AHat[k * n1 + i];
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
__global__ void computeQtimesV(float** d_PointerQ, float** d_PointerV, float** d_PointerQv, int n1, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n1 * n1 * batchsize) {
        int b = tid / (n1 * n1);
        int i = (tid % (n1 * n1)) / n1;
        int j = (tid % (n1 * n1)) % n1;

        float* d_Q = d_PointerQ[b];
        float* d_v = d_PointerV[b];
        float* d_Qv = d_PointerQv[b];

        if (j == 0) {
            d_Qv[i] = 0.0;
        }
        __syncthreads();
        printf("b: %d, i: %d, j: %d" b, i, j);
        d_Qv[i] += d_Q[i * n1 + j] * d_v[j];
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
__global__ void computeQvTimesVtransposed(float** d_PointerQv, float** d_PointerV, float** d_PointerQvvt, float** d_PointerTau, int n1, int k, int batchsize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n1 * n1 * batchsize) {
        int b = tid / (n1 * n1);
        int i = (tid % (n1 * n1)) / n1;
        int j = (tid % (n1 * n1)) % n1;

        float* d_tau = d_PointerTau[b];
        float* d_v = d_PointerV[b];
        float* d_Qv = d_PointerQv[b];
        float* d_Qvvt = d_PointerQvvt[b];
        
        d_Qvvt[i * n1 + j] = d_tau[k] * d_Qv[i] * d_v[j];
    }
}

// kernel for computing Q - Qvvt
// d_PointerQ = an array of pointers to the start of each Q matrix in d_Q
// d_PointerQvvt = an array of pointers to the start of each Qvvt matrix in d_Qvvt
// n1 = the max number of rows of the matrices
// n2 = the max number of columns of the matrices
// batchsize = the number of matrices in the batch
__global__ void computeQminusQvvt(float** d_PointerQ, float** d_PointerQvvt, int n1, int batchsize) {
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
    int numBlocks;

    // create input and output arrays
    float* d_tau;
    float** d_PointerTau;
    int info;
    
    // malloc space for tau
    gpuAssert(
        cudaMalloc((void**) &d_tau, tauMemSize));
    gpuAssert(
        cudaMalloc((void**) &d_PointerTau, tauPointerMemSize));

    numBlocks = (batchsize * ltau + BLOCKSIZE - 1) / BLOCKSIZE;
    deviceToDevicePointerKernel <<< numBlocks, BLOCKSIZE>>> (d_PointerTau, d_tau, batchsize * ltau, ltau);

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
        cudaMalloc((void**) &d_v, n1 * batchsize * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerV, batchsize * sizeof(float*)));
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    deviceToDevicePointerKernel <<< numBlocks, BLOCKSIZE >>> (d_PointerV, d_v, batchsize, n1);

    gpuAssert(
        cudaMalloc((void**) &d_Qv, n1 * batchsize * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerQv, batchsize * sizeof(float*)));
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    deviceToDevicePointerKernel <<< numBlocks, BLOCKSIZE >>> (d_PointerQv, d_Qv, batchsize, n1);

    gpuAssert(
        cudaMalloc((void**) &d_Qvvt, n1 * n1 * batchsize * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerQvvt, batchsize * sizeof(float*)));
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    deviceToDevicePointerKernel <<<numBlocks, BLOCKSIZE >>> (d_PointerQvvt, d_Qvvt, batchsize, n1 * n1);

    // copy R from AHat
    numBlocks = (n1 * n2 * batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    copyRFromAHat <<<numBlocks, BLOCKSIZE >>> (d_PointerAHat, d_PointerR, n1, n2, batchsize);

    // set Q to I
    numBlocks = (n1 * n1 * batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    setQToIdentity <<<numBlocks, BLOCKSIZE>>>(d_PointerQ, n1, batchsize);

    for (int k = 0; k < n2; k++) {
        // make v
        numBlocks = (n1 * batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        makeV <<<numBlocks, BLOCKSIZE>>>(d_PointerAHat, d_PointerV, n1, k, batchsize);
        float* h_v = (float*) malloc(n1 * batchsize * sizeof(float));
        gpuAssert(
            cudaMemcpy(h_v, d_v, n1 * batchsize * sizeof(float), cudaMemcpyDeviceToHost));
        for (int b = 0; b < batchsize; b++) {
            printf("v[%d] = [", b);
            for (int i = 0; i < n1; i++) {
                printf("%f, ", h_v[b * n1 + i]);
            }
            printf("]\n");
        }

        // compute Q * v
        numBlocks = (n1 * n1 * batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        computeQtimesV <<<numBlocks, BLOCKSIZE>>>(d_PointerQ, d_PointerV, d_PointerQv, n1, batchsize);
        float* h_Qv = (float*) malloc(n1 * batchsize * sizeof(float));
        gpuAssert(
            cudaMemcpy(h_Qv, d_Qv, n1 * batchsize * sizeof(float), cudaMemcpyDeviceToHost));
        for (int b = 0; b < batchsize; b++) {
            printf("Qv[%d] = [", b);
            for (int i = 0; i < n1; i++) {
                printf("%f, ", h_Qv[b * n1 + i]);
            }
            printf("]\n");
        }

        // compute Qv * v^T
        numBlocks = (n1 * n1 * batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        computeQvTimesVtransposed <<<numBlocks, BLOCKSIZE >>>(d_PointerQv, d_PointerV, d_PointerQvvt, d_PointerTau, n1, k, batchsize);
        float* h_Qvvt = (float*) malloc(n1 * n1 * batchsize * sizeof(float));
        gpuAssert(
            cudaMemcpy(h_Qvvt, d_Qvvt, n1 * n1 * batchsize * sizeof(float), cudaMemcpyDeviceToHost));
        for (int b = 0; b < batchsize; b++) {
            printf("Qvvt[%d] = [", b);
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < n1; j++) {
                    printf("%f, ", h_Qvvt[b * n1 * n1 + i * n1 + j]);
                }
                printf("\n");
            }
            printf("]\n");
        }

        // compute Q - Qvvt
        numBlocks = (n1 * n1 * batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        computeQminusQvvt <<<numBlocks, BLOCKSIZE>>>(d_PointerQ, d_PointerQvvt, n1, batchsize);
    }

    // free arrays and destroy cHandle
    gpuAssert(
        cudaFree(d_tau));
    gpuAssert(
        cudaFree(d_PointerTau));
    gpuAssert(
        cudaFree(d_v));
    gpuAssert(
        cudaFree(d_PointerV));
    gpuAssert(
        cudaFree(d_Qv));
    gpuAssert(
        cudaFree(d_PointerQv));
    gpuAssert(
        cudaFree(d_Qvvt));
    gpuAssert(
        cudaFree(d_PointerQvvt));
    
    return 0;
}

#endif
