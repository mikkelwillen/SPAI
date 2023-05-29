#ifndef UPDATE_QR_H
#define UPDATE_QR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"
#include "permutation.cu.h"
#include "LSProblem.cu.h"
#include "helperKernels.cu.h"
#include "SPAIkernels.cu.h"

/* Function for updating the QR decomposition
cHandle = cublas handle
d_A = pointer to A in device memory
d_PointerQ = pointer to Q in device memory
d_PointerR = pointer to R in device memory
d_PointerI = pointer to I in device memory
d_PointerJ = pointer to J in device memory
d_PointerSortedJ = pointer to sortedJ in device memory
d_PointerITilde = pointer to ITilde in device memory
d_PointerJTilde = pointer to JTilde in device memory
d_n1 = pointer to n1 in device memory
d_n2 = pointer to n2 in device memory
d_n1Tilde = pointer to n1Tilde in device memory
d_n2Tilde = pointer to n2Tilde in device memory
d_n1Union = pointer to n1Union in device memory
d_n2Union = pointer to n2Union in device memory
d_PointerMHat_k = pointer to mHat_k in device memory
d_residualNorm = pointer to residualNorm in device memory
maxn1 = maximum value of n1
maxn2 = maximum value of n2
maxn1Tilde = maximum value of n1Tilde
maxn2Tilde = maximum value of n2Tilde
maxn1Union = maximum value of n1Union
maxn2Union = maximum value of n2Union
i = current iteration
batchsize = batchsize */
int updateQR(cublasHandle_t cHandle, CSC* A, CSC* d_A, float** d_PointerQ, float** d_PointerR, int** d_PointerI, int** d_PointerJ, int** d_PointerSortedJ, int** d_PointerITilde, int** d_PointerJTilde, int* d_n1, int* d_n2, int* d_n1Tilde, int* d_n2Tilde, int* d_n1Union, int* d_n2Union, float** d_PointerMHat_k, float* d_residualNorm, int maxn1, int maxn2, int maxn1Tilde, int maxn2Tilde, int maxn1Union, int maxn2Union, int i, int batchsize) {
    printf("\n------UPDATE QR------\n");
    int numBlocks;

    // create AIJTilde
    float* d_AIJTilde;
    float** d_PointerAIJTilde;

    gpuAssert(
        cudaMalloc((void**) &d_AIJTilde,  maxn1 * maxn2Tilde * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerAIJTilde, batchsize * sizeof(float*)));

    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerAIJTilde, d_AIJTilde, batchsize, maxn1 * maxn2Tilde);
    
    numBlocks = (batchsize * maxn1 * maxn2Tilde * A->m + BLOCKSIZE - 1) / BLOCKSIZE;
    CSCToBatchedDenseMatrices<<<numBlocks, BLOCKSIZE>>>(d_A, d_PointerAIJTilde, d_PointerI, d_PointerJTilde, d_n1, d_n2Tilde, maxn1, maxn2Tilde, A->m, batchsize);

    // create AITildeJTilde
    float* d_AITildeJTilde;
    float** d_PointerAITildeJTilde;

    gpuAssert(
        cudaMalloc((void**) &d_AITildeJTilde,  maxn1Tilde * maxn2Tilde * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerAITildeJTilde, batchsize * sizeof(float*)));
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerAITildeJTilde, d_AITildeJTilde, batchsize, maxn1Tilde * maxn2Tilde);

    numBlocks = (batchsize * maxn1Tilde * maxn2Tilde * A->m + BLOCKSIZE - 1) / BLOCKSIZE;
    CSCToBatchedDenseMatrices<<<numBlocks, BLOCKSIZE>>>(d_A, d_PointerAITildeJTilde, d_PointerITilde, d_PointerJTilde, d_n1Tilde, d_n2Tilde, maxn1Tilde, maxn2Tilde, A->m, batchsize);

    // create permutation matrices Pr and Pc
    int* d_Pr;
    int* d_Pc;

    int** d_PointerPr;
    int** d_PointerPc;

    gpuAssert(
        cudaMalloc((void**) &d_Pr,  maxn1 * maxn1 * sizeof(int)));
    gpuAssert(
        cudaMalloc((void**) &d_Pc,  maxn2 * maxn2 * sizeof(int)));

    gpuAssert(
        cudaMalloc((void**) &d_PointerPr, batchsize * sizeof(int*)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerPc, batchsize * sizeof(int*)));
    
    // numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    // intDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerPr, d_Pr, batchsize, maxn1 * maxn1);

    // intDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerPc, d_Pc, batchsize, maxn2 * maxn2);


    printf("done with updateQR\n");

    return 0;
}

#endif