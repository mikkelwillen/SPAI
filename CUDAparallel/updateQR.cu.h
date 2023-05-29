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
d_PointerIUnion = pointer to IUnion in device memory
d_PointerJUnion = pointer to JUnion in device memory
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
int updateQR(cublasHandle_t cHandle, CSC* A, CSC* d_A, float** d_PointerQ, float** d_PointerR, int** d_PointerI, int** d_PointerJ, int** d_PointerSortedJ, int** d_PointerITilde, int** d_PointerJTilde, int** d_PointerIUnion, int** d_PointerJUnion, int* d_n1, int* d_n2, int* d_n1Tilde, int* d_n2Tilde, int* d_n1Union, int* d_n2Union, float** d_PointerMHat_k, float* d_residualNorm, int maxn1, int maxn2, int maxn1Tilde, int maxn2Tilde, int maxn1Union, int maxn2Union, int i, int batchsize) {
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

    // create permutation matrices Pc (we dont need Pr, since we never permute the rows)
    float* d_Pc;
    float** d_PointerPc;

    gpuAssert(
        cudaMalloc((void**) &d_Pc,  maxn2 * maxn2 * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerPc, batchsize * sizeof(float*)));
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerPc, d_Pc, batchsize, maxn2 * maxn2);

    createPermutationMatrices(NULL, d_PointerPc, d_PointerIUnion, d_PointerJUnion, d_n1Union, d_n2Union, maxn1Union, maxn2Union, batchsize);

    // 13.2) ABreve of size n1 x n2Tilde = Q^T * AIJTilde
    float* d_ABreve;
    float** d_PointerABreve;

    gpuAssert(
        cudaMalloc((void**) &d_ABreve,  maxn1 * maxn2Tilde * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerABreve, batchsize * sizeof(float*)));
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerABreve, d_ABreve, batchsize, maxn1 * maxn2Tilde);

    numBlocks = (batchsize * maxn1 * maxn2Tilde + BLOCKSIZE - 1) / BLOCKSIZE;
    computeABreve<<<numBlocks, BLOCKSIZE>>>(d_PointerQ, d_PointerAIJTilde, d_PointerABreve, d_n1, d_n2Tilde, maxn1, maxn2Tilde, batchsize);

    // 13.3) Set B1 = ABreve[0:n2, 0:n2Tilde]
    float* d_B1;
    float** d_PointerB1;

    gpuAssert(
        cudaMalloc((void**) &d_B1,  maxn2 * maxn2Tilde * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerB1, batchsize * sizeof(float*)));
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerB1, d_B1, batchsize, maxn2 * maxn2Tilde);

    numBlocks = (batchsize * maxn2 * maxn2Tilde + BLOCKSIZE - 1) / BLOCKSIZE;
    setB1<<<numBlocks, BLOCKSIZE>>>(d_PointerABreve, d_PointerB1, d_n2, d_n2Tilde, maxn2, maxn2Tilde, batchsize);

    
    printf("done with updateQR\n");

    return 0;
}

#endif