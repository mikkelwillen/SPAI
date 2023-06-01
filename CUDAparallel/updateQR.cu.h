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
d_mHat_k = batched mHat_k in device memory
d_PointerMHat_k = pointer to mHat_k in device memory
d_PointerResidual = pointer to residual in device memory
d_residualNorm = pointer to residualNorm in device memory
maxn1 = maximum value of n1
maxn2 = maximum value of n2
maxn1Tilde = maximum value of n1Tilde
maxn2Tilde = maximum value of n2Tilde
maxn1Union = maximum value of n1Union
maxn2Union = maximum value of n2Union
i = current iteration
batchsize = batchsize */
int updateQR(cublasHandle_t cHandle, CSC* A, CSC* d_A, float* d_Q, float* d_R, float** d_PointerQ, float** d_PointerR, int** d_PointerI, int** d_PointerJ, int** d_PointerSortedJ, int** d_PointerITilde, int** d_PointerJTilde, int** d_PointerIUnion, int** d_PointerJUnion, int* d_n1, int* d_n2, int* d_n1Tilde, int* d_n2Tilde, int* d_n1Union, int* d_n2Union, float* d_mHat_k, float** d_PointerMHat_k, float** d_PointerResidual, float* d_residualNorm, int maxn1, int maxn2, int maxn1Tilde, int maxn2Tilde, int maxn1Union, int maxn2Union, int i, int batchsize) {
    printf("\n------UPDATE QR------\n");
    int numBlocks;

    // create AIJTilde
    float* d_AIJTilde;
    float** d_PointerAIJTilde;

    gpuAssert(
        cudaMalloc((void**) &d_AIJTilde, batchsize * maxn1 * maxn2Tilde * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerAIJTilde, batchsize * sizeof(float*)));

    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerAIJTilde, d_AIJTilde, batchsize, maxn1 * maxn2Tilde);
    
    numBlocks = (batchsize * maxn1 * maxn2Tilde * A->m + BLOCKSIZE - 1) / BLOCKSIZE;
    CSCToBatchedDenseMatrices<<<numBlocks, BLOCKSIZE>>>(d_A, d_PointerAIJTilde, d_PointerI, d_PointerJTilde, d_n1, d_n2Tilde, maxn1, maxn2Tilde, A->m, batchsize);

    // print AIJTilde
    float* h_AIJTilde = (float*) malloc(batchsize * maxn1 * maxn2Tilde * sizeof(float));
    gpuAssert(
        cudaMemcpy(h_AIJTilde, d_AIJTilde, batchsize * maxn1 * maxn2Tilde * sizeof(float), cudaMemcpyDeviceToHost));
    // printf("AIJTilde:\n");
    // for (int i = 0; i < batchsize; i++) {
    //     printf("i = %d\n", i);
    //     for (int j = 0; j < maxn1; j++) {
    //         for (int k = 0; k < maxn2Tilde; k++) {
    //             printf("%f ", h_AIJTilde[i * maxn1 * maxn2Tilde + j * maxn2Tilde + k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // create AITildeJTilde
    float* d_AITildeJTilde;
    float** d_PointerAITildeJTilde;

    gpuAssert(
        cudaMalloc((void**) &d_AITildeJTilde, batchsize * maxn1Tilde * maxn2Tilde * sizeof(float)));
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
        cudaMalloc((void**) &d_Pc, batchsize * maxn2Union * maxn2Union * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerPc, batchsize * sizeof(float*)));
    printf("--d_PointerPc\n");
    printPointerArray<<<1, 1>>>(d_PointerPc, maxn2Union, maxn2Union, batchsize);
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerPc, d_Pc, batchsize, maxn2Union * maxn2Union);

    createPermutationMatrices(NULL, d_PointerPc, d_PointerIUnion, d_PointerJUnion, d_n1Union, d_n2Union, maxn1Union, maxn2Union, batchsize);

    // 13.2) ABreve of size n1 x n2Tilde = Q^T * AIJTilde
    float* d_ABreve;
    float** d_PointerABreve;

    gpuAssert(
        cudaMalloc((void**) &d_ABreve, batchsize * maxn1 * maxn2Tilde * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerABreve, batchsize * sizeof(float*)));
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerABreve, d_ABreve, batchsize, maxn1 * maxn2Tilde);

    numBlocks = (batchsize * maxn1 * maxn2Tilde + BLOCKSIZE - 1) / BLOCKSIZE;
    computeABreve<<<numBlocks, BLOCKSIZE>>>(d_PointerQ, d_PointerAIJTilde, d_PointerABreve, d_n1, d_n2Tilde, maxn1, maxn2Tilde, batchsize);

    // print ABreve
    float* h_ABreve = (float*) malloc(batchsize * maxn1 * maxn2Tilde * sizeof(float));
    gpuAssert(
        cudaMemcpy(h_ABreve, d_ABreve, batchsize * maxn1 * maxn2Tilde * sizeof(float), cudaMemcpyDeviceToHost));
    printf("\nABreve:\n");
    for (int i = 0; i < batchsize; i++) {
        printf("\nBatch %d:\n", i);
        for (int j = 0; j < maxn1; j++) {
            for (int k = 0; k < maxn2Tilde; k++) {
                printf("%f ", h_ABreve[i * maxn1 * maxn2Tilde + j * maxn2Tilde + k]);
            }
            printf("\n");
        }
    }

    printf("maxn1: %d\n", maxn1);
    printf("maxn2: %d\n", maxn2);
    printf("maxn1Tilde: %d\n", maxn1Tilde);
    printf("maxn2Tilde: %d\n", maxn2Tilde);
    printf("maxn1Union: %d\n", maxn1Union);
    printf("maxn2Union: %d\n", maxn2Union);

    // 13.3) Set B1 = ABreve[0:n2, 0:n2Tilde]
    float* d_B1;
    float** d_PointerB1;

    gpuAssert(
        cudaMalloc((void**) &d_B1, batchsize * maxn2 * maxn2Tilde * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerB1, batchsize * sizeof(float*)));
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerB1, d_B1, batchsize, maxn2 * maxn2Tilde);

    numBlocks = (batchsize * maxn2 * maxn2Tilde + BLOCKSIZE - 1) / BLOCKSIZE;
    setB1<<<numBlocks, BLOCKSIZE>>>(d_PointerABreve, d_PointerB1, d_n2, d_n2Tilde, maxn2, maxn2Tilde, batchsize);

    // print B1
    float* h_B1 = (float*) malloc(batchsize * maxn2 * maxn2Tilde * sizeof(float));
    gpuAssert(
        cudaMemcpy(h_B1, d_B1, batchsize * maxn2 * maxn2Tilde * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\nB1:\n");
    for (int i = 0; i < batchsize; i++) {
        printf("\nBatch %d:\n", i);
        for (int j = 0; j < maxn2; j++) {
            for (int k = 0; k < maxn2Tilde; k++) {
                printf("%f ", h_B1[i * maxn2 * maxn2Tilde + j * maxn2Tilde + k]);
            }
            printf("\n");
        }
    }

    // 13.4) Set B2 = ABreve[n2 + 1:n1, 0:n2Tilde] + A(ITilde, JTilde)
    float* d_B2;
    float** d_PointerB2;

    gpuAssert(
        cudaMalloc((void**) &d_B2, batchsize * maxn1Union * maxn2Tilde * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerB2, batchsize * sizeof(float*)));
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerB2, d_B2, batchsize, maxn1Union * maxn2Tilde);

    numBlocks = (batchsize * maxn1Union * maxn2Tilde + BLOCKSIZE - 1) / BLOCKSIZE;
    setB2<<<numBlocks, BLOCKSIZE>>>(d_PointerABreve, d_PointerAITildeJTilde, d_PointerB2, d_n1, d_n1Union, d_n2, d_n2Tilde, maxn1Union, maxn2Tilde, batchsize);

    float* h_B2 = (float*) malloc(batchsize * maxn1Union * maxn2Tilde * sizeof(float));
    gpuAssert(
        cudaMemcpy(h_B2, d_B2, batchsize * maxn1Union * maxn2Tilde * sizeof(float), cudaMemcpyDeviceToHost));
    printf("B2:\n");
    for (int b = 0; b < batchsize; b++) {
        printf("b = %d\n", b);
        for (int i = 0; i < maxn1Union; i++) {
            for (int j = 0; j < maxn2Tilde; j++) {
                printf("%f ", h_B2[b * maxn1Union * maxn2Tilde + i * maxn2Tilde + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // 13.5) Do QR factorization of B2
    float* d_B2Q;
    float* d_B2R;

    float** d_PointerB2Q;
    float** d_PointerB2R;

    gpuAssert(
        cudaMalloc((void**) &d_B2Q,  batchsize * maxn1Union * maxn1Union * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_B2R,  batchsize * maxn1Union * maxn2Tilde * sizeof(float)));

    gpuAssert(
        cudaMalloc((void**) &d_PointerB2Q, batchsize * sizeof(float*)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerB2R, batchsize * sizeof(float*)));

    printf("maxn1 = %d\n", maxn1);
    printf("maxn2 = %d\n", maxn2);
    printf("maxn1Tilde = %d\n", maxn1Tilde);
    printf("maxn2Tilde = %d\n", maxn2Tilde);
    printf("maxn1Union = %d\n", maxn1Union);
    printf("maxn2Union = %d\n", maxn2Union);
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerB2Q, d_B2Q, batchsize, maxn1Union * maxn1Union);

    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerB2R, d_B2R, batchsize, maxn1Union * maxn2Tilde);

    int qrSuccess = qrBatched(cHandle, d_PointerB2, d_PointerB2Q, d_PointerB2R, maxn1Union, maxn2Tilde, batchsize);

    if (qrSuccess != 0) {
        printf("QR failed\n");
        
        return 1;
    }

    // 13.6) compute QB and RB from algorithm 17
    // make frist matrix with Q in the upper left corner and identity in the lower right corner of size n1Union x n1Union
    float* d_firstMatrix;
    float** d_PointerFirstMatrix;

    gpuAssert(
        cudaMalloc((void**) &d_firstMatrix, batchsize * maxn1Union * maxn1Union * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerFirstMatrix, batchsize * sizeof(float*)));
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerFirstMatrix, d_firstMatrix, batchsize, maxn1Union * maxn1Union);

    numBlocks = (batchsize * maxn1Union * maxn1Union + BLOCKSIZE - 1) / BLOCKSIZE;
    setFirstMatrix<<<numBlocks, BLOCKSIZE>>>(d_PointerFirstMatrix, d_PointerQ, d_n1, d_n1Union, maxn1Union, batchsize);

    // make second matrix with identity in the upper left corner and QB in the lower right corner of size n1Union x n1Union");
    float* d_secondMatrix;
    float** d_PointerSecondMatrix;

    gpuAssert(
        cudaMalloc((void**) &d_secondMatrix, batchsize * maxn1Union * maxn1Union * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerSecondMatrix, batchsize * sizeof(float*)));
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerSecondMatrix, d_secondMatrix, batchsize, maxn1Union * maxn1Union);

    numBlocks = (batchsize * maxn1Union * maxn1Union + BLOCKSIZE - 1) / BLOCKSIZE;
    setSecondMatrix<<<numBlocks, BLOCKSIZE>>>(d_PointerSecondMatrix, d_PointerB2Q, d_n1Tilde, d_n1Union, d_n2, maxn1Union, batchsize);

    // compute unsortedQ = firstMatrix * secondMatrix
    float* d_unsortedQ;
    float** d_PointerUnsortedQ;

    gpuAssert(
        cudaMalloc((void**) &d_unsortedQ, batchsize * maxn1Union * maxn1Union * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerUnsortedQ, batchsize * sizeof(float*)));
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerUnsortedQ, d_unsortedQ, batchsize, maxn1Union * maxn1Union);

    numBlocks = (batchsize * maxn1Union * maxn1Union + BLOCKSIZE - 1) / BLOCKSIZE;
    matrixMultiplication<<<numBlocks, BLOCKSIZE>>>(d_PointerFirstMatrix, d_PointerSecondMatrix, d_PointerUnsortedQ, d_n1Union, d_n1Union, d_n1Union, maxn1Union, maxn1Union, maxn1Union, batchsize);

    // set unsorted R
    float* d_unsortedR;
    float** d_PointerUnsortedR;

    gpuAssert(
        cudaMalloc((void**) &d_unsortedR, batchsize * maxn1Union * maxn2Union * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_PointerUnsortedR, batchsize * sizeof(float*)));
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerUnsortedR, d_unsortedR, batchsize, maxn1Union * maxn2Union);

    numBlocks = (batchsize * maxn1Union * maxn2Union + BLOCKSIZE - 1) / BLOCKSIZE;
    setUnsortedR<<<numBlocks, BLOCKSIZE>>>(d_PointerUnsortedR, d_PointerR, d_PointerB1, d_PointerB2R, d_n1, d_n1Union, d_n2, d_n2Union, d_n2Tilde, maxn1Union, maxn2Union, batchsize);

    // print R
    float* h_unsortedR = (float*) malloc(batchsize * maxn1Union * maxn2Union * sizeof(float));
    gpuAssert(
        cudaMemcpy(h_unsortedR, d_unsortedR, batchsize * maxn1Union * maxn2Union * sizeof(float), cudaMemcpyDeviceToHost));
    // printf("R:\n");
    // for (int i = 0; i < batchsize; i++) {
    //     printf("batch = %d\n", i);
    //     for (int j = 0; j < maxn1Union; j++) {
    //         for (int k = 0; k < maxn2Union; k++) {
    //             printf("%f ", h_unsortedR[i * maxn1Union * maxn2Union + j * maxn2Union + k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // free and malloc space for new mHat_k
    gpuAssert(
        cudaFree(d_mHat_k));
    gpuAssert(
        cudaMalloc((void**) &d_mHat_k, batchsize * maxn2Union * sizeof(float)));
    
    gpuAssert(
        cudaFree(d_PointerMHat_k));
    gpuAssert(
        cudaMalloc((void**) &d_PointerMHat_k, batchsize * sizeof(float*)));
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerMHat_k, d_mHat_k, batchsize, maxn2Union);

    // 13.7) compute the new solution m_k with the least squares problem
    int lsSuccess = LSProblem(cHandle, d_A, A, d_PointerUnsortedQ, d_PointerUnsortedR, d_PointerMHat_k, d_PointerPc, d_PointerResidual, d_PointerIUnion, d_PointerJUnion, d_n1Union, d_n2Union, maxn1Union, maxn2Union, i, d_residualNorm, batchsize);

    if (lsSuccess != 0) {
        printf("LSProblem failed\n");
        
        return 1;
    }

    // permute J and store it in sortedJ
    int* sortedJ;
    gpuAssert(
        cudaMalloc((void**) &sortedJ, batchsize * maxn2Union * sizeof(int)));
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    intDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerSortedJ, sortedJ, batchsize, maxn2Union);
    printf("j sorted\n");
    numBlocks = (batchsize * maxn2Union + BLOCKSIZE - 1) / BLOCKSIZE;
    permuteJ<<<numBlocks, BLOCKSIZE>>>(d_PointerSortedJ, d_PointerJUnion, d_PointerPc, d_n2Union, maxn2Union, batchsize);

    // set Q and R to unsortedQ and unsortedR
    gpuAssert(
        cudaFree(d_Q));
    gpuAssert(
        cudaFree(d_R));
    printf("Q and R freed\n");

    gpuAssert(
        cudaMalloc((void**) &d_Q, batchsize * maxn1Union * maxn1Union * sizeof(float)));
    gpuAssert(
        cudaMalloc((void**) &d_R, batchsize * maxn1Union * maxn2Union * sizeof(float)));
    
    gpuAssert(
        cudaMemcpy(d_Q, d_unsortedQ, batchsize * maxn1Union * maxn1Union * sizeof(float), cudaMemcpyDeviceToDevice));
    gpuAssert(
        cudaMemcpy(d_R, d_unsortedR, batchsize * maxn1Union * maxn2Union * sizeof(float), cudaMemcpyDeviceToDevice));
    printf("Q and R set\n");
    
    numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerQ, d_Q, batchsize, maxn1Union * maxn1Union);
    printf("Q pointer set\n");

    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerR, d_R, batchsize, maxn1Union * maxn2Union);
    printf("R pointer set\n");

    // free memory
    gpuAssert(
        cudaFree(d_AIJTilde));
    gpuAssert(
        cudaFree(d_PointerAIJTilde));
    gpuAssert(
        cudaFree(d_AITildeJTilde));
    gpuAssert(
        cudaFree(d_PointerAITildeJTilde));
    gpuAssert(
        cudaFree(d_ABreve));
    gpuAssert(
        cudaFree(d_PointerABreve));
    gpuAssert(
        cudaFree(d_B1));
    gpuAssert(
        cudaFree(d_PointerB1));
    gpuAssert(
        cudaFree(d_B2));
    gpuAssert(
        cudaFree(d_PointerB2));
    gpuAssert(
        cudaFree(d_B2Q));
    gpuAssert(
        cudaFree(d_PointerB2Q));
    gpuAssert(
        cudaFree(d_B2R));
    gpuAssert(
        cudaFree(d_PointerB2R));
    gpuAssert(
        cudaFree(d_firstMatrix));
    gpuAssert(
        cudaFree(d_secondMatrix));
    gpuAssert(
        cudaFree(d_unsortedQ));
    gpuAssert(
        cudaFree(d_unsortedR));
    
    return 0;
}

#endif