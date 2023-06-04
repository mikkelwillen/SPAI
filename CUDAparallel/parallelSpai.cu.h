#ifndef SPAI_H
#define SPAI_H

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "csc.cu.h"
#include "constants.cu.h"
#include "qrBatched.cu.h"
#include "invBatched.cu.h"
#include "updateQR.cu.h"
#include "LSProblem.cu.h"
#include "helperKernels.cu.h"
#include "SPAIkernels.cu.h"
#include "singular.cu.h"


// A            = matrix we want to compute SPAI on
// m, n         = size of array
// tolerance    = tolerance
// maxIteration = constraint for the maximal number of iterations
// s            = number of rho_j - the most profitable indices
// batchsize    = number of matrices to be processed in parallel
CSC* parallelSpai(CSC* A, float tolerance, int maxIterations, int s, const int batchsize) {
    printf("---------PARALLEL SPAI---------\n");
    printf("running with parameters: tolerance: %f, maxIterations: %d, s: %d, batchsize: %d\n", tolerance, maxIterations, s, batchsize);
    printCSC(A);

    // check if matrix is singular
    int checkSingular = checkSingularity(A);
    if (checkSingular == 1) {
        printf("Matrix is singular\n");

        return NULL;
    }

    // initialize cublas
    cublasHandle_t cHandle;
    cublasStatus_t stat;
    stat = cublasCreate(&cHandle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        printf("CUBLAS error: %d\n", stat);

        return NULL;
    }

    int numBlocks;
    // initialize M and set to diagonal
    CSC* M = createDiagonalCSC(A->m, A->n);
    printf("after m\n");

    // make A dense and copy to device
    int* I = (int*) malloc(A->m * sizeof(int));
    int* J = (int*) malloc(A->n * sizeof(int));
    for (int i = 0; i < A->m; i++) {
        I[i] = i;
    }
    for (int i = 0; i < A->n; i++) {
        J[i] = i;
    }
    float* h_ADense = CSCToDense(A, I, J, A->m, A->n);
    float* d_ADense;
    gpuAssert(
        cudaMalloc((void**) &d_ADense, A->m * A->n * sizeof(float)));
    gpuAssert(
        cudaMemcpy(d_ADense, h_ADense, A->m * A->n * sizeof(float), cudaMemcpyHostToDevice));
    
    free(I);
    free(J);
    // free(h_ADense);

    // copy A to device
    CSC* d_A = copyCSCFromHostToDevice(A);
    printf("after d_A\n");
    CSC* d_M = copyCSCFromHostToDevice(M);
    printf("after d_M\n");
    
    // compute the number of batches
    int numberOfBatches = (A->n + batchsize - 1) / batchsize;

    for (int i = 0; i < numberOfBatches; i++) {
        printf("---------BATCH: %d---------\n", i);
        int iteration = 0;
        
        int* d_n1;
        int* d_n2;
        
        int** d_PointerI;
        int** d_PointerJ;
        int** d_PointerSortedJ;

        // malloc space
        gpuAssert(
            cudaMalloc((void**) &d_PointerI, batchsize * sizeof(int*)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerJ, batchsize * sizeof(int*)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerSortedJ, batchsize * sizeof(int*)));
        gpuAssert(
            cudaMalloc((void**) &d_n1, batchsize * sizeof(int)));
        gpuAssert(
            cudaMalloc((void**) &d_n2, batchsize * sizeof(int)));
        
        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        computeIandJ<<<numBlocks, BLOCKSIZE>>>(d_A, d_M, d_PointerI, d_PointerJ, d_PointerSortedJ, d_n1, d_n2, i, batchsize, A->n);

        // find the max value of n1 and n2
        int* h_n1 = (int*) malloc(batchsize * sizeof(float));
        int* h_n2 = (int*) malloc(batchsize * sizeof(float));

        gpuAssert(
            cudaMemcpy(h_n1, d_n1, batchsize * sizeof(float), cudaMemcpyDeviceToHost));
        gpuAssert(
            cudaMemcpy(h_n2, d_n2, batchsize * sizeof(float), cudaMemcpyDeviceToHost));

        int maxn1 = 0;
        int maxn2 = 0;
        for (int j = 0; j < batchsize; j++) {
            if (h_n1[j] > maxn1) {
                maxn1 = h_n1[j];
            }
            if (h_n2[j] > maxn2) {
                maxn2 = h_n2[j];
            }
        }

        // free space
        // free(h_n1);
        // free(h_n2);

        // create d_AHat
        float* d_AHat;
        float** d_PointerAHat;

        gpuAssert(
            cudaMalloc((void**) &d_AHat, batchsize * maxn1 * maxn2 * sizeof(float)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerAHat, batchsize * sizeof(float*)));

        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerAHat, d_AHat, batchsize, maxn1 * maxn2);

        numBlocks = (batchsize * maxn1 * maxn2 + BLOCKSIZE - 1) / BLOCKSIZE;
        setMatrixZero<<<numBlocks, BLOCKSIZE>>>(d_PointerAHat, maxn1, maxn2, batchsize);

        numBlocks = (batchsize * maxn1 * maxn2 * A->m + BLOCKSIZE - 1) / BLOCKSIZE;
        CSCToBatchedDenseMatrices<<<numBlocks, BLOCKSIZE>>>(d_A, d_PointerAHat, d_PointerI, d_PointerJ, d_n1, d_n2, maxn1, maxn2, A->m, batchsize);
        

        // print AHat
        float* h_AHat = (float*) malloc(batchsize * maxn1 * maxn2 * sizeof(float));
        gpuAssert(
            cudaMemcpy(h_AHat, d_AHat, batchsize * maxn1 * maxn2 * sizeof(float), cudaMemcpyDeviceToHost));

        // printf("--printing h_AHat--\n");
        // for (int b = 0; b < batchsize; b++) {
        //     printf("b: %d\n", b);
        //     for (int j = 0; j < maxn1; j++) {
        //         for (int k = 0; k < maxn2; k++) {
        //             printf("%f ", h_AHat[b * maxn1 * maxn2 + j * maxn2 + k]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }
        printf("maxn1: %d\n", maxn1);
        printf("maxn2: %d\n", maxn2);
        // initialize d_Q and d_R
        float* d_Q;
        float* d_R;
        float** d_PointerQ;
        float** d_PointerR;

        gpuAssert(
            cudaMalloc((void**) &d_Q, batchsize * maxn1 * maxn1 * sizeof(float)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerQ, batchsize * sizeof(float*)));
        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerQ, d_Q, batchsize, maxn1 * maxn1);


        gpuAssert(
            cudaMalloc((void**) &d_R, batchsize * maxn1 * maxn2 * sizeof(float)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerR, batchsize * sizeof(float*)));
        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerR, d_R, batchsize, maxn1 * maxn2);

        int qrSuccess = qrBatched(cHandle, d_PointerAHat, d_PointerQ, d_PointerR, maxn1, maxn2, batchsize);

        if (qrSuccess != 0) {
            printf("QR failed\n");
            
            return NULL;
        }

        // // print Q
        // float* h_Q = (float*) malloc(batchsize * maxn1 * maxn1 * sizeof(float));
        // gpuAssert(
        //     cudaMemcpy(h_Q, d_Q, batchsize * maxn1 * maxn1 * sizeof(float), cudaMemcpyDeviceToHost));
        // printf("--printing h_Q--\n");
        // for (int b = 0; b < batchsize; b++) {
        //     printf("b: %d\n", b);
        //     for (int j = 0; j < maxn1; j++) {
        //         for (int k = 0; k < maxn1; k++) {
        //             printf("%f ", h_Q[b * maxn1 * maxn1 + j * maxn1 + k]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        // overwrite AHat, since qr overwrote it previously
        numBlocks = (batchsize * maxn1 * maxn2 + BLOCKSIZE - 1) / BLOCKSIZE;
        setMatrixZero<<<numBlocks, BLOCKSIZE>>>(d_PointerAHat, maxn1, maxn2, batchsize);

        numBlocks = (batchsize * maxn1 * maxn2 * A->m + BLOCKSIZE - 1) / BLOCKSIZE;
        CSCToBatchedDenseMatrices<<<numBlocks, BLOCKSIZE>>>(d_A, d_PointerAHat, d_PointerI, d_PointerJ, d_n1, d_n2, maxn1, maxn2, A->m, batchsize);


        // initialize mHat_k, residual, residualNorm
        float* d_mHat_k;
        float** d_PointerMHat_k;

        float* d_residual;
        float** d_PointerResidual;

        float* d_residualNorm;

        gpuAssert(
            cudaMalloc((void**) &d_mHat_k, batchsize * maxn2 * sizeof(float)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerMHat_k, batchsize * sizeof(float*)));
        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerMHat_k, d_mHat_k, batchsize, maxn2);

        gpuAssert(
            cudaMalloc((void**) &d_residual, batchsize * A->m * sizeof(float)));
        gpuAssert(
            cudaMalloc((void**) &d_PointerResidual, batchsize * sizeof(float*)));
        floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerResidual, d_residual, batchsize, A->m);

        gpuAssert(
            cudaMalloc((void**) &d_residualNorm, batchsize * sizeof(float)));

        LSProblem(cHandle, d_A, A, d_ADense, d_PointerQ, d_PointerR, d_PointerMHat_k, NULL, d_PointerResidual, d_PointerI, d_PointerJ, d_n1, d_n2, maxn1, maxn2, i, d_residualNorm, batchsize);

        // print residual
        float* h_residual = (float*) malloc(batchsize * A->m * sizeof(float));
        // gpuAssert(
        //     cudaMemcpy(h_residual, d_residual, batchsize * A->m * sizeof(float), cudaMemcpyDeviceToHost));
        // printf("--printing h_residual--\n");
        // for (int b = 0; b < batchsize; b++) {
        //     printf("b: %d\n", b);
        //     for (int j = 0; j < A->m; j++) {
        //         printf("%f ", h_residual[b * A->m + j]);
        //     }
        //     printf("\n");
        // }
        // free(h_residual);
        
        // check if the tolerance is met
        int toleranceNotMet = 0;
        float* h_residualNorm = (float*) malloc(batchsize * sizeof(float));
        gpuAssert(
            cudaMemcpy(h_residualNorm, d_residualNorm, batchsize * sizeof(float), cudaMemcpyDeviceToHost));
        
        // printf("--printing h_residualNorm--\n");
        // for (int b = 0; b < batchsize; b++) {
        //     printf("%f ", h_residualNorm[b]);
        // }
        
        for (int b = 0; b < batchsize; b++) {
            if (h_residualNorm[b] > tolerance) {
                toleranceNotMet = 1;
            }
        }

        // free(h_residualNorm);
        printf("toleranceNotMet: %d\n", toleranceNotMet);

        // while the tolerance is not met, continue the loop
        while (toleranceNotMet && maxIterations > iteration) {
            printf("\n-------Iteration: %d-------\n", iteration);
            iteration++;
        
            // compute the length of L and set L
            int* d_l;
            int** d_PointerL;

            gpuAssert(
                cudaMalloc((void**) &d_l, batchsize * sizeof(int)));
            gpuAssert(
                cudaMalloc((void**) &d_PointerL, batchsize * sizeof(int*)));

            numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            computeLengthOfL<<< numBlocks, BLOCKSIZE >>>(d_l, d_PointerResidual, d_PointerI, d_PointerL, A->m, d_n1, i, batchsize);
            printf("hallo1\n");

            // check what indeces to keep
            int* d_KeepArray;
            int** d_PointerKeepArray;

            gpuAssert(
                cudaMalloc((void**) &d_KeepArray, batchsize * A->n * sizeof(int)));
            gpuAssert(
                cudaMalloc((void**) &d_PointerKeepArray, batchsize * sizeof(int*)));
            
            numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            intDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerKeepArray, d_KeepArray, batchsize, A->n);

            numBlocks = (batchsize * A->n + BLOCKSIZE - 1) / BLOCKSIZE;
            computeKeepArray<<<numBlocks, BLOCKSIZE>>>(d_A, d_PointerKeepArray, d_PointerL, d_PointerJ, d_n2, d_l, batchsize);

            int* d_n2Tilde;
            gpuAssert(
                cudaMalloc((void**) &d_n2Tilde, batchsize * sizeof(int)));
            
            numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            computeN2Tilde<<<numBlocks, BLOCKSIZE>>>(d_PointerKeepArray, d_n2Tilde, A->n, batchsize);

            printf("hallo2\n");
            // find the max value of n2Tilde
            int maxn2Tilde = 0;
            int* h_n2Tilde = (int*) malloc(batchsize * sizeof(int));
            gpuAssert(
                cudaMemcpy(h_n2Tilde, d_n2Tilde, batchsize * sizeof(int), cudaMemcpyDeviceToHost));
            
            for (int b = 0; b < batchsize; b++) {
                if (h_n2Tilde[b] > maxn2Tilde) {
                    maxn2Tilde = h_n2Tilde[b];
                }
            }

            // fill JTilde
            int** d_PointerJTilde;

            gpuAssert(
                cudaMalloc((void**) &d_PointerJTilde, batchsize * sizeof(int*)));
            
            numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            computeJTilde<<<numBlocks, BLOCKSIZE>>>(d_PointerKeepArray, d_PointerJTilde, d_n2Tilde, A->n, batchsize);

            // compute rhoSquared
            float* d_rhoSquared;
            float** d_PointerRhoSquared;

            gpuAssert(
                cudaMalloc((void**) &d_rhoSquared, batchsize * maxn2Tilde * sizeof(float)));
            gpuAssert(
                cudaMalloc((void**) &d_PointerRhoSquared, batchsize * sizeof(float*)));
            printf("hallo3\n");
            numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerRhoSquared, d_rhoSquared, batchsize, maxn2Tilde);
            
            numBlocks = (batchsize * maxn2Tilde + BLOCKSIZE - 1) / BLOCKSIZE;
            computeRhoSquared<<<numBlocks, BLOCKSIZE>>>(d_A, d_PointerRhoSquared, d_PointerResidual, d_PointerJTilde, d_residualNorm, d_n2Tilde, maxn2Tilde, batchsize);

            // find the smallest s values of rhoSquared
            int* d_newN2Tilde;
            int* d_smallestIndices;
            int** d_PointerSmallestIndices;

            int* d_smallestJTilde;
            int** d_PointerSmallestJTilde;

            gpuAssert(
                cudaMalloc((void**) &d_newN2Tilde, batchsize * sizeof(int)));
            gpuAssert(
                cudaMalloc((void**) &d_smallestIndices, batchsize * s * sizeof(int)));
            gpuAssert(
                cudaMalloc((void**) &d_PointerSmallestIndices, batchsize * sizeof(int*)));

            numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            intDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerSmallestIndices, d_smallestIndices, batchsize, s);

            gpuAssert(
                cudaMalloc((void**) &d_smallestJTilde, batchsize * s * sizeof(int)));
            gpuAssert(
                cudaMalloc((void**) &d_PointerSmallestJTilde, batchsize * sizeof(int*)));
            printf("hallo4\n");
            numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            intDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerSmallestJTilde, d_smallestJTilde, batchsize, s);
        
            numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            computeSmallestIndices<<<numBlocks, BLOCKSIZE>>>(d_PointerRhoSquared, d_PointerSmallestIndices, d_PointerSmallestJTilde, d_PointerJTilde, d_newN2Tilde, d_n2Tilde, s, batchsize);

            // find maxn2Tilde after finding the smallest s values
            maxn2Tilde = 0;
            gpuAssert(
                cudaMemcpy(h_n2Tilde, d_newN2Tilde, batchsize * sizeof(int), cudaMemcpyDeviceToHost));
            for (int b = 0; b < batchsize; b++) {
                if (h_n2Tilde[b] > maxn2Tilde) {
                    maxn2Tilde = h_n2Tilde[b];
                }
            }

            free(h_n2Tilde);
            

            // find ITilde and make IUnion and JUnion
            int* d_n1Tilde;
            int* d_n1Union;
            int* d_n2Union;

            int** d_PointerITilde;
            int** d_PointerIUnion;
            int** d_PointerJUnion;

            gpuAssert(
                cudaMalloc((void**) &d_n1Tilde, batchsize * sizeof(int)));
            gpuAssert(
                cudaMalloc((void**) &d_n1Union, batchsize * sizeof(int)));
            gpuAssert(
                cudaMalloc((void**) &d_n2Union, batchsize * sizeof(int)));
            
            gpuAssert(
                cudaMalloc((void**) &d_PointerITilde, batchsize * sizeof(int*)));
            gpuAssert(
                cudaMalloc((void**) &d_PointerIUnion, batchsize * sizeof(int*)));
            gpuAssert(
                cudaMalloc((void**) &d_PointerJUnion, batchsize * sizeof(int*)));
            printf("hallo5\n");
            numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            computeITilde<<<numBlocks, BLOCKSIZE>>>(d_A, d_PointerI, d_PointerJ, d_PointerITilde, d_PointerSmallestJTilde, d_PointerIUnion, d_PointerJUnion, d_n1, d_n2, d_n1Tilde, d_newN2Tilde, d_n1Union, d_n2Union, batchsize);
            printf("hallo6\n");
            // find maxn1Tilde, maxn1Union and maxn2Union
            int maxn1Tilde = 0;
            int maxn1Union = 0;
            int maxn2Union = 0;

            int* h_n1Tilde = (int*) malloc(batchsize * sizeof(int));
            int* h_n1Union = (int*) malloc(batchsize * sizeof(int));
            int* h_n2Union = (int*) malloc(batchsize * sizeof(int));
            printf("hallo7\n");
            gpuAssert(
                cudaMemcpy(h_n1Tilde, d_n1Tilde, batchsize * sizeof(int), cudaMemcpyDeviceToHost));
            gpuAssert(
                cudaMemcpy(h_n1Union, d_n1Union, batchsize * sizeof(int), cudaMemcpyDeviceToHost));
            gpuAssert(
                cudaMemcpy(h_n2Union, d_n2Union, batchsize * sizeof(int), cudaMemcpyDeviceToHost));
            printf("hallo8\n");
            for (int b = 0; b < batchsize; b++) {
                if (h_n1Tilde[b] > maxn1Tilde) {
                    maxn1Tilde = h_n1Tilde[b];
                }

                if (h_n1Union[b] > maxn1Union) {
                    maxn1Union = h_n1Union[b];
                }
                
                if (h_n2Union[b] > maxn2Union) {
                    maxn2Union = h_n2Union[b];
                }
            }
            printf("hallo6\n");
            // 13) Update the QR factorization of A(IUnion, JUnion) and compute the residual norm
            int updateSuccess = updateQR(cHandle, A, d_A, d_ADense, d_Q, d_R, d_PointerQ, d_PointerR, d_PointerI, d_PointerJ, d_PointerSortedJ, d_PointerITilde, d_PointerSmallestJTilde, d_PointerIUnion, d_PointerJUnion, d_n1, d_n2, d_n1Tilde, d_newN2Tilde, d_n1Union, d_n2Union, d_mHat_k, d_PointerMHat_k, d_PointerResidual, d_residualNorm, maxn1, maxn2, maxn1Tilde, maxn2Tilde, maxn1Union, maxn2Union, i, batchsize);

            if (updateSuccess != 0) {
                printf("updateQR failed\n");

                return NULL;
            }
            printf("updateQR success\n");

            float* h_Q = (float*) malloc(batchsize * maxn1Union * maxn1Union * sizeof(float));
            gpuAssert(
                cudaMemcpy(h_Q, d_Q, batchsize * maxn1Union * maxn1Union * sizeof(float), cudaMemcpyDeviceToHost));
            // printf("--printing h_Q--\n");
            // for (int b = 0; b < batchsize; b++) {
            //     printf("b: %d\n", b);
            //     for (int j = 0; j < maxn1Union; j++) {
            //         for (int k = 0; k < maxn1Union; k++) {
            //             printf("%f ", h_Q[b * maxn1Union * maxn1Union + j * maxn1Union + k]);
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }

            // set I and J to IUnion and JUnion
            numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
            copyIandJ<<<numBlocks, BLOCKSIZE>>>(d_PointerI, d_PointerJ, d_PointerIUnion, d_PointerJUnion, d_n1Union, d_n2Union, batchsize);
            printf("copyIandJ success\n");

            // printf("--I--\n");
            // intPrintPointerArray<<<1, 1>>>(d_PointerI, 1, maxn1Union, batchsize);

            // printf("--J--\n");
            // intPrintPointerArray<<<1, 1>>>(d_PointerJ, 1, maxn2Union, batchsize);

            // printf("--SortedJ--\n");
            // intPrintPointerArray<<<1, 1>>>(d_PointerSortedJ, 1, maxn2Union, batchsize);

            // set n1 and n2 to n1Union and n2Union
            gpuAssert(
                cudaFree(d_n1));
            gpuAssert(
                cudaFree(d_n2));
            printf("cudaFree success\n");

            gpuAssert(
                cudaMalloc((void**) &d_n1, batchsize * sizeof(int)));
            gpuAssert(
                cudaMalloc((void**) &d_n2, batchsize * sizeof(int)));

            gpuAssert(
                cudaMemcpy(d_n1, d_n1Union, batchsize * sizeof(int), cudaMemcpyDeviceToDevice));
            gpuAssert(
                cudaMemcpy(d_n2, d_n2Union, batchsize * sizeof(int), cudaMemcpyDeviceToDevice));
            printf("cudaMemcpy success\n");
            
            // set maxn1 and maxn2 to maxn1Union and maxn2Union
            maxn1 = maxn1Union;
            maxn2 = maxn2Union;


            // int* h_l = (int*) malloc(batchsize * sizeof(int));
            // gpuAssert(
            //     cudaMemcpy(h_l, d_l, batchsize * sizeof(int), cudaMemcpyDeviceToHost));
            // printf("--printing h_l--\n");
            // for (int b = 0; b < batchsize; b++) {
            //     printf("b: %d, l: %d\n", b, h_l[b]);
            // }

            float* h_rhoSquared = (float*) malloc(batchsize * maxn2Tilde * sizeof(float));
            gpuAssert(
                cudaMemcpy(h_rhoSquared, d_rhoSquared, batchsize * maxn2Tilde * sizeof(float), cudaMemcpyDeviceToHost));
            // printf("--printing h_rhoSquared--\n");
            // for (int b = 0; b < batchsize; b++) {
            //     printf("b: %d\n", b);
            //     for (int j = 0; j < maxn2Tilde; j++) {
            //         printf("%f ", h_rhoSquared[b * maxn2Tilde + j]);
            //     }
            //     printf("\n");
            // }
            
            float* h_mHat_k = (float*) malloc(batchsize * maxn2 * sizeof(float));
            gpuAssert(
                cudaMemcpy(h_mHat_k, d_mHat_k, batchsize * maxn2 * sizeof(float), cudaMemcpyDeviceToHost));
            
            // printf("--printing h_mHat_k--\n");
            // for (int b = 0; b < batchsize; b++) {
            //     printf("b: %d\n", b);
            //     for (int j = 0; j < maxn2; j++) {
            //         printf("%f ", h_mHat_k[b * maxn2 + j]);
            //     }
            //     printf("\n");
            // }
            free(h_mHat_k);

            // h_residual = (float*) malloc(batchsize * A->m * sizeof(float));
            gpuAssert(
                cudaMemcpy(h_residual, d_residual, batchsize * A->m * sizeof(float), cudaMemcpyDeviceToHost));
            
            // printf("--printing h_residual--\n");
            // for (int b = 0; b < batchsize; b++) {
            //     printf("b: %d\n", b);
            //     for (int j = 0; j < A->m; j++) {
            //         printf("%f ", h_residual[b * A->m + j]);
            //     }
            //     printf("\n");
            // }
            // free(h_residual);

            // h_residualNorm = (float*) malloc(batchsize * sizeof(float));
            gpuAssert(
                cudaMemcpy(h_residualNorm, d_residualNorm, batchsize * sizeof(float), cudaMemcpyDeviceToHost));
            
            // printf("--printing h_residualNorm--\n");
            // for (int b = 0; b < batchsize; b++) {
            //     printf("%f ", h_residualNorm[b]);
            // }
            // free(h_residualNorm);

            toleranceNotMet = 0;
            for (int b = 0; b < batchsize; b++) {
                if (h_residualNorm[b] > tolerance) {
                    toleranceNotMet = 1;
                }
            }

            // free memory
            freeArraysInPointerArray<<<1, 1>>>(d_PointerITilde, batchsize);
            freeArraysInPointerArray<<<1, 1>>>(d_PointerJTilde, batchsize);
            freeArraysInPointerArray<<<1, 1>>>(d_PointerIUnion, batchsize);
            freeArraysInPointerArray<<<1, 1>>>(d_PointerJUnion, batchsize);

            gpuAssert(
                cudaFree(d_l));
            gpuAssert(
                cudaFree(d_PointerL));
            gpuAssert(
                cudaFree(d_KeepArray));
            gpuAssert(
                cudaFree(d_PointerKeepArray));
            gpuAssert(
                cudaFree(d_n2Tilde));
            gpuAssert(
                cudaFree(d_PointerJTilde));
            
        }

        // free(h_Q);
        // h_Q = (float*) malloc(batchsize * maxn1 * maxn1 * sizeof(float));
        // float* h_R = (float*) malloc(batchsize * maxn1 * maxn2 * sizeof(float));
        // gpuAssert(
        //     cudaMemcpy(h_Q, d_Q, batchsize * maxn1 * maxn1 * sizeof(float), cudaMemcpyDeviceToHost));
        // gpuAssert(
        //     cudaMemcpy(h_R, d_R, batchsize * maxn1 * maxn2 * sizeof(float), cudaMemcpyDeviceToHost));
        
        // printf("--printing h_Q--\n");
        // for (int b = 0; b < batchsize; b++) {
        //     printf("b: %d\n", b);
        //     for (int j = 0; j < maxn1; j++) {
        //         for (int k = 0; k < maxn1; k++) {
        //             printf("%f ", h_Q[b * maxn1 * maxn1 + j * maxn1 + k]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        // printf("--printing h_R--\n");
        // for (int b = 0; b < batchsize; b++) {
        //     printf("b: %d\n", b);
        //     for (int j = 0; j < maxn1; j++) {
        //         for (int k = 0; k < maxn2; k++) {
        //             printf("%f ", h_R[b * maxn1 * maxn2 + j * maxn2 + k]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }

        float* h_mHat_k = (float*) malloc(batchsize * maxn2 * sizeof(float));
        gpuAssert(
            cudaMemcpy(h_mHat_k, d_mHat_k, batchsize * maxn2 * sizeof(float), cudaMemcpyDeviceToHost));

        // printf("--printing h_mHat_k--\n");
        // for (int b = 0; b < batchsize; b++) {
        //     printf("b: %d\n", b);
        //     for (int j = 0; j < maxn2; j++) {
        //         printf("%f ", h_mHat_k[b * maxn2 + j]);
        //     }
        //     printf("\n");
        // }
        free(h_mHat_k);

        h_residual = (float*) malloc(batchsize * A->m * sizeof(float));
        gpuAssert(
            cudaMemcpy(h_residual, d_residual, batchsize * A->m * sizeof(float), cudaMemcpyDeviceToHost));
        
        // printf("--printing h_residual--\n");
        // for (int b = 0; b < batchsize; b++) {
        //     printf("b: %d\n", b);
        //     for (int j = 0; j < A->m; j++) {
        //         printf("%f ", h_residual[b * A->m + j]);
        //     }
        //     printf("\n");
        // }
        free(h_residual);

        // printf("--SortedJ--\n");
        // intPrintPointerArray<<<1, 1>>>(d_PointerSortedJ, 1, maxn2, batchsize);
        
        // printf("maxn2: %d\n", maxn2);

        // printf("--printing mHat_k--\n");
        // printPointerArray<<<1, 1>>>(d_PointerMHat_k, 1, maxn2, batchsize);
        
        updateBatchColumnsCSC <<<1, 1>>> (d_M, d_PointerMHat_k, d_PointerSortedJ, d_n2, maxn2, i, batchsize);


        // free memory
        numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
        freeArraysInPointerArray<<<numBlocks, BLOCKSIZE>>>(d_PointerI, batchsize);
        freeArraysInPointerArray<<<numBlocks, BLOCKSIZE>>>(d_PointerJ, batchsize);
        freeArraysInPointerArray<<<numBlocks, BLOCKSIZE>>>(d_PointerSortedJ, batchsize);
        gpuAssert(
            cudaFree(d_n1));
        gpuAssert(
            cudaFree(d_n2));
        gpuAssert(
            cudaFree(d_PointerI));
        gpuAssert(
            cudaFree(d_PointerJ));
        gpuAssert(
            cudaFree(d_PointerSortedJ));
        gpuAssert(
            cudaFree(d_AHat));
        gpuAssert(
            cudaFree(d_PointerAHat));
        gpuAssert(
            cudaFree(d_Q));
        gpuAssert(
            cudaFree(d_PointerQ));
        gpuAssert(
            cudaFree(d_R));
        gpuAssert(
            cudaFree(d_PointerR));
        gpuAssert(
            cudaFree(d_mHat_k));
        gpuAssert(
            cudaFree(d_PointerMHat_k));
        gpuAssert(
            cudaFree(d_residual));
        gpuAssert(
            cudaFree(d_residualNorm));
        
        
    }

    // copy M back to host
    M = copyCSCFromDeviceToHost(d_M);

    // free memory
    freeDeviceCSC(d_A);
    freeDeviceCSC(d_M);
    gpuAssert(
        cudaFree(d_ADense));
    
    cublasDestroy(cHandle);

    return M;
}

#endif