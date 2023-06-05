#ifndef KERNEL_TESTS_CU_H
#define KERNEL_TESTS_CU_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "csc.cu.h"
#include "constants.cu.h"
#include "SPAIkernels.cu.h"
#include "helperKernels.cu.h"

void seqMatrixMultiplication(float *A, float *B, float *C, int dim1, int dim2, int dim3, int batchsize) {
    for (int b = 0; b < batchsize; b++) {
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim3; j++) {
                float sum = 0;
                for (int k = 0; k < dim2; k++) {
                    sum += A[b * dim1 * dim2 + i * dim2 + k] * B[b * dim2 * dim3 + k * dim3 + j];
                }
                C[b * dim1 * dim3 + i * dim3 + j] = sum;
            }
        }
    }
}

int matrixMultiplicationTest(float* A, float* B, float* C, int dim1, int dim2, int dim3, int batchsize) {
    
    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    { // timing the GPU implementations
        gettimeofday(&t_start, NULL);

        for(int i=0; i<RUNS_CPU; i++) {
            seqMatrixMultiplication(A, B, C, dim1, dim2, dim3, batchsize);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
        printf("\n\nSequential matrixMul runs in: %lu microsecs\n\n\n"
              , elapsed, gigaBytesPerSec);
    }

    float* d_A;
    float* d_B;
    float* d_C;

    float** d_PointerA;
    float** d_PointerB;
    float** d_PointerC;

    cudaMalloc((void**)&d_A, batchsize * dim1 * dim2 * sizeof(float));
    cudaMalloc((void**)&d_B, batchsize * dim2 * dim3 * sizeof(float));
    cudaMalloc((void**)&d_C, batchsize * dim1 * dim3 * sizeof(float));

    cudaMemcpy(d_A, A, batchsize * dim1 * dim2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, batchsize * dim2 * dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, batchsize * dim1 * dim3 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_PointerA, batchsize * sizeof(float*));
    cudaMalloc((void**)&d_PointerB, batchsize * sizeof(float*));
    cudaMalloc((void**)&d_PointerC, batchsize * sizeof(float*));

    int numBlocks = (batchsize - BLOCKSIZE + 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerA, d_A, batchsize, dim1 * dim2);
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerB, d_B, batchsize, dim2 * dim3);
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerC, d_C, batchsize, dim1 * dim3);

        { // timing the GPU implementations
        gettimeofday(&t_start, NULL);

        for(int i=0; i<RUNS_GPU; i++) {
            int numBlocks = (batchsize * dim1 * dim3 + BLOCKSIZE - 1) / BLOCKSIZE;
            matrixMultiplication<<<numBlocks, BLOCKSIZE>>>(d_PointerA, d_PointerB, d_PointerC, NULL, NULL, NULL, dim1, dim2, dim3, batchsize);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        gigaBytesPerSec = 2 * dim1 * dim2 * dim3 * sizeof(int) * 1.0e-3f / elapsed;
        printf("\n\nParallel matrixMul runs in: %lu microsecs, GB/sec: %.2f\n\n\n"
              , elapsed, gigaBytesPerSec);
    }

    // gpuAssert( cudaPeekAtLastError() );
    return 0;
}

int runMatrixMultiplicationTest() {

    int dim1 = 10000;
    int dim2 = 10000;
    int dim3 = 10000;
    float sparsity = 1.0;

    CSC* cscA = createRandomCSC(dim1, dim2, sparsity);
    CSC* cscB = createRandomCSC(dim2, dim3, sparsity);

    int* AI = (int*) malloc(sizeof(int) * dim1);
    int* AJ = (int*) malloc(sizeof(int) * dim2);
    for (int i = 0; i < dim1; i++) {
        AI[i] = i;
    }
    for (int i = 0; i < dim2; i++) {
        AJ[i] = i;
    }

    int* BI = (int*) malloc(sizeof(int) * dim2);
    int* BJ = (int*) malloc(sizeof(int) * dim3);
    for (int i = 0; i < dim2; i++) {
        BI[i] = i;
    }
    for (int i = 0; i < dim3; i++) {
        BJ[i] = i;
    }

    float* A = CSCToDense(cscA, AI, AJ, dim1, dim2);
    float* B = CSCToDense(cscB, BI, BJ, dim2, dim3);
    float* C = (float*) malloc(sizeof(float) * dim1 * dim3);

    int batchsize = 1;

    matrixMultiplicationTest(A, B, C, dim1, dim2, dim3, batchsize);
}

# endif
