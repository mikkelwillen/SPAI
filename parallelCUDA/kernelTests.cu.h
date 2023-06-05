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

/* kernel for creating a random dense matrix */
__global__ void createRandomMatrix(float* d_M, int m, int n, float sparsity) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < m * n) {
        d_M[tid] = (float) (((tid + 1 )% 2) * (tid % 3) + (tid % 4) * (tid % 5));
    }
}

void seqMatrixMultiplication(float* A, float* B, float* C, int dim1, int dim2, int dim3, int batchsize) {
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

void seqSetSecondMatrix(float* A, float* B, unsigned long int dim1, unsigned long int dim2, int batchsize) {
    printf("bib\n");
    for (int b = 0; b < batchsize; b++) {
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim1; j++) {
                if (i < dim2 && j < dim2) {
                    // printf("b, i, j = %d, %d, %d\n", b, i, j);
                    A[b * dim1 * dim1 + i * dim1 + j] = B[b * dim2 * dim2 + i * dim2 + j];
                } else if (i == j) {
                    A[b * dim1 * dim1 + i * dim1 + j] = 1.0;
                } else {
                    A[b * dim1 * dim1 + i * dim1 + j] = 0.0;
                }
            }
        }
    }
}

void seqCSCToDense(CSC* csc, float* dense, int* I, int* J, int n1, int n2, int batchsize) {

    for (int b = 0; b < batchsize; b++) {
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int l = csc->offset[J[b * batchsize + j]]; l < csc->offset[J[b * batchsize + j] + 1]; l++) {
                    if (I[b * batchsize + i] == csc->flatRowIndex[l]) {
                        dense[b * n1 * n2 + i * n2 + j] = csc->flatData[l];
                    }
                }
            }
        }
    }
}

int matrixMultiplicationTest(float* d_A, float* d_B, float* d_C, int dim1, int dim2, int dim3, int batchsize) {
    printf("matrixMultiplicationTest - dim1: %d, dim2: %d, dim3 %d\n", dim1, dim2, dim3);
    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    float* h_A = (float*) malloc(batchsize * dim1 * dim2 * sizeof(float));
    float* h_B = (float*) malloc(batchsize * dim2 * dim3 * sizeof(float));
    float* h_C = (float*) malloc(batchsize * dim1 * dim3 * sizeof(float));

    cudaMemcpy(h_A, d_A, batchsize * dim1 * dim2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, batchsize * dim2 * dim3 * sizeof(float), cudaMemcpyDeviceToHost);

    { // timing the CPU implementations
        gettimeofday(&t_start, NULL);

        for(int i=0; i<RUNS_CPU; i++) {
            seqMatrixMultiplication(h_A, h_B, h_C, dim1, dim2, dim3, batchsize);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
        printf("\n\nSequential matrixMul runs in: %lu microsecs\n\n\n"
              , elapsed, gigaBytesPerSec);
    }

    float** d_PointerA;
    float** d_PointerB;
    float** d_PointerC;

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
        gigaBytesPerSec = 2 * dim1 * dim2 * dim3 * batchsize * sizeof(int) * 1.0e-3f / elapsed;
        printf("\n\nParallel matrixMul runs in: %lu microsecs, GB/sec: %.2f\n\n\n"
              , elapsed, gigaBytesPerSec);
    }

    // gpuAssert( cudaPeekAtLastError() );
    return 0;
}

int setSecondMatrixTest(float* d_A, float* d_B, unsigned long int dim1, unsigned long int dim2, int batchsize) {
    printf("setSecondMatrixTest - dim1: %d, dim2: %d\n", dim1, dim2);
    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    float* h_A = (float*) malloc(batchsize * dim1 * dim1 * sizeof(float));
    float* h_B = (float*) malloc(batchsize * dim2 * dim2 * sizeof(float));

    gpuAssert(
        cudaMemcpy(h_B, d_B, batchsize * dim2 * dim2 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("memcpy done\n");

    { // timing the CPU implementations
        gettimeofday(&t_start, NULL);

        for(int i=0; i<RUNS_CPU; i++) {
            seqSetSecondMatrix(h_A, h_B, dim1, dim2, batchsize);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
        printf("\n\nSequential secondMatrix runs in: %lu microsecs\n\n\n"
              , elapsed, gigaBytesPerSec);
    }

    float** d_PointerA;
    float** d_PointerB;

    cudaMalloc((void**)&d_PointerA, batchsize * sizeof(float*));
    cudaMalloc((void**)&d_PointerB, batchsize * sizeof(float*));

    int numBlocks = (batchsize + BLOCKSIZE - 1) / BLOCKSIZE;
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerA, d_A, batchsize, dim1 * dim1);
    floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerB, d_B, batchsize, dim2 * dim2);

    int* h_n1Tilde = (int*) malloc(batchsize * sizeof(int));
    int* h_n1Union = (int*) malloc(batchsize * sizeof(int));
    int* h_n2 = (int*) malloc(batchsize * sizeof(int));

    for (int i = 0; i < batchsize; i++) {
        h_n1Tilde[i] = 0;
        h_n1Union[i] = dim1;
        h_n2[i] = dim2;
    }

    int* d_n1Tilde;
    int* d_n1Union;
    int* d_n2;

    cudaMalloc((void**)&d_n1Tilde, batchsize * sizeof(int));
    cudaMalloc((void**)&d_n1Union, batchsize * sizeof(int));
    cudaMalloc((void**)&d_n2, batchsize * sizeof(int));

    cudaMemcpy(d_n1Tilde, h_n1Tilde, batchsize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n1Union, h_n1Union, batchsize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n2,h_n2, batchsize* sizeof(int), cudaMemcpyHostToDevice);

    { // timing the GPU implementations
        gettimeofday(&t_start, NULL);

        cudaError_t err = cudaDeviceSynchronize();
        printf("err: %d\n", err);

        for(int i=0; i<RUNS_GPU; i++) {
            int numBlocks = (batchsize * dim1 * dim1 + BLOCKSIZE - 1) / BLOCKSIZE;
            setSecondMatrix<<<numBlocks, BLOCKSIZE>>>(d_PointerA, d_PointerB, d_n1Tilde, d_n1Union, d_n2, dim1, batchsize);
        }

        err = cudaDeviceSynchronize();
        printf("err: %d\n", err);

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        gigaBytesPerSec = 2 * dim1 * dim1 * batchsize * sizeof(float) * 1.0e-3f / elapsed;
        printf("\n\nParallel secondMatrix runs in: %lu microsecs, GB/sec: %.2f\n\n\n"
              , elapsed, gigaBytesPerSec);
    }

    // gpuAssert( cudaPeekAtLastError() );
    return 0;
}

// int CSCToBatchedTest(CSC* A, float* ABatched, int* I, int* J, int dim1, int dim2, int batchsize) {
//     double gigaBytesPerSec;
//     unsigned long int elapsed;
//     struct timeval t_start, t_end, t_diff;

//     { // timing the CPU implementations
//         gettimeofday(&t_start, NULL);

//         for(int i=0; i<RUNS_CPU; i++) {
//             seqCSCToDense(A, ABatched,  dim1, dim2, batchsize);
//         }
        
//         cudaDeviceSynchronize();

//         gettimeofday(&t_end, NULL);
//         timeval_subtract(&t_diff, &t_end, &t_start);
//         elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
//         printf("\n\nSequential matrixMul runs in: %lu microsecs\n\n\n"
//               , elapsed, gigaBytesPerSec);
//     }

//     float* d_A;
//     float* d_B;

//     float** d_PointerA;
//     float** d_PointerB;

//     cudaMalloc((void**)&d_A, batchsize * dim1 * dim1 * sizeof(float));
//     cudaMalloc((void**)&d_B, batchsize * dim2 * dim2 * sizeof(float));

//     cudaMemcpy(d_A, A, batchsize * dim1 * dim1 * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, batchsize * dim2 * dim2 * sizeof(float), cudaMemcpyHostToDevice);

//     cudaMalloc((void**)&d_PointerA, batchsize * sizeof(float*));
//     cudaMalloc((void**)&d_PointerB, batchsize * sizeof(float*));

//     int numBlocks = (batchsize - BLOCKSIZE + 1) / BLOCKSIZE;
//     floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerA, d_A, batchsize, dim1 * dim1);
//     floatDeviceToDevicePointerKernel<<<numBlocks, BLOCKSIZE>>>(d_PointerB, d_B, batchsize, dim2 * dim2);

//     int* h_n1Tilde = (int*) malloc(batchsize * sizeof(int));
//     int* h_n1Union = (int*) malloc(batchsize * sizeof(int));
//     int* h_n2 = (int*) malloc(batchsize * sizeof(int));

//     for (int i = 0; i < batchsize; i++) {
//         h_n1Tilde[i] = 0;
//         h_n1Union[i] = dim1;
//         h_n2[i] = dim2;
//     }

//     int* d_n1Tilde;
//     int* d_n1Union;
//     int* d_n2;

//     cudaMalloc((void**)&d_n1Tilde, batchsize * sizeof(int));
//     cudaMalloc((void**)&d_n1Union, batchsize * sizeof(int));
//     cudaMalloc((void**)&d_n2, batchsize * sizeof(int));

//     cudaMemcpy(d_n1Tilde, h_n1Tilde, batchsize * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_n1Union, h_n1Union, batchsize * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_n2,h_n2, batchsize* sizeof(int), cudaMemcpyHostToDevice);

//     { // timing the GPU implementations
//         gettimeofday(&t_start, NULL);

//         for(int i=0; i<RUNS_GPU; i++) {
//             int numBlocks = (batchsize * dim1 * dim1 + BLOCKSIZE - 1) / BLOCKSIZE;
//             setSecondMatrix<<<numBlocks, BLOCKSIZE>>>(d_PointerA, d_PointerB, d_n1Tilde, d_n1Union, d_n2, dim1, batchsize);
//         }
        
//         cudaDeviceSynchronize();

//         gettimeofday(&t_end, NULL);
//         timeval_subtract(&t_diff, &t_end, &t_start);
//         elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
//         gigaBytesPerSec = 2 * dim1 * dim1 * batchsize * sizeof(int) * 1.0e-3f / elapsed;
//         printf("\n\nParallel matrixMul runs in: %lu microsecs, GB/sec: %.2f\n\n\n"
//               , elapsed, gigaBytesPerSec);
//     }

//     // gpuAssert( cudaPeekAtLastError() );
//     return 0;
// }

int runMatrixMultiplicationTest() {
    unsigned long int dim1 = 10000;
    unsigned long int dim2 = 10000;
    unsigned long int dim3 = 10000;
    float sparsity = 1.0;
    int batchsize = 1;
    
    time_t t;
    srand((unsigned) time(&t));

    float* A;
    float* B;
    float* C;

    cudaMalloc((void**)&A, dim1 * dim2 * sizeof(float));
    cudaMalloc((void**)&B, dim2 * dim3 * sizeof(float));
    cudaMalloc((void**)&C, dim1 * dim3 * sizeof(float));

    int numBlocks = (dim1 * dim2 + BLOCKSIZE - 1) / BLOCKSIZE;
    createRandomMatrix <<<numBlocks, BLOCKSIZE>>> (A, dim1, dim2, sparsity);
    
    numBlocks = (dim2 * dim3 + BLOCKSIZE - 1) / BLOCKSIZE;
    createRandomMatrix <<<numBlocks, BLOCKSIZE>>> (B, dim2, dim3, sparsity);

    matrixMultiplicationTest(A, B, C, dim1, dim2, dim3, batchsize);
}

int runSetSecondMatrixTest() {
    unsigned long int dim1 = 500;
    unsigned long int dim2 = 250;
    float sparsity = 1.0;
    int batchsize = 1;

    time_t t;
    srand((unsigned) time(&t));

    float* A;
    float* B;

    cudaMalloc((void**)&A, batchsize * dim1 * dim1 * sizeof(float));
    cudaMalloc((void**)&B, batchsize * dim2 * dim2 * sizeof(float));
    
    int numBlocks = (batchsize * dim2 * dim2 + BLOCKSIZE - 1) / BLOCKSIZE;
    createRandomMatrix <<<numBlocks, BLOCKSIZE>>> (B, dim2, dim2, sparsity);

    cudaDeviceSynchronize();
    float* h_A = (float*) malloc(batchsize * dim1 * dim1 * sizeof(float));
    cudaMemcpy(h_A, A, batchsize * dim1 * dim1 * sizeof(float), cudaMemcpyDeviceToHost);

    float* h_B = (float*) malloc(batchsize * dim2 * dim2 * sizeof(float));
    cudaMemcpy(h_B, B, batchsize * dim2 * dim2 * sizeof(float), cudaMemcpyDeviceToHost);

    setSecondMatrixTest(A, B, dim1, dim2, batchsize);
}

# endif
