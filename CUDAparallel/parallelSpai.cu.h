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

// kernel for computing I, J n1 and n2
// d_A = device pointer to A
// d_I = device pointer to I
// d_J = device pointer to J
// 
__global__ void computeIandJ(CSC* d_A, CSC* d_M, int** d_I, int** d_J, int* d_n1, int* d_n2, int batchnumber, int batchsize) {
    int tid = batchnumber * batchsize + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchsize) {
        int n2 = d_M->offset[tid + 1] - d_M->offset[tid];
        int* J = (int*) malloc(n2 * sizeof(int));

        // iterate through the row indeces from offset[k] to offset[k+1] and take all elements from the flatRowIndex
        int h = 0;
        for (int i = d_M->offset[tid]; i < d_M->offset[tid + 1]; i++) {
            J[h] = d_M->flatRowIndex[i];
            h++;
        }

        // We initialize I to -1, and the iterate through all elements of J. Then we iterate through the row indeces of A from the offset J[j] to J[j] + 1. If the row index is already in I, we dont do anything, else we add it to I.
        int* I = (int*) malloc(d_A->m * sizeof(int));
        for (int i = 0; i < d_A->m; i++) {
            I[i] = -1;
        }

        int n1 = 0;
        for (int j = 0; j < n2; j++) {
            for (int i = d_A->offset[J[j]]; i < d_A->offset[J[j] + 1]; i++) {
                int keep = 1;
                for (int k = 0; k < d_A->m; k++) {
                    if (I[k] == d_A->flatRowIndex[i]) {
                        keep = 0;
                        break;
                    }
                }
                if (keep) {
                    I[n1] = d_A->flatRowIndex[i];
                    n1++;
                }
            }
        }

        // set device values
        // giver det mening at parallelisere dette?
        d_I[tid] = &I[0];
        d_J[tid] = &J[0];
        d_n1[tid] = n1;
        d_n2[tid] = n2;
    }
}

// A = matrix we want to compute SPAI on
// m, n = size of array
// tolerance = tolerance
// maxIteration = constraint for the maximal number of iterations
// s = number of rho_j - the most profitable indices
// batchsize = number of matrices to be processed in parallel
CSC* parallelSpai(CSC* A, float tolerance, int maxIterations, int s, int batchsize) {
    printCSC(A);

    // initialize cublas
    cublasHandle_t cHandle;
    cublasStatus_t stat;
    stat = cublasCreate(&cHandle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        printf("CUBLAS error: %d\n", stat);

        return NULL;
    }

    // initialize M and set to diagonal
    CSC* M = createDiagonalCSC(A->m, A->n);

    // copy A to device
    CSC* d_A = copyCSCFromHostToDevice(A);
    CSC* d_M = copyCSCFromHostToDevice(M);
    
    // compute the batchnumber 
    int batchnumber = (A->n + batchsize - 1) / batchsize;

    for (int i = 0; i < batchnumber; i++) {
        int** d_I;
        int** d_J;
        int* d_n1;
        int* d_n2;

        // malloc space
        gpuAssert(
            cudaMalloc((void**) &d_I, batchsize * sizeof(int*)));
        gpuAssert(
            cudaMalloc((void**) &d_J, batchsize * sizeof(int*)));
        gpuAssert(
            cudaMalloc((void**) &d_n1, batchsize * sizeof(int)));
        gpuAssert(
            cudaMalloc((void**) &d_n2, batchsize * sizeof(int)));
        
        computeIandJ<<<1, batchsize>>>(d_A, d_M, d_I, d_J, d_n1, d_n2, i, batchsize);



        int* n2 = (int*) malloc(batchsize * sizeof(float));
        gpuAssert(
            cudaMemcpy(n2, d_n2, batchsize * sizeof(float), cudaMemcpyDeviceToHost));
        
        // print n2
        printf("n2: ");
        for (int j = 0; j < batchsize; j++) {
            printf("%f ", n2[j]);
        }
        printf("\n");

    }
}

#endif