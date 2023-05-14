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

// A = matrix we want to compute SPAI on
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

    // compute the batchnumber 
    int batchnumber = (A->n + batchsize - 1) / batchsize;

    
}

#endif