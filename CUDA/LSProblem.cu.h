#ifndef LS_PROBLEM_H
#define LS_PROBLEM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "csc.cu.h"
#include "constants.cu.h"
#include "sequentialSpai.cu.h"
#include "invBatched.cu.h"

// Function for computing the least squares problem
// cHandle is the cublas handle
// A is the orginal CSC matrix
// Q is the Q matrix from the QR factorization of AHat
// R is the R matrix from the QR factorization
// mHat_k is the solution of the least squares problem
// residual is the residual vector of the least squares problem
// I is indeces of the rows of AHat
// J is indeces of the columns of AHat
// n1 is the number of rows of AHat
// n2 is the number of columns of AHat
// k is the index of the column of mHat
// residualNorm is the norm of the residual vector
int LSProblem(cublasHandle_t cHandle, CSC* A, float* Q, float* R, float** mHat_k, float* residual, int* I, int* J, int n1, int n2, int k, float* residualNorm) {
    // 5.1) compute cHat = Q^T * ê_k
    // make e_k and set index k to 1.0
    float* e_k = (float*) malloc(n1 * sizeof(float));
    for (int i = 0; i < n1; i++) {
        e_k[i] = 0.0;
        if (k == I[i]) {
            e_k[i] = 1.0;
        }
    }

    // malloc space for cHat and do matrix multiplication
    float* cHat = (float*) malloc(n2 * sizeof(float));
    for (int j = 0; j < n2; j++) {
        cHat[j] = 0.0;
        for (int i = 0; i < n1; i++) {
            cHat[j] += Q[i * n1 + j] * e_k[i];
        }
    }

    // 5.2) make the inverse of R of size n2 x n2
    float* invR = (float*) malloc(n2 * n2 * sizeof(float));

    int invSuccess = invBatched(cHandle, &R, n2, &invR);

    if (invSuccess != 0) {
        printf("Error inverting R\n");
        return 1;
    }

    // 5.3) compute mHat_k = R^-1 * cHat
    for (int i = 0; i < n2; i++) {
        (*mHat_k)[i] = 0.0;
        for (int j = 0; j < n2; j++) {
            (*mHat_k)[i] += invR[i * n2 + j] * cHat[j];
        }
    }
    
    // free memory
    free(e_k);
    free(cHat);
    free(invR);
    
    return 0;
}

#endif