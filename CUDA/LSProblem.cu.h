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

void* LSProblem(cublasHandle_t cHandle, CSC* A, float* Q, float* R, float** mHat_k, float** residual, int* I, int* J, int n1, int n2, int k, float* residualNorm) {
    // e) compute cHat = Q^T * ê_k
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
    for (int i = 0; i < n1; i++) {
        cHat[i] = 0.0;
        for (int j = 0; j < n2; j++) {
            cHat[j] += Q[j*n2 + i] * e_k[i];
        }
    }

    // f) compute mHat_k = R^-1 * cHat
    // make the inverse of R of size n2 x n2
    float* invR = (float*) malloc(n2 * n2 * sizeof(float));
    invBatched(cHandle, R, n2, invR);

    printf("invR:\n");
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n2; j++) {
            printf("%f ", invR[i * n2 + j]);
        }
        printf("\n");
    }
    // realloc mHat_k to size n2 and do matrix multiplication
    if (mHat_k != NULL) {
        printf("freeing mHat_k\n");
        free(mHat_k);
    }
    printf("after possible free\n");
    *mHat_k = (float*) malloc(n2 * sizeof(float));
    for (int i = 0; i < n2; i++) {
        *mHat_k[i] = 0.0;
        for (int j = 0; j < n2; j++) {
            *mHat_k[i] += invR[i * n2 + j] * cHat[j];
        }
    }
    printf("mHat_k:\n");
    for (int i = 0; i < n2; i++) {
        printf("%f ", mHat_k[i]);
    }
    // g) compute residual = A * mHat_k - e_k
    // malloc space for residual
    if (residual != NULL) {
        free(residual);
    }
    *residual = (float*) malloc(A->m * sizeof(float));

    // do matrix multiplication
    for (int i = 0; i < A->m; i++) {
        *residual[i] = 0.0;
        for (int j = 0; j < n2; j++) {
            for (int h = A->offset[k]; h < A->offset[k+1]; h++) {
                if (i == A->flatRowIndex[h]) {
                    *residual[i] += A->flatData[h] * mHat_k[j];
                }
            }
        }
        if (i == k) {
            *residual[i] -= 1.0;
        }
    }
    printf("residual:\n");
    for (int i = 0; i < A->m; i++) {
        printf("%f ", *residual[i]);
    }

    // compute the norm of the residual
    *residualNorm = 0.0;
    for (int i = 0; i < A->m; i++) {
        *residualNorm += *residual[i] * *residual[i];
    }
    *residualNorm = sqrt(*residualNorm);
}

#endif