#ifndef QR_BATCHED_H
#define QR_BATCHED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"

int qrBatched(float* AHat, int n1, int n2, float* Q, float* R) {
    printf("\nDo QR decomposition of AHat\n");
    // create cublas handle
    cublasHandle_t cHandle;
    cublasStatus_t stat = cublasCreate(&cHandle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("\ncublasCreate failed");
        printf("\ncublas error: %d\n", stat);
    }

    int lda = 1;
    int min = MIN(n1, n2);
    int ltau = MAX(1, min);
    float* d_AHat;
    float* d_Tau;
    float* h_Tau = (float*) malloc(sizeof(float) * ltau * ltau * BATCHSIZE);
    int info;

    printf("ltau: %d\n", ltau);

    // qr initialization
    cudaMalloc((void**) &d_AHat, n1 * n2 * BATCHSIZE * sizeof(float));
    cudaMalloc((void**) &d_Tau, ltau * ltau * BATCHSIZE * sizeof(float));
    
    cudaMemcpy(d_AHat, AHat, n1 * n2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_Tau, 0.0, ltau * ltau * BATCHSIZE * sizeof(float));

    stat = cublasSgeqrfBatched(cHandle,
                                n1,
                                n2,
                                &d_AHat,
                                lda,
                                &d_Tau,
                                &info,
                                BATCHSIZE);
    
    if (info != 0) {
        printf("\nparameters are invalid\n");
    }
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("\ncublasSgeqrfBatched failed");
        printf("\ncublas error: %d\n", stat);
    }

    cudaMemcpy(h_Tau, d_Tau, ltau * ltau * BATCHSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nh_Tau: ");
    for (int i = 0; i < ltau * BATCHSIZE; i++) {
        printf("\nith vector: ");
        for (int j = 0; j < ltau; j++) {
            printf("%f ", h_Tau[i * ltau + j]);
        }
    }
    printf("\nh_Tau: %f", h_Tau[0]);
}

#endif
