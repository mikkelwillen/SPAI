#ifndef QR_BATCHED_H
#define QR_BATCHED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"

// vi skal have kigget på nogle forskellige størrelser
// fx skal vi finde ud af, om vi vil kopiere alt data til GPU'en med det samme, eller om vi skal kopiere hver aHat separat og så lave en batch af dem

__global__ void printKernel(int length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        printf("tid %d: \n", tid);
    }
}

__global__ void printDeviceArrayKernel(float* h_AHat, int length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        printf("tid %d: %f\n", tid, h_AHat[tid]);
    }
}

__global__ void deviceToDevicePointerKernel(float** d_AHat, float* h_AHat, int batch, int n1, int n2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < BATCHSIZE) {
        d_AHat[tid] = &h_AHat[tid * n1 * n2];
    }
}

__global__ void tauDeviceToDevicePointerKernel(float** d_Tau, float* h_Tau, int batch, int ltau) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < BATCHSIZE * ltau) {
        d_Tau[tid] = &h_Tau[tid * ltau];
    }
}

__global__ void devicePointerToDeviceKernel(float** d_tau, float* h_tau, int batch, int n1, int n2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < BATCHSIZE * n1 * n2) {
        int i = tid / n1;
        int j = tid % n1;
        h_tau[tid] = d_tau[i][j];
    }
}


// virker sgu ikke rigtigt
__global__ void printDeviceArrayPointerKernel(float** d_AHat, int length, int batch) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("hallo?\n");
    if (tid < length) {
        printf("tid %d: %f\n", tid, d_AHat[batch][tid]);
    }
}

int qrBatched(float* AHat, int n1, int n2, float* Q, float* R) {
    printf("\nDo QR decomposition of AHat\n");
    // create cublas handle
    cublasHandle_t cHandle;
    cublasStatus_t stat = cublasCreate(&cHandle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("\ncublasCreate failed");
        printf("\ncublas error: %d\n", stat);
    }

    int lda = n1;
    int min = MIN(n1, n2);
    int ltau = MAX(1, min);
    const size_t tauMemSize = n1T * n1T * BATCHSIZE * sizeof(float);
    const size_t tauPointerMemSize = BATCHSIZE * n1T * sizeof(float*);
    const size_t AHatMemSize = n1 * n2 * BATCHSIZE * sizeof(float);
    const size_t AHatPointerMemSize = BATCHSIZE * sizeof(float*);
    float* tau = (float*) malloc(tauMemSize);
    float* h_AHat;
    float* h_Tau;
    float** d_AHat;
    float** d_Tau;
    int info;

    gpuAssert(
        cudaMalloc((void**) &h_AHat, AHatMemSize));
    gpuAssert(
        cudaMemcpy(h_AHat, AHat, AHatMemSize, cudaMemcpyHostToDevice));
    gpuAssert(
        cudaMalloc((void**) &d_AHat, AHatPointerMemSize));
    deviceToDevicePointerKernel <<< 1, BATCHSIZE >>> (d_AHat, h_AHat, BATCHSIZE, n1, n2);
    
    gpuAssert(
        cudaMalloc((void**) &h_Tau, tauMemSize));
    gpuAssert(
        cudaMalloc((void**) &d_Tau, tauPointerMemSize));
    tauDeviceToDevicePointerKernel <<< 1, BATCHSIZE * n1T >>> (d_Tau, h_Tau, BATCHSIZE, n1T);

    stat = cublasSgeqrfBatched(cHandle,
                                n1,
                                n2,
                                d_AHat,
                                lda,
                                d_Tau,
                                &info,
                                BATCHSIZE);
    
    if (info != 0) {
        printf("\nparameters are invalid\n");
    }
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("\ncublasSgeqrfBatched failed");
        printf("\ncublas error: %d\n", stat);
    }

    printf("after cublasSgeqrfBatched\n");

    gpuAssert(
        cudaMemcpy(AHat, h_AHat, AHatMemSize, cudaMemcpyDeviceToHost));
    gpuAssert(
        cudaMemcpy(tau, h_Tau, tauMemSize, cudaMemcpyDeviceToHost));
    
    // print tau
    for (int i = 0; i < BATCHSIZE; i++) {
        printf("tau %d:\n", i);
        for (int j = 0; j < n1T; j++) {
            for (int k = 0; k < n1T; k++) {
                printf("%f ", tau[i * n1T * n1T + j * n1T + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    // print AHat
    for (int i = 0; i < BATCHSIZE; i++) {
        printf("AHat %d:\n", i);
        for (int j = 0; j < n1; j++) {
            for (int k = 0; k < n2; k++) {
                printf("%f ", AHat[i * n1 * n2 + j * n2 + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    // make R
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n2; j++) {
            if (i <= j) {
                R[i * n2 + j] = AHat[i * n2 + j];
            } else {
                R[i * n2 + j] = 0;
            }
        }
    }

    // make Q
    for (int j = 0; j < n1T; j++) {
        // compute v * v^T
        float* v = (float*) malloc(n1 * sizeof(float));
        for (int i = 0; i < n1; i++) {
            if (i > j) {
                v[i] = 0;
            } else if (i == j) {
                v[i] = 1;
            }
            else {
                v[i] = AHat[i * n2 + j];
            }
        }

        float* vvt = (float*) malloc(n1 * n1 * sizeof(float));
        for (int i = 0; i < n1; i++) {
            for (int k = 0; k < n1; k++) {
                vvt[i * n1 + k] = v[i] * v[k];
            }
        }

        // malloc H_j
        float* H = (float*) malloc(n1T * n1T * sizeof(float));
        // compute H_j
        
    }


    // make Q
    // for (int j = 0; j < ltau; j++) {
    //     compute v * v^T
    //     float vvt = 0;
    //     for (int i = 0; i < ltau; i++) {
    //         vvt += tau[i * ltau + j] * tau[i * ltau + j];
    //     }
    //     printf("vvt: %f\n", vvt);

    //     malloc H_j
    //     float* H = (float*) malloc(ltau * ltau * sizeof(float));
    //     compute H_j
    //     for (int i = 0; i < ltau; i++) {
    //         for (int k = 0; k < ltau; k++) {
    //             if (i == k) {
    //                 H[i * ltau + k] = 1 - 2 * tau[j * ltau + i] * tau[j * ltau + k] / vvt;
    //             } else {
    //                 H[i * ltau + k] = -2 * tau[j * ltau + i] * tau[j * ltau + k] / vvt;
    //             }
    //         }
    //     }
    //     print H_j
    //     printf("H_%d:\n", j);
    //     for (int i = 0; i < ltau; i++) {
    //         for (int k = 0; k < ltau; k++) {
    //             printf("%f ", H[i * ltau + k]);
    //         }
    //         printf("\n");
    //     }

    //     if (j == 0) {
    //         for (int i = 0; i < ltau; i++) {
    //             for (int k = 0; k < ltau; k++) {
    //                 Q[i * ltau + k] = H[i * ltau + k];
    //             }
    //         }
    //     } else {
    //         float* QH = (float*) malloc(ltau * ltau * sizeof(float));
    //         for (int i = 0; i < ltau; i++) {
    //             for (int k = 0; k < ltau; k++) {
    //                 QH[i * ltau + k] = 0;
    //                 for (int m = 0; m < ltau; m++) {
    //                     QH[i * ltau + k] += Q[i * ltau + m] * H[m * ltau + k];
    //                 }
    //             }
    //         }
    //         print QH
    //         printf("QH_%d:\n", j);
    //         for (int i = 0; i < ltau; i++) {
    //             for (int k = 0; k < ltau; k++) {
    //                 printf("%f ", QH[i * ltau + k]);
    //             }
    //             printf("\n");
    //         }

    //         for (int i = 0; i < ltau; i++) {
    //             for (int k = 0; k < ltau; k++) {
    //                 Q[i * ltau + k] = QH[i * ltau + k];
    //             }
    //         }
    //     }
    // }

    printf("print R\n");
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n2; j++) {
            printf("%f ", R[i * n2 + j]);
        }
        printf("\n");
    }

    printf("print Q\n");
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            printf("%f ", Q[i * n2 + j]);
        }
        printf("\n");
    }

    // compute Q * R
    float* QR = (float*) malloc(n1 * n2 * sizeof(float));
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            QR[i * n2 + j] = 0;
            for (int k = 0; k < n2; k++) {
                QR[i * n2 + j] += Q[i * n2 + k] * R[k * n2 + j];
            }
        }
    }

    printf("print QR\n");
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            printf("%f ", QR[i * n2 + j]);
        }
        printf("\n");
    }


    // // free and destroy
    // gpuAssert(
    //     cudaFree(d_AHat));
    // gpuAssert(
    //     cudaFree(d_Tau));
    // free(h_Tau);
    // cublasDestroy(cHandle);
    return 0;
}

#endif
