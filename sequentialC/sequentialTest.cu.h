#ifndef STEST_H
#define STEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "csc.cu.h"
#include "constants.cu.h"
#include "sequentialSpai.cu.h"
#include "cuSOLVERInv.cu.h"

/*
Function for testing and timing the sequential SPAI implementation
A = the matrix to be inverted
tolerance = the tolerance
maxIteration = the maximum number of iterations
s = smallest indices to be used
*/
int sequentialTest(CSC* A, float tolerance, int maxIteration, int s) {

    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    { // timing the GPU implementations
        sequentialSpai(A, tolerance, maxIteration, s);
        gettimeofday(&t_start, NULL);

        for(int i=0; i<RUNS_CPU; i++) {
            sequentialSpai(A, tolerance, maxIteration, s);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        gigaBytesPerSec = 2 * A->countNonZero * A->n * sizeof(float) * 1.0e-3f / elapsed;
        printf("\n\nSequential SPAI runs in: %lu microsecs, GB/sec: %.2f\n\n\n"
              , elapsed, gigaBytesPerSec);
    }

    // gpuAssert( cudaPeekAtLastError() );
    return 0;

}

/*
Function for testing and timing the cuSOLVER implementation
A = the matrix to be inverted
n = the size of the matrix (n x n)
*/
int sequentialTestCuSOLVER(float* A, int n) {

    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    { // timing the GPU implementations
        float* res = cuSOLVERInversion(A, n, n);
        free(res);
        printf("starting test\n");
        gettimeofday(&t_start, NULL);

        for(int i=0; i<RUNS_CPU; i++) {
            res = cuSOLVERInversion(A, n, n);
            free(res);
        }

        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        printf("ending test\n");
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
        gigaBytesPerSec = 2 * n * n * n * 1.0e-3f / elapsed;
        printf("\ncuSOLVER runs in: %lu microsecs, GB/sec: %.2f\n"
              , elapsed, gigaBytesPerSec);
    }

    // gpuAssert( cudaPeekAtLastError() );
    return 0;

}

/* 
Function for testing the accuracy of the sequential SPAI implementation
A = the matrix to be inverted
m = the number of rows in the matrix
n = the number of columns in the matrix
sparsity = the sparsity of the matrix
tolerance = the tolerance
maxIteration = the maximum number of iterations
s = smallest indices to be used
*/
int runIdentityTest(CSC* cscA, int m, int n, float sparsity, float tolerance, int maxIterations, int s) {
    float* identity = (float*) malloc (sizeof(float) * n * n);

    struct CSC* res = sequentialSpai(cscA, tolerance, maxIterations, s);
    printf("After sequentialSpai\n");
    int* I = (int*) malloc(sizeof(int) * m);
    int* J = (int*) malloc(sizeof(int) * n);
    for (int i = 0; i < m; i++) {
        I[i] = i;
    }
    for (int i = 0; i < n; i++) {
        J[i] = i;
    }

    float* A = CSCToDense(cscA, I, J, m, n);
    float* inv = CSCToDense(res, I, J, m, n);
    
    // identity = A * inv
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n;j++) {
            identity[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                identity[i * n + j] += A[i * n + k] * inv[k * n + j];
            }
        }
    }

    // print A
    printf("A:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n;j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
    }

    // print inv
    printf("inv:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n;j++) {
            printf("%f ", inv[i * n + j]);
        }
        printf("\n");
    }

    // print identity
    printf("identity:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j++) {
            printf("%f ", identity[i * n + j]);
        }
        printf("\n");
    }

    // calculate error
    float error = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j++) {
            error += (identity[i * n + j] - (i == j ? 1.0 : 0.0)) * (identity[i * n + j] - (i == j ? 1.0 : 0.0));
        }
    }

    printf("Error: %.15f%\n", error);
}

/*
Function for testing the accuracy of the cuSOLVER implementation
A = the matrix to be inverted
n = the size of the matrix (n x n)
*/
int runcuSOLVERTest(float* A, int n) {
    float* inv = cuSOLVERInversion(A, n, n);
    printf("After cuSOLVERInversion\n");
    
    // identity = A * inv
    float* identity = (float*) malloc (sizeof(float) * n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j++) {
            identity[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                identity[i * n + j] += A[i * n + k] * inv[k * n + j];
            }
        }
    }

    // print A
    printf("A:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
    }

    // print inv
    printf("inv:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j++) {
            printf("%f ", inv[i * n + j]);
        }
        printf("\n");
    }

    // print identity
    printf("identity:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j++) {
            printf("%f ", identity[i * n + j]);
        }
        printf("\n");
    }

    // calculate error
    float error = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j++) {
            error += (identity[i * n + j] - (i == j ? 1.0 : 0.0)) * (identity[i * n + j] - (i == j ? 1.0 : 0.0));
        }
    }

    printf("Error: %f%\n", error);
}

#endif