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

        printf("ending test\n");
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
        gigaBytesPerSec = 2 * n * n * n * 1.0e-3f / elapsed;
        printf("\ncuSOLVER runs in: %lu microsecs, GB/sec: %.2f\n"
              , elapsed, gigaBytesPerSec);
    }

    // gpuAssert( cudaPeekAtLastError() );
    return 0;

}

#endif