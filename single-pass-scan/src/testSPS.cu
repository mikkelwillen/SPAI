#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hostSkel.cu.h"
#include "constants.cu.h"
#include "rTSKernels.cu.h"
#include "sPSKernels.cu.h"

void initArray(int32_t* inp_arr, const uint32_t N, const int R) {
    const uint32_t M = 2*R+1;
    for(uint32_t i=0; i<N; i++) {
        inp_arr[i] = (rand() % M) - R;
    }
}

int bandwidthScan( const uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
                 , const size_t   N     // length of the input array
                 , int* h_in            // device input  of length N
                 , int* h_ref           // device result of length N
) {
    // dry run to exercise the d_out allocation!
    // const uint32_t num_blocks = (N + B - 1) / B;
    CPU_scan< Add<int> >(h_in, N, h_ref);

    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    { // timing the GPU implementations
        gettimeofday(&t_start, NULL);

        for(int i=0; i<RUNS_GPU; i++) {
            CPU_scan< Add<int> >(h_in, N, h_ref);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        gigaBytesPerSec = 2 * N * sizeof(int) * 1.0e-3f / elapsed;
        printf("Naive CPU scan runs in: %lu microsecs, GB/sec: %.2f\n\n\n"
              , elapsed, gigaBytesPerSec);
    }

    gpuAssert( cudaPeekAtLastError() );
    return 0;
}

int scanTestRTS(  const uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
                , const size_t   N     // length of the input array
                , int* h_in            // host input    of size: N * sizeof(int)
                , int* h_ref
                , int* d_in            // device input  of size: N * sizeof(ElTp)
                , const int version
) {
    const size_t mem_size = N * sizeof(int);
    int* d_tmp;
    int* d_out;
    int* h_out = (int*)malloc(mem_size);
    cudaMalloc((void**)&d_tmp, MAX_BLOCK * sizeof(int));
    gpuAssert( cudaMalloc((void**)&d_out,   mem_size) );
    cudaMemset(d_out, 0, N * sizeof(int));

    { // reduce then scan reference test
        // dry run to exercise d_tmp allocation
        scanInc< Add<int> > ( B, N, d_out, d_in, d_tmp );

        // time the GPU computation
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        for(int i=0; i<RUNS_GPU; i++) {
            scanInc< Add<int> > ( B, N, d_out, d_in, d_tmp );
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        double gigaBytesPerSec = N  * (2*sizeof(int) + sizeof(int)) * 1.0e-3f / elapsed;
        printf("Scan Inclusive AddI32 GPU Kernel runs in: %lu microsecs, GB/sec: %.2f\n"
            , elapsed, gigaBytesPerSec);
        cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

        gpuAssert(cudaPeekAtLastError());
    }

    { // Validation
        uint32_t loop = 1;
        for(uint32_t i = 0; i < N; i++) {
            if(loop && h_out[i] != h_ref[i]) {
                printf("!!!INVALID!!!: Reduce-then-scan at index %d, dev-val: %d, host-val: %d\n"
                      , i, h_out[i], h_ref[i]);
                loop = 0;
                //exit(1);
            }
        }
        if (loop) {printf("Reduce-then-scan AddI32: VALID result!\n\n");}
    }


    free(h_out);
    cudaFree(d_tmp);
    gpuAssert( cudaFree(d_out) );

    return 0;
}

//template<class OP>                     // element-type and associative operator properties
int scanTestSPS(  const uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
                , const size_t   N     // length of the input array
                , int* h_in            // host input    of size: N * sizeof(int)
                , int* h_ref           // reference solution
                , int* d_in            // device input  of size: N * sizeof(ElTp)
                , const int version
) {
    const size_t mem_size = N * sizeof(int);
    int* d_out;
    int* h_out = (int*)malloc(mem_size);
    gpuAssert( cudaMalloc((void**)&d_out,   mem_size) );
    cudaMemset(d_out, 0, N * sizeof(int));

    { // single-pass scan test
        // dry run to exercise d_tmp allocation
        // singlePassScan< Add<int> > (B, N, d_out, d_in, d_fs, d_as, d_ps, version);
        singlePassScan< Add<int> > (B, N, d_out, d_in, version);

        // time the GPU computation
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        for(int i=0; i<RUNS_GPU; i++) {
            singlePassScan< Add<int> > (B, N, d_out, d_in, version);
            // singlePassScan< Add<int> > (B, N, d_out, d_in, d_fs, d_as, d_ps, version);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        double gigaBytesPerSec = N  * (2*sizeof(int) + sizeof(int)) * 1.0e-3f / elapsed;
        printf("Single-pass scan AddI32 GPU Kernel runs in: %lu microsecs, GB/sec: %.2f\n"
            , elapsed, gigaBytesPerSec);
        cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

        gpuAssert(cudaPeekAtLastError());
    }

    { // Validation
        for(uint32_t i = 0; i < N; i++) {
            if(h_out[i] != h_ref[i]) {
                printf("!!!INVALID!!!: Single-pass scan AddI32 at index %d, dev-val: %d, host-val: %d\n"
                      , i, h_out[i], h_ref[i]);
                printf("Next 5 values of h_out and h_ref and h_in:\n %d, %d, %d \n%d, %d, %d \n%d, %d, %d \n%d, %d, %d \n%d, %d, %d \n",h_out[i + 1], h_ref[i + 1], h_in[i+1], h_out[i + 2], h_ref[i + 2], h_in[i+2], h_out[i + 3], h_ref[i + 3], h_in[i+3], h_out[i + 4], h_ref[i + 4], h_in[i+4], h_out[i + 5], h_ref[i + 5], h_in[i+5]);
                exit(1);
            }
        }
        printf("Single-pass scan AddI32: VALID result!\n\n");
    }

    free(h_out);
    gpuAssert( cudaFree(d_out) );

    return 0;
}

int main (int argc, char * argv[]) {
    if (argc != 4) {
        printf("Usage: %s <array-length> <block-size> <version>\n", argv[0]);
        exit(1);
    }

    initHwd();

    const uint32_t N = atoi(argv[1]);
    const uint32_t B = atoi(argv[2]);
    const uint32_t version = atoi(argv[3]);

    printf("Testing parallel basic blocks for input length: %d and CUDA-block size: %d\n\n\n", N, B);

    const size_t mem_size = N * sizeof(int);
    int* h_in    = (int*) malloc(mem_size);
    int* d_in;
    gpuAssert( cudaMalloc((void**)&d_in ,   mem_size) );

    initArray(h_in, N, 13);
    gpuAssert( cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice) );

    int* h_ref = (int*)malloc(mem_size);

    // CPU_scan< Add<int> >(h_in, N, h_ref);
    bandwidthScan(B, N, h_in, h_ref);

    // inclusive scan and segmented scan with int addition
    scanTestRTS(B, N, h_in, h_ref, d_in, version);
    scanTestSPS(B, N, h_in, h_ref, d_in, version);
    // scanIncAddI32< Add<int> >(B, N, h_in, d_in, d_out, version);

    // cleanup memory
    free(h_in);
    free(h_ref);
    gpuAssert( cudaFree(d_in ) );
}
