#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hostSkel.cu.h"
#include "constants.cu.h"
#include "qrKernels.cu.h"
#include "spaiKernels.cu.h"


// lav fra array til matrix
void initMatrix(int32_t* inp_arr, const uint32_t N, const int R) {
    const uint32_t M = 2*R+1;
    for(uint32_t i=0; i<N; i++) {
        inp_arr[i] = (rand() % M) - R;
    }
}

int main (int argc, char * argv[]) {
    if (argc != 4) {
        printf("Usage: %s <matrix-M> <matrix-N> <block-size> \n", argv[0]);
        exit(1);
    }

    initHwd();

    const uint32_t M = atoi(argv[1]);
    const uint32_t N = atoi(argv[2]);
    const uint32_t B = atoi(argv[3]);

    printf("Testing parallel basic blocks for input length: %d and CUDA-block size: %d\n\n\n", N, B);

    // sizeof float?
    const size_t mem_size = M * N * sizeof(int);
    int* h_in    = (int*) malloc(mem_size);
    int* d_in;
    gpuAssert( cudaMalloc((void**)&d_in ,   mem_size) );

    initMatrix(h_in, M, N, 13);
    gpuAssert( cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice) );

    int* h_ref = (int*)malloc(mem_size);

    // indset tests her

    // cleanup memory
    free(h_in);
    free(h_ref);
    gpuAssert( cudaFree(d_in ) );
}
