#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

__global__ void simpleKernel(int length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        printf("Hello World! I'm thread %d\n", tid);
    }
}

int main() {
    int length = 10;
    int threadsPerBlock = 256;
    int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
    simpleKernel<<<blocksPerGrid, threadsPerBlock>>>(length);
    cudaDeviceSynchronize();
    return 0;
}