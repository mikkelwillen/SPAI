#ifndef HELPER_KERNELS_CU_H
#define HELPER_KERNELS_CU_H

#include <stdio.h>
#include <stdlib.h>

// kernel to copy d_data to d_Pointer
// d_Pointer = an array of pointers to the start of each matrix in d_data
// d_data = an array of batch matrices
// pointerArraySize = the size of d_Pointer
// dataSize = the size of each matrix in d_data
__global__ void deviceToDevicePointerKernel(float** d_Pointer, float* d_data, int pointerArraySize, int dataSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pointerArraySize) {
        d_Pointer[tid] = &d_data[tid * dataSize];
    }
}

// kernel to copy int d_data to d_Pointer
// d_Pointer = an array of pointers to the start of each matrix in d_data
// d_data = an array of batch matrices
// pointerArraySize = the size of d_Pointer
// dataSize = the size of each matrix in d_data
__global__ void iDeviceToDevicePointerKernel(int** d_Pointer, int* d_data, int pointerArraySize, int dataSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pointerArraySize) {
        d_Pointer[tid] = &d_data[tid * dataSize];
    }
}

#endif