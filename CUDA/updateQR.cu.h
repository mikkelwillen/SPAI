#ifndef UPDATE_QR_H
#define UPDATE_QR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"

void* updateQR(CSC* A, float* Q, float* R, int* I, int* J, int* ITilde, int* JTilde, int* IUnion, int* JUnion, int n1, int n2, int n1Tilde, int n2Tilde, int n1Union, int n2Union, float* m_kOut) {
    printf("------UPDATE QR------\n");

    // Create ABar = A(UnionI, UnionJ)
    float* ABar = CSCToDense(A, IUnion, JUnion, n1Union, n2Union);

}




#endif