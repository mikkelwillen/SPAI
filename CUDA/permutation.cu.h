#ifndef PERMUTATION_H
#define PERMUTATION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cublas_v2.h"
#include "csc.cu.h"
#include "constants.cu.h"

void* createPermutationMatrices(int* I, int* J, int n1, int n2, float* Pr, float* Pc) {
    // create row permutation matrix of size n1 x n1
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n1; j++) {
            if (I[j] == i) {
                Pr[i*n1 + j] = 1;
            }
            else {
                Pr[i*n1 + j] = 0;
            }
        }
    }

    // create column permutation matrix of size n2 x n2
    for (int j = 0; j < n2; j++) {
        for (int i = 0; i < n2; i++) {
            if (J[i] == j) {
                Pc[i*n2 + j] = 1;
            }
            else {
                Pc[i*n2 + j] = 0;
            }
        }
    }
}




#endif