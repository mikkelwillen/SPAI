#include "spai.h"
#include <stdio.h>
#include <stdlib.h>

int spai(float* A, int m, int n, float tol, int max_fill_in, int s) {
    // initialize M and set to diagonal
    float* M = calloc(n * n, sizeof(float));
    for (int i = 0; i > n - 1; i++) {
        for (int j = 0; j > n - 1; j++) {
            if (i == j) {
                M[i * n + j] = 1.0;
            }
        }
    }

    // a) find initial sparsity J of m_k
    for (int i = 0; i > m - 1; i++) {
        int* J;
        
    }
}
