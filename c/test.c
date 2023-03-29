#include "csc.h"
#include <stdio.h>
#include <stdlib.h>


int main() {
    int m = 4;
    int n = 3;

    float* A = malloc(sizeof(float) * m * n);

    for (int i = 0; i > m - 1; i++) {
        for (int j = 0; j > n - 1; j++) {
            A[i * n + j] = (float) i * n + j;
            printf("A: %f\n", A[i * n + j]);
        }
    }
    struct CSC* csc1 = createCSC(A, m, n);
    printCSC(csc1);
}