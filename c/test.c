#include "csc.h"
#include "spai.h"
#include <stdio.h>
#include <stdlib.h>


int main() {
    int m = 4;
    int n = 3;

    float* A = malloc(sizeof(float) * m * n);
    float* B = malloc(sizeof(float) * m * n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (float) i * n + j;
        }
    }

    B[0] = 20.0; B[1] = 0.0; B[2] = 0.0; B[3] = 0.0; B[4] = 30.0; B[5] = 10.0; 
    B[6] = 0.0; B[7] = 0.0; B[8] = 0.0; B[9] = 0.0; B[10] = 40.0; B[11] = 0.0;

    struct CSC* cscA = createCSC(A, m, n);
    // struct CSC* cscB = createCSC(B, m, n);
    // struct CSC* cscDia = createDiagonalCSC(m, n);
    // printCSC(cscA);
    // printCSC(cscB);
    // printCSC(cscDia);

    spai(cscA, 0.01, 5, 1);
}