#include "spai.h"
#include "csc.h"
#include <stdio.h>
#include <stdlib.h>

CSC* spai(CSC* A, float tol, int max_fill_in, int s) {
    printCSC(A);
    // initialize M and set to diagonal
    CSC* M = createDiagonalCSC(A->m, A->n);

    // m_k = column in M
    for (int k = 0; k < M->n; k++) {

        // a) Find the initial sparsity J of m_k
        // malloc space for the indeces from offset[k] to offset[k + 1]
        int n2 = M->offset[k + 1] - M->offset[k];
        int* J = malloc(sizeof(int) * n2);

        // iterate through row indeces from offset[k] to offset[k + 1] and take all elements from the flatRowIndex
        int h = 0;
        for (int i = M->offset[k]; i < M->offset[k + 1]; i++) {
            J[h] = M->flatRowIndex[i];
            h++;
        }

        // // printJ
        // printf("\nJ: ");
        // for (int i = 0; i < n2; i++) {
        //     printf("%d ", J[i]);
        // }

        // b) Compute the row indices I of the corresponding nonzero entries of A(i, J)
        // We initialize I to -1, and the iterate through all elements of J. Then we iterate through the row indeces of A from the offset J[j] to J[j] + 1. If the row index is already in I, we dont do anything, else we add it to I.
        int* I = malloc(sizeof(int) * A->m);
        for (int i = 0; i < A->m; i++) {
            I[i] = -1;
        }
        int n1 = 0;
        for (int j = 0; j < n2; j++) {
            for (int i = A->offset[J[j]]; i < A->offset[J[j] + 1]; i++) {
                int keep = 1;
                for (int h = 0; h < A->m; h++) {
                    if (A->flatRowIndex[i] == I[h]) {
                        keep = 0;
                    }
                }
                if (keep == 1) {
                    I[n1] = A->flatRowIndex[i];
                    n1++;
                }
            }
        }

        // // print I
        // printf("\nI: ");
        // for (int i = 0; i < n1; i++) {
        //     printf("%d ", I[i]);
        // }

        // c) Create Â = A(I, J)
        // We initialize AHat to zeros. Then we iterate through all indeces of J, and iterate through all indeces of I. For each of the indices of I and the indices in the flatRowIndex, we check if they match. If they do, we add that element to AHat.
        float* AHat = calloc(n1 * n2, sizeof(float));
        for(int j = 0; j < n2; j++) {
            for (int i = 0; i < n1; i++) {
                for (int l = A->offset[J[j]]; l < A->offset[J[j] + 1]; l++) {
                    if (I[i] == A->flatRowIndex[l]) {
                        AHat[j * n2 + i] = A->flatData[l];
                    }
                }
            }
        }

        // print AHat
        printf("\nAhat: ");
        for (int i = 0; i < n1 * n2; i++) {
            printf("%f ", AHat[i]);
        }

        // d) do QR decomposition of AHat
        
    }
    
    return M;
}
