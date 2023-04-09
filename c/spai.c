#include "spai.h"
#include "csc.h"
#include <stdio.h>
#include <stdlib.h>

struct CSC* spai(struct CSC* A, float tol, int max_fill_in, int s) {
    // initialize M and set to diagonal
    struct CSC* M = createDiagonalCSC(A->m, A->n);

    // m_k = column in M
    for (int k = 0; k < M->n; k++) {

        // a) Find the initial sparsity J of m_k
        // malloc space for the indeces from offset[k] to offset[k + 1]
        int n2 = M->offset[k + 1] - M->offset[k];
        int* J = malloc(sizeof(int) * n2);

        // iterate through row indeces from offset[k] to offset[k + 1]
        int j = 0;
        for (int i = M->offset[k]; i < M->offset[k + 1]; i++) {
            J[j] = M->flatRowIndex[i];
            j++;
        }

        // b) Compute the row indices I of the corresponding nonzero entries of A(i, J)
        printf("test1");
        int* I = calloc(M->m, sizeof(int));
        printf("test");
        int n1 = 0;
        for (int j = 0; j < n2; j++) {
            for (int i = A->offset[j]; A->offset[j + 1]; i++) {
                int keep = 1;
                for (int h = 0; h < M->m; h++) {
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
    }
    
    return M;
}
