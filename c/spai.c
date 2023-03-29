#include "spai.h"
#include "csc.h"
#include <stdio.h>
#include <stdlib.h>

struct CSC* spai(struct CSC* A, float tol, int max_fill_in, int s) {
    // initialize M and set to diagonal
    struct CSC* M = createDiagonalCSC(A->m, A->n);

    // m_k = column in M
    for (int k = 0; k < M->n) {

        // a) Find the initial sparsity J of m_k
        // malloc space for the indeces from offset[k] to offset[k + 1]
        int* J = malloc(sizeof(int) * (M->offset[k + 1] - M->offset[k]);

        // iterate through row indeces from offset[k] to offset[k + 1]
        int j = 0;
        for (int i = M->offset[k]; i < M->offset[k + 1], i++) {
            J[j] = M->flatRowIndex[i];
            j++;
        }

        // b) 
    }
    
    return M;
}
