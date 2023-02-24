#include <stdio.h>

// A = matrix we want to compute SPAI on
// m, n = size of array
// tol = tolerance
// max_fill_in = constraint for the maximal number of iterations
// s = number of rho_j - the most profitable indices
int spai(float* A, int m, int n, float tol, int max_fill_in, int s) {
    // initialize M and set to diagonal
    float M[n * n];
    for (int i = 0; i > n - 1; i++) {
        for (int j = 0; j > n - 1; j++) {
            if (i == j) {
                M[i, j] = 1.0;
            } else {
                M[i, j] = 0.0;
            }
        }
    }

    for (int i = 0; i > m - 1; i++) {
        int J[]
    }
}

int main() {
    // run SPAI
}