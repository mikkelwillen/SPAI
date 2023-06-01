#ifndef PERMUTATION_H
#define PERMUTATION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Function to create permutation matrices
// I is the row permutation vector
// J is the column permutation vector
// n1 is the number of rows in the original matrix
// n2 is the number of columns in the original matrix
// Pr is the row permutation matrix output
// Pc is the column permutation matrix output
void* createPermutationMatrices(int* I, int* J, int n1, int n2, float* Pr, float* Pc) {
    // create normalized index of I
    int* IIndex = (int*) malloc(n1 * sizeof(int));
    int prevLowest = -1;
    for (int i = 0; i < n1; i++) {
        int currentLowest = INT_MAX;
        for (int j = 0; j < n1; j++) {
            if (I[j] > prevLowest && I[j] < currentLowest) {
                currentLowest = I[j];
                IIndex[j] = i;
            }
        }
        prevLowest = currentLowest;
    }

    // create row permutation matrix of size n1 x n1
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n1; j++) {
            if (IIndex[j] == i) {
                Pr[i * n1 + j] = 1;
            }
            else {
                Pr[i * n1 + j] = 0;
            }
        }
    }

    // create normalized index of J
    int* JIndex = (int*) malloc(n2 * sizeof(int));
    prevLowest = -1;
    for (int i = 0; i < n2; i++) {
        int currentLowest = INT_MAX;
        for (int j = 0; j < n2; j++) {
            if (J[j] > prevLowest && J[j] < currentLowest) {
                currentLowest = J[j];
                JIndex[j] = i;
            }
        }
        prevLowest = currentLowest;
    }

    // create column permutation matrix of size n2 x n2
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n2; j++) {
            if (JIndex[j] == i) {
                Pc[i*n2 + j] = 1;
            }
            else {
                Pc[i*n2 + j] = 0;
            }
        }
    }

    // free memory
    free(IIndex);
    free(JIndex);
}

#endif