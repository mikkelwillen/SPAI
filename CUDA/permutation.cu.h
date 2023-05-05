#ifndef PERMUTATION_H
#define PERMUTATION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    // print IIndex
    printf("IIndex:\n");
    for (int i = 0; i < n1; i++) {
        printf("%d ", IIndex[i]);
    }

    // create row permutation matrix of size n1 x n1
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n1; j++) {
            if (IIndex[j] == i) {
                Pr[i*n1 + j] = 1;
            }
            else {
                Pr[i*n1 + j] = 0;
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