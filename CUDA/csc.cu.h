#ifndef CSC_H
#define CSC_H

#include <stdio.h>

// A struct representing a sparse matrix
// int m;
// int n;
// int countNonZero;
// int* offset;
// float* flatData;
// int* flatRowIndex;
typedef struct CSC {
    int m;
    int n;
    int countNonZero;
    int* offset;
    float* flatData;
    int* flatRowIndex;
} CSC;

// Function for creating a compressed sparse column matrix
// A = Dense matrix
// m = number of rows
// n = number of columns
CSC* createCSC(float* A, int m, int n) {
    CSC* csc = (CSC*) malloc(sizeof(CSC));
    csc->m = m;
    csc->n = n;

    int count = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (A[i * n + j] != 0.0) {
                count++;
            }
        }
    }
    csc->countNonZero = count;

    csc->offset = (int*) malloc(sizeof(int) * (n + 1));
    int scan = 0;
    for (int j = 0; j < n; j++) {
        csc->offset[j] = scan;
        for (int i = 0; i < m; i++) {
            if (A[i * n + j] != 0.0) {
                scan++;
            }
        }
    }
    csc->offset[n] = scan;

    csc->flatData = (float*) malloc(sizeof(float) * csc->countNonZero);
    int index = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (A[i * n + j] != 0.0) {
                csc->flatData[index] = A[i * n + j];
                index++;
            }
        }
    }

    csc->flatRowIndex = (int*) malloc(sizeof(int) * csc->countNonZero);
    index = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (A[i * n + j] != 0.0) {
                csc->flatRowIndex[index] = i;
                index++;
            }
        }
    }

    return csc;
}

// Function for creating a compressed sparse column matrix with 1's in the diagonal
// This is made specifically for creating M in SPAI 
// m = number of rows
// n = number of columns
CSC* createDiagonalCSC(int m, int n){
    CSC* csc = (CSC*) malloc(sizeof(CSC));
    csc->m = n;
    csc->n = m;
    csc->countNonZero = n;

    csc->offset = (int*) malloc(sizeof(int) * (csc->n + 1));
    for (int j = 0; j < csc->n + 1; j++) {
        if (j < csc->m) {
            csc->offset[j] = j;
        } else {
            csc->offset[j] = csc->m;
        }
    }

    csc->flatData = (float*) malloc(sizeof(float) * csc->countNonZero);
    for (int j = 0; j < n; j++) {
        csc->flatData[j] = 1.0;
    }

    csc->flatRowIndex = (int*) malloc(sizeof(int) * csc->countNonZero);
    for (int i = 0; i < n; i++) {
        csc->flatRowIndex[i] = i;
    }

    return csc;
}

// Function for creating a random CSC with specified sparsity
// m = number of rows
// n = number of columns
// sparsity = The sparsity of the matrix. Should be a number between 0.0-1.0. 
CSC* createRandomCSC(int m, int n, float sparsity){
    float* M = (float*) malloc(sizeof(float) * m * n);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float x = ((float) rand() / (float) (RAND_MAX));
            if (x < sparsity) {
                float y = ((float) rand() / (float) (RAND_MAX)) * 100.0;
                M[i * n + j] = y;
            } else {
                M[i * n + j] = 0.0;
            }
        }
    }

    CSC* A = createCSC(M, m, n);

    return A;
}

// Frees all the elements of a CSC struct
void freeCSC(CSC* csc) {
    free(csc->offset);
    free(csc->flatData);
    free(csc->flatRowIndex);
}

// Prints all the elements of a CSC struct
void printCSC(CSC* csc) {
    printf("\n--------Printing CSC data--------\n");
    printf("csc->m: %d\n", csc->m);
    printf("csc->n: %d\n", csc->n);
    printf("csc->countNonZero: %d\n", csc->countNonZero);
    printf("csc->offset: ");
    for (int i = 0; i < csc->n + 1; i++){
        printf("%d ", csc->offset[i]);
    }
    printf("\n");
    printf("csc->flatData:");
    for (int i = 0; i < csc->countNonZero; i++) {
        printf("%f ", csc->flatData[i]);
    }
    printf("\n");
    printf("csc->flatRowIndex: ");
    for (int i = 0; i < csc->countNonZero; i++) {
        printf("%d ", csc->flatRowIndex[i]);
    }
    printf("\n");
}

#endif