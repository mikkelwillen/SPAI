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
struct CSC {
    int m;
    int n;
    int countNonZero;
    int* offset;
    float* flatData;
    int* flatRowIndex;
};

// Function for creating a compressed sparse column matrix
// A = Dense matrix
// m = number of rows
// n = number of columns
struct CSC* createCSC(float* A, int m, int n);

// Function for creating a compressed sparse column matrix with 1's in the diagonal
// This is made specifically for creating M in SPAI 
// m = number of rows
// n = number of columns
struct CSC* createDiagonalCSC(int m, int n);

// Frees all the elements of a CSC struct
void freeCSC(struct CSC* csc);

// Prints all the elements of a CSC struct
void printCSC(struct CSC* csc);

#endif