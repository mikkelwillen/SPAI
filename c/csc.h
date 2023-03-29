#ifndef CSC_H
#define CSC_H

#include <stdio.h>

// A struct representing a sparse matrix
struct CSC;

// Function for creating a compressed sparse column matrix
// A = Dense matrix
// m = number of rows
// n = number of columns
struct CSC* createCSC(float* A, int m, int n);

void freeCSC(struct CSC* csc);

void printCSC(struct CSC* csc);

#endif