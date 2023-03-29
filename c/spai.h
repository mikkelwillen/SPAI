#ifndef SPAI_H
#define SPAI_H

#include <stdio.h>

// A = matrix we want to compute SPAI on
// m, n = size of array
// tol = tolerance
// max_fill_in = constraint for the maximal number of iterations
// s = number of rho_j - the most profitable indices
int spai(float* A, int m, int n, float tol, int max_fill_in, int s);

#endif