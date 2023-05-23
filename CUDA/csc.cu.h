#ifndef CSC_H
#define CSC_H

#include <stdio.h>
#include <time.h>

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

// Frees all the elements of a CSC struct
// csc = The CSC struct to be freed
void freeCSC(CSC* csc) {
    free(csc->offset);
    free(csc->flatData);
    free(csc->flatRowIndex);
}

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
    
    time_t t;
    srand((unsigned) time(&t));
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float x = ((float) rand() / (float) (RAND_MAX));
            if (x < sparsity) {
                float y = ((float) rand() / (float) (RAND_MAX)) * 100.0 + (float) rand() / (float) (RAND_MAX);
                M[i * n + j] = y;
            } else {
                M[i * n + j] = 0.0;
            }
        }
    }

    CSC* A = createCSC(M, m, n);
    free(M);
    
    return A;
}

// Function for updating a CSC matrix with a new value
// csc = The CSC matrix
// newValues = The new values to be inserted
// k = the column index of the new values
// J = the row indices of the new values
// n2 = the number of new values
CSC* updateKthColumnCSC(CSC* A, float* newVaules, int k, int* J, int n2) {
    CSC* newA = (CSC*) malloc(sizeof(CSC));
    newA->m = A->m;
    newA->n = A->n;
    
    // Compute the new number of nonzeros
    int deltaNonzeros = 0;
    for (int i = 0; i < n2; i++) {
        if (newVaules[i] != 0.0) {
            deltaNonzeros++;
        }
    }
    deltaNonzeros -= A->offset[k + 1] - A->offset[k];

    // set the new number of nonzeros
    newA->countNonZero = A->countNonZero + deltaNonzeros;

    // Malloc space for the new offset array
    newA->offset = (int*) malloc(sizeof(int) * (A->n + 1));

    // Copy the offset values before k
    for (int i = 0; i < k + 1; i++) {
        newA->offset[i] = A->offset[i];
    }

    // Compute the new offset values for k and onwards
    for (int i = k + 1; i < A->n + 1; i++) {
        newA->offset[i] = A->offset[i] + deltaNonzeros;
    }

    // Malloc space
    newA->flatData = (float*) malloc(sizeof(float) * newA->countNonZero);
    newA->flatRowIndex = (int*) malloc(sizeof(int) * newA->countNonZero);

    // Copy the old flatData and flatRowIndex values before k
    for (int i = 0; i < A->offset[k] + 1; i++) {
        newA->flatData[i] = A->flatData[i];
        newA->flatRowIndex[i] = A->flatRowIndex[i];
    }

    // insert the new values into the flatData and flatRowIndex from k
    int index = 0;
    for (int i = A->offset[k]; i < A->offset[k] + deltaNonzeros + 1; i++) {
        newA->flatData[i] = newVaules[index];
        newA->flatRowIndex[i] = J[index];
        index++;
    }

    // Copy the old flatData and flatRowIndex values after k
    for (int i = newA->offset[k + 1]; i < newA->countNonZero; i++) {
        newA->flatData[i] = A->flatData[i - deltaNonzeros];
        newA->flatRowIndex[i] = A->flatRowIndex[i - deltaNonzeros];
    }

    return newA;
}

// Function for tranforming a CSC matrix to a dense matrix with specific rows and columns
// csc = The CSC matrix
// I = The row indices of the dense matrix
// J = The column indices of the dense matrix
// n1 = The number of rows in the dense matrix
// n2 = The number of columns in the dense matrix
float* CSCToDense(CSC* csc, int* I, int* J, int n1, int n2) {
    float* dense = (float*) calloc(n1 * n2, sizeof(float));

    for(int j = 0; j < n2; j++) {
        for (int i = 0; i < n1; i++) {
            for (int l = csc->offset[J[j]]; l < csc->offset[J[j] + 1]; l++) {
                if (I[i] == csc->flatRowIndex[l]) {
                    dense[j * n2 + i] = csc->flatData[l];
                }
            }
        }
    }

    return dense;
}

// Function for multiplying 2 CSC matrices
// A = The first CSC matrix
// B = The second CSC matrix
CSC* multiplyCSC(CSC* A, CSC* B) {
    CSC* C = (CSC*) malloc(sizeof(CSC));
    C->m = A->m;
    C->n = B->n;
    C->countNonZero = 0;
    C->offset = (int*) malloc(sizeof(int) * (C->n + 1));
    C->offset[0] = 0;
    C->flatData = (float*) malloc(sizeof(float) * (A->countNonZero + B->countNonZero));
    C->flatRowIndex = (int*) malloc(sizeof(int) * (A->countNonZero + B->countNonZero));

    for (int j = 0; j < C->n; j++) {
        int countNonZero = 0;
        for (int i = 0; i < C->m; i++) {
            float sum = 0.0;
            for (int l = A->offset[i]; l < A->offset[i + 1]; l++) {
                for (int k = B->offset[j]; k < B->offset[j + 1]; k++) {
                    if (A->flatRowIndex[l] == B->flatRowIndex[k]) {
                        sum += A->flatData[l] * B->flatData[k];
                    }
                }
            }
            if (sum != 0.0) {
                C->flatData[C->countNonZero] = sum;
                C->flatRowIndex[C->countNonZero] = i;
                C->countNonZero++;
                countNonZero++;
            }
        }
        C->offset[j + 1] = C->offset[j] + countNonZero;
    }

    return C;
}

// Prints all the elements of a CSC struct
void printCSC(CSC* csc) {
    printf("\n\n--------Printing CSC data--------\n");
    printf("csc->m: %d\n", csc->m);
    printf("csc->n: %d\n", csc->n);
    printf("csc->countNonZero: %d\n", csc->countNonZero);
    printf("csc->offset: ");
    for (int i = 0; i < csc->n + 1; i++){
        printf("%d ", csc->offset[i]);
    }
    printf("\n");
    printf("csc->flatData: ");
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