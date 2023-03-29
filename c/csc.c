#include "csc.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

struct CSC {
    int m;
    int n;
    int countNonZero;
    int* offset;
    float* flatData;
    int* flatColsIndex;
}; 

struct CSC* createCSC(float* A, int m, int n) {
    struct CSC* csc = malloc(sizeof(struct CSC));
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

    csc->offset = malloc(sizeof(int) * (n + 1));
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

    csc->flatData = malloc(sizeof(float) * csc->countNonZero);
    int index = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (A[i * n + j] != 0.0) {
                csc->flatData[index] = A[i * n + j];
                index++;
            }
        }
    }

    csc->flatColsIndex = malloc(sizeof(int) * csc->countNonZero);
    index = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (A[i * n + j] != 0.0) {
                csc->flatColsIndex[index] = i;
                index++;
            }
        }
    }

    return csc;
}

struct CSC* createDiagonalCSC(int m, int n){
    struct CSC* csc = malloc(sizeof(struct CSC));
    csc->m = n;
    csc->n = m;
    csc->countNonZero = n;

    csc->offset = malloc(sizeof(int) * (csc->n + 1));
    for (int j = 0; j < m + 1; j++) {
        if (j < n)
        csc->offset[j] = j;
    }

    csc->flatData = malloc(sizeof(float) * csc->countNonZero);
    for (int j = 0; j < n; j++) {
        csc->flatData[j] = 1.0;
    }

    csc->flatColsIndex = malloc(sizeof(int) * csc->countNonZero);
    for (int i = 0; i < n; i++) {
        csc->flatColsIndex[i] = i;
    }

    return csc;
}

void freeCSC(struct CSC* csc) {
    free(csc->offset);
    free(csc->flatData);
    free(csc->flatColsIndex);
}

void printCSC(struct CSC* csc) {
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
    printf("csc->flatColsIndex: ");
    for (int i = 0; i < csc->countNonZero; i++) {
        printf("%d ", csc->flatColsIndex[i]);
    }
    printf("\n");
}