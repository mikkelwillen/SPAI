#include <stdio.h>
#include <stdlib.h>
#include "csc.cu.h"
#include "constants.cu.h"
#include "sequentialSpai.cu.h"
#include "sequentialTest.cu.h"
#include "qrBatched.cu.h"


int main() {
    initHwd();
    int m = 4;
    int n = 3;



    float* A = (float*) malloc(sizeof(float) * m * n);
    float* B = (float*) malloc(sizeof(float) * m * n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (float) i * n + j;
        }
    }

    B[0] = 20.0; B[1] = 0.0;   B[2] = 0.0; 
    B[3] = 0.0;  B[4] = 30.0;  B[5] = 10.0; 
    B[6] = 0.0;  B[7] = 0.0;   B[8] = 0.0; 
    B[9] = 0.0;  B[10] = 40.0; B[11] = 0.0;

    struct CSC* cscA = createCSC(A, m, n);
    struct CSC* cscB = createCSC(B, m, n);
    struct CSC* cscC = createRandomCSC(10, 10, 0.2);
    // struct CSC* cscDia = createDiagonalCSC(m, n);
    // printCSC(cscA);
    // printCSC(cscB);
    // printCSC(cscDia);
    // struct CSC* M = sequentialSpai(cscB, 0.01, 5, 1);
    // printCSC(M);

    // sequentialTest(cscB);

    float* Q = (float*) malloc(m * m * sizeof(float));
    float* R = (float*) malloc(sizeof(float) * m * n);

    qrBatched(B, m, n, Q, R);

    // compute QR
    float* QR = (float*) malloc(sizeof(float) * m * n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++){
            QR[i * n + j] = 0.0;
            for (int k = 0; k < m; k++) {
                QR[i * n + j] += Q[i * m + k] * R[k * n + j];
            }
        }
    }
    
    // print QR
    printf("QR:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++){
            printf("%f ", QR[i * n + j]);
        }
        printf("\n");
    }
}