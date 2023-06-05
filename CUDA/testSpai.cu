#include <stdio.h>
#include <stdlib.h>
#include "csc.cu.h"
#include "constants.cu.h"
#include "sequentialSpai.cu.h"
#include "sequentialTest.cu.h"
#include "qrBatched.cu.h"
#include "invBatched.cu.h"
#include "permutation.cu.h"
#include "updateQR.cu.h"
#include "cuSOLVERInv.cu.h"


int main(int argc, char** argv) {
    if (argc == 1) {
        initHwd();
        float tolerance = 0.01;
        int s = 1;

        // RUNNING OUR SPAI ONCE
        int n = 500;
        int maxIterations = n - 1;
        float sparsity = 0.1;

        float* test = (float*) malloc(sizeof(float) * n * n);
        CSC* cscTest = createRandomCSC(n, n, sparsity);

        runIdentityTest(cscTest, n, n, sparsity, tolerance, maxIterations, s);

        // // RUNNING CUSOLVER ONCE
        // int n = 1000;
        // float sparsity = 0.3;

        // float* test = (float*) malloc(sizeof(float) * n * n);
        // CSC* cscTest = createRandomCSC(n, n, sparsity);

        // // Create I as all the row indices in cscTest
        // int* I = (int*) malloc(sizeof(int) * n);
        // int* J = (int*) malloc(sizeof(int) * n);
        // for (int i = 0; i < n; i++) {
        //     I[i] = i;
        //     J[i] = i;
        // }

        // float* denseTest = CSCToDense(cscTest, I, J, n, n);
        // sequentialTestCuSOLVER(denseTest, n);

        // int* sizeArray = (int*) malloc(sizeof(int) * 5);
        // sizeArray[0] = 10;
        // sizeArray[1] = 100;
        // sizeArray[2] = 1000;
        // sizeArray[3] = 10000;
        // sizeArray[4] = 100000;

        // float* sparsityArray = (float*) malloc(sizeof(float) * 3);
        // sparsityArray[0] = 0.1;
        // sparsityArray[1] = 0.3;
        // sparsityArray[2] = 0.5;
        // printf("Starting tests\n");

        // for (int i = 2; i < 5; i++) {
        //     for (int j = 0; j < 3; j++) {
        //         int n = sizeArray[i];
        //         float sparsity = sparsityArray[j];
        //         int maxIterations = n - 1;
        //         int success1;
        //         // int success2;

        //         // printf("\n\nStarting test for size: %d and sparsity: %f \n", n, sparsity);
        //         // printf("Running sequential C test\n");
        //         float* test = (float*) malloc(sizeof(float) * n * n);
        //         CSC* cscTest = createRandomCSC(n, n, sparsity);
        //         // success1 = sequentialTest(cscTest, tolerance, maxIterations, s);
        //         // success1 = runIdentityTest(cscTest, n, n, sparsity, tolerance, maxIterations, s);

        //         printf("Running cuSOLVER test\n");
        //         // When we want to test cuSOLVER run this:
        //         // float* test = (float*) malloc(sizeof(float) * n * n);
        //         // CSC* cscTest = createRandomCSC(n, n, sparsity);

        //         // Create I as all the row indices in cscTest
        //         int* I = (int*) malloc(sizeof(int) * n);
        //         int* J = (int*) malloc(sizeof(int) * n);
        //         for (int i = 0; i < n; i++) {
        //             I[i] = i;
        //             J[i] = i;
        //         }

        //         float* denseTest = CSCToDense(cscTest, I, J, n, n);
        //         success1 = sequentialTestCuSOLVER(denseTest, n);

        //         printf("Done with test for size: %d and sparsity: %f \n\n", n, sparsity);  
        //     }
        // }

    
        // float* A = (float*) malloc(sizeof(float) * m * n);
        // float* B = (float*) malloc(sizeof(float) * m * n);
        // float* C = (float*) malloc(sizeof(float) * m * m);
        // float* m4 = (float*) malloc(sizeof(float) * 4 * 4);
        // float* m5 = (float*) malloc(sizeof(float) * 5 * 5);
    
        // for (int i = 0; i < m; i++) {
        //     for (int j = 0; j < n; j++) {
        //         A[i * n + j] = (float) i * n + j;
        //     }
        // }
    
        // C[0] = 20.0; C[1] = 0.0;   C[2] = 0.0; 
        // C[3] = 0.0;  C[4] = 30.0;  C[5] = 11.0; 
        // C[6] = 25.0;  C[7] = 0.0;   C[8] = 10.0;
    
        // B[0] = 20.0; B[1] = 0.0;   B[2] = 0.0; 
        // B[3] = 0.0;  B[4] = 30.0;  B[5] = 10.0; 
        // B[6] = 0.0;  B[7] = 0.0;   B[8] = 0.0; 
        // B[9] = 0.0;  B[10] = 40.0; B[11] = 0.0;

        // m4[0] = 10.0; m4[1] = 10.0; m4[2] = 1.2; m4[3] = 14.0;
        // m4[4] = 0.0; m4[5] = 10.0; m4[6] = 2.0; m4[7] = 0.0;
        // m4[8] = 13.0; m4[9] = 0.0; m4[10] = 5.3; m4[11] = 1.0;
        // m4[12] = 0.0; m4[13] = 5.0; m4[14] = 0.0; m4[15] = 0.0;

        // m4[0] = 10.0; m4[1] = 10.0; m4[2] = 0.0; m4[3] = 14.0;
        // m4[4] = 0.0; m4[5] = 10.0; m4[6] = 2.0; m4[7] = 0.0;
        // m4[8] = 13.0; m4[9] = 0.0; m4[10] = 0.0; m4[11] = 1.0;
        // m4[12] = 0.0; m4[13] = 5.0; m4[14] = 0.0; m4[15] = 0.0;

    
        // struct CSC* cscA = createCSC(A, m, n);
        // struct CSC* cscB = createCSC(B, m, n);
        // struct CSC* cscC = createRandomCSC(m, m, 0.5);
        // struct CSC* cscD = createCSC(C, n, n);
        // struct CSC* cscM4 = createCSC(m4, 4, 4);
    
        // run test
        // runIdentityTest(cscC, m, m, 0.5, tolerance, m - 1, s);
        // runcuSOLVERTest(m4, 4);
        // printf("after running cuSOLVER test\n");
    
        // free memory
        // freeCSC(cscA);
        // freeCSC(cscB);
        // freeCSC(cscC);
        // freeCSC(cscD);

        // free(A);
        // free(B);
        // free(C);   

    } else if (argc == 7) {
        // read args
        printf("hallo?\n");
        int sizeOfMatrix = atoi(argv[1]);
        int numberOfTests = atoi(argv[2]);
        float sparsity = atof(argv[3]);
        float tolerance = atof(argv[4]);
        int maxIterations = atoi(argv[5]);
        int s = atoi(argv[6]);

        printf("sizeOfMatrix: %d\n", sizeOfMatrix);
        printf("numberOfTests: %d\n", numberOfTests);
        printf("sparsity: %f\n", sparsity);
        printf("tolerance: %f\n", tolerance);
        printf("maxIterations: %d\n", maxIterations);
        printf("s: %d\n", s);

        for (int i = 0; i < numberOfTests; i++) {
            CSC* csc = createRandomCSC(sizeOfMatrix, sizeOfMatrix, sparsity);

            sequentialTest(csc, tolerance, maxIterations, s);
        }
    }

    return 0;
}