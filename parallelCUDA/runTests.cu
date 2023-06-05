#include <stdio.h>
#include <stdlib.h>
#include "csc.cu.h"
#include "constants.cu.h"
#include "parallelSpai.cu.h"
#include "parallelTest.cu.h"
#include "qrBatched.cu.h"
#include "invBatched.cu.h"
#include "permutation.cu.h"
#include "updateQR.cu.h"
#include "kernelTests.cu.h"

int main(int argc, char** argv) {
    if (argc == 1) {
        initHwd();
    
        int* sizeArray = (int*) malloc(sizeof(int) * 5);
        sizeArray[0] = 10;
        sizeArray[1] = 100;
        sizeArray[2] = 500
        sizeArray[3] = 1000;
        sizeArray[4] = 5000;

        for (int i = 0; i < 5; i++) {
            runMatrixMultiplicationTest(i);
        }
        
        for (int i = 0; i < 4; i++) {
            runSetSecondMatrix(i);
        }
    }
}