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
    
        int* sizeArray = (int*) malloc(sizeof(int) * 15);
        sizeArray[0] = 10;
        sizeArray[1] = 100;
        sizeArray[2] = 200;
        sizeArray[3] = 300;
        sizeArray[4] = 400;
        sizeArray[5] = 500;
        sizeArray[6] = 1000;
        sizeArray[7] = 1500;
        sizeArray[8] = 2000;
        sizeArray[9] = 2500;
        sizeArray[10] = 3000;
        sizeArray[11] = 3500;
        sizeArray[12] = 4000;
        sizeArray[13] = 4500;
        sizeArray[14] = 5000;

        // for (int i = 13; i < 15; i++) {
        //     int size = sizeArray[i];
        //     runMatrixMultiplicationTest(size);
        // }
        
        for (int i = 8; i < 15; i++) {
            int size = sizeArray[i];
            runSetSecondMatrixTest(size);
        }

        for (int i = 0; i < 15; i++) {
            int size = sizeArray[i];
            runCSCToBatchedTest(size);
        }
    }
}