COMPILER = nvcc -lcublas -lcusolver
OPT_FLAGS = -O3 -arch=compute_35

SOURCES = sequentialSpai.cu.h sequentialTest.cu.h
HELPERS = csc.cu.h constants.cu.h qrBatched.cu.h invBatched.cu.h permutation.cu.h updateQR.cu.h LSProblem.cu.h singular.cu.h cuSOLVERInv.cu.h

TEST1 = testSpai.cu
TEST2 = runTests.cu
EXECUTABLE1 = testSpai
EXECUTABLE2 = runTests

default: compile run

compile: $(EXECUTABLE1)
$(EXECUTABLE1): $(TEST1) $(SOURCES) $(HELPERS)
	$(COMPILER) -o $(EXECUTABLE1) $(TEST1)

run: $(EXECUTABLE1)
	./$(EXECUTABLE1)

test: $(EXECUTABLE2)
$(EXECUTABLE2): $(TEST2) $(SOURCES) $(HELPERS)
	$(COMPILER) -o $(EXECUTABLE2) $(TEST2)
	./$(EXECUTABLE2)

clean:
	rm -f $(EXECUTABLE1) *.o