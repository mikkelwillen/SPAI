COMPILER = nvcc -lcublas -lcusolver
OPT_FLAGS = -O3 -arch=compute_35

SOURCES = parallelSpai.cu.h parallelTest.cu.h
HELPERS = csc.cu.h constants.cu.h qrBatched.cu.h invBatched.cu.h permutation.cu.h updateQR.cu.h LSProblem.cu.h helperKernels.cu.h

TEST1 = testSpai.cu
EXECUTABLE1 = testSpai


default: compile run


compile: $(EXECUTABLE1)

$(EXECUTABLE1): $(TEST1) $(SOURCES) $(HELPERS)
	$(COMPILER) -o $(EXECUTABLE1) $(TEST1)


run: $(EXECUTABLE1)
	./$(EXECUTABLE1)

clean:
	rm -f $(EXECUTABLE1) *.o