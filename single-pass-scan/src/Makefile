COMPILER?=nvcc
OPT_FLAGS?=-O3 -arch=compute_35
LDFLAGS?=-lOpenCL

TEST_SPS=test-sps

ARRAY_SZ=50003565

.PHONY: clean all run

default: compile v1

compile: $(TEST_SPS)

$(TEST_SPS): testSPS.cu hostSkel.cu.h sPSKernels.cu.h rTSKernels.cu.h constants.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_SPS) testSPS.cu

run: $(TEST_SPS)
	./$(TEST_SPS) 50003565 128

v0: $(TEST_SPS)
	./$(TEST_SPS) $(ARRAY_SZ) 128 0

v1: $(TEST_SPS)
	./$(TEST_SPS) $(ARRAY_SZ) 128 1

v2: $(TEST_SPS)
	./$(TEST_SPS) $(ARRAY_SZ) 128 2

v3: $(TEST_SPS)
	./$(TEST_SPS) $(ARRAY_SZ) 128 3

clean:
	rm -f $(TEST_SPS)
