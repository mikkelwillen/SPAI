COMPILER?=nvcc
OPT_FLAGS?=-O3 -arch=compute_35
LDFLAGS?=-lOpenCL

TEST_SPAI=test-spai

ARRAY_SZ=50003565

.PHONY: clean all run

default: compile v1

compile: $(TEST_SPAI)

$(TEST_SPAI): tests.cu hostSkel.cu.h spaiKernels.cu.h qrKernels.cu.h constants.cu.h
	$(COMPILER) $(OPT_FLAGS) -o $(TEST_SPAI) tests.cu


clean:
	rm -f $(TEST_SPS)
