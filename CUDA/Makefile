COMPILER = nvcc -lcublas
OPT_FLAGS = -O3 -arch=compute_35

SOURCES = sequentialSpai.cu.h
HELPERS = csc.cu.h constants.cu.h

TEST1 = sequentialTest.cu
EXECUTABLE1 = SequentialTest


default: compile run


compile: $(EXECUTABLE1)

$(EXECUTABLE1): $(TEST1) $(SOURCES) $(HELPERS)
	$(COMPILER) -o $(EXECUTABLE1) $(TEST1)


run: $(EXECUTABLE1)
	./$(EXECUTABLE1)


clean:
	rm -f $(EXECUTABLE1) *.o