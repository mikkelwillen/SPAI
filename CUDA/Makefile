COMPILER = nvcc -O3 -arch=compute_35

SOURCES = sequentialSpai.cu
HELPERS = csc.cu

TEST1 = sequentialTest.cu
EXECUTABLE1 = SequentialTest


default: compile run

.cu.o: $(SOURCES) $(HELPERS)
	$(COMPILER) -c $@ $<


compile: $(EXECUTABLE1)

$(EXECUTABLE1): $(TEST1) $(SOURCES) $(HELPERS)
	$(COMPILER) -o $(EXECUTABLE1) $(TEST1)


run: $(EXECUTABLE1)
	./$(EXECUTABLE1)


clean:
	rm -f $(EXECUTABLE1)