# PILAA
Parallel Implementation of Linear Algebra Algorithms

Bachelor Project by Caroline Amalie Kierkegaard and Mikkel Willén

Accessing DIKU-server
```bash
$ ssh -l <ku_id> futharkhpa01fl.unicph.domain
$ ssh -l <ku_id> futharkhpa02fl.unicph.domain
$ ssh -l <ku_id> futharkhpa03fl.unicph.domain
```

Running the code
```bash
# Running the tests of the python implementation
$ python3 /python/test.py

# Running the tests of the sequential C implementation
$ make -C CUDA

# Running the tests of the parallel CUDA implementation
$ make -C CUDAparallel
```