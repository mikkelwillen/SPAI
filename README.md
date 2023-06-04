# SPAI preconditioner on GPU
This project contains the code for the Bachelor Project "Parallel Implementation of the SPAI algorithm" by Caroline Amalie Kierkegaard and Mikkel Willén. The project was supervised by Cosmin Oancea and was submitted in June 2023.

## Project description
This thesis explores the Sparse Approximate Inverse (SPAI) preconditioner, which calculates
an approximate inverse of large and sparse matrices. The SPAI algorithm iteratively constructs
a sparse approximation of the inverse matrix by minimising the norm, column by column, while
preserving the sparsity. It is inherently parallel and thus our aim is to execute it efficiently on
the GPU. The thesis elaborates the original SPAI algorithm by Grote and Huckle by incorporat-
ing theoretical explanations of QR decomposition with Householder reflections and permutations.
We have implemented sequential prototypes in Python and C, followed by a parallel version using
CUDA kernels. Experimental results demonstrate the improved speed of the parallel implemen-
tation compared to Scipy and cuSOLVERS functions for finding the exact inverse.
This thesis shows that the SPAI preconditioner efficiently computes approximate inverses,
e.g. for large Hessian matrices in the Newton method, and the parallel GPU implementation
outperforms the sequential CPU execution.

### Accessing DIKU-server
```bash
$ ssh -l <ku_id> futharkhpa01fl.unicph.domain
$ ssh -l <ku_id> futharkhpa02fl.unicph.domain
$ ssh -l <ku_id> futharkhpa03fl.unicph.domain
```

### Running the code
```bash
# Running the tests of the python implementation
$ python3 /python/test.py

# Running the tests of the sequential C implementation
$ make -C CUDA

# Running the tests of the parallel CUDA implementation
$ make -C CUDAparallel
```