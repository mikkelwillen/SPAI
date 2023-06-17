import numpy as np
import matplotlib.pyplot as plt

# Parallel kernel tests vs sequential tests
matrixSize = ["10 x 10", "100 x 100", "200 x 200", "300 x 300", "400 x 400", "500 x 500", "1000 x 1000", "1500 x 1500", "2000 x 2000", "2500 x 2500", "3000 x 3000", "3500 x 3500", "4000 x 4000", "4500 x 4500", "5000 x 5000"]
max = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

# Matrix multiplication
sequential = [0.000011, 0.006522, 0.050982, 0.172691, 0.382062, 0.487120, 3.618151, 12.158891, 31.119870, 174.973655, 218.114464, 386.017526, 395.417521, 749.014510, 1029.577479]
kernel = [0.000008, 0.000021, 0.000033, 0.000047, 0.000098, 0.000181, 0.001314, 0.004671, 0.010957, 0.021899, 0.032665, 0.055819, 0.072808, 0.095990, 0.120646]
plt.plot(matrixSize, sequential, linestyle="-", color="blue", label="Sequential")
plt.plot(matrixSize, kernel, linestyle="-", color="Orange", label="Parallel")
plt.xlabel("Matrix Size")
plt.ylabel("Time (s)")
plt.title("Speed test Matrix Multiplication")
plt.grid(axis = 'y')
plt.legend()
plt.show()

peak = [0.01, 3.81, 9.70, 15.32, 13.06, 11.05, 6.09, 3.85, 2.92, 2.28, 2.20, 1.76, 1.76, 1.69, 1.66]
plt.plot(matrixSize, peak, linestyle="-", color="blue", label="% \of peak performance")
plt.xlabel("Matrix Size")
plt.ylabel("% of peak performance")
plt.title("%-utilization of hardware with Matrix Multiplication")
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.grid(axis = 'y')
plt.legend()
plt.show()

# Set second matrix
sequential = [0.000005, 0.000074, 0.000273, 0.000644, 0.001121, 0.001733, 0.007266, 0.015941]
kernel = [0.000008, 0.000008, 0.000008, 0.000008, 0.000009, 0.000010, 0.000019, 0.000031]
plt.plot(matrixSize, sequential, linestyle="-", color="blue", label="Sequential")
plt.plot(matrixSize, kernel, linestyle="-", color="Orange", label="Parallel")
plt.xlabel("Matrix size")
plt.ylabel("Time (s)")
plt.title("Speed test Set Second Matrix")
plt.grid(axis = 'y')
plt.legend()
plt.show()

peak = [0.1, 10.0, 40.0, 90.0, 142.22, 200.00, 421.05, 580.65]
plt.plot(matrixSize, peak, linestyle="-", color="blue", label="% \of peak performance")
plt.plot(matrixSize, max, linestyle="-", color="Orange", label="100%")
plt.xlabel("Matrix size")
plt.ylabel("% of peak performance")
plt.title("%-utilization of hardware with Set Second Matrix")
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.grid(axis = 'y')
plt.legend()
plt.show()

# CSC to Batched dense
sequential = [0.000011, 0.006513, 0.050452, 0.168493,0.374001, 0.458631, 3.530385, 11.977657, 28.161084, 56.214897, 96.528684, 153.856608, 227.295646, 325.104471]
kernel = [0.000010, 0.000010, 0.000010, 0.000010, 0.000011, 0.000010, 0.000028, 0.000012, 0.000017, 0.000023, 0.000188, 0.000038, 0.000400, ]
plt.plot(matrixSize, sequential, linestyle="-", color="blue", label="Sequential")
plt.plot(matrixSize, kernel, linestyle="-", color="Orange", label="Parallel")
plt.xlabel("Matrix size")
plt.ylabel("Time (s)")
plt.title("Speed test CSC to Batched Dense Matrix")
plt.grid(axis = 'y')
plt.legend()
plt.show()

peak = [0.8, 8.0, 32.00, 72.00, 116.36, 200.00, 285.75, 1500.00, 1882.35, 2173.91, 382.98, 2578.95, 2723.40]
plt.plot(matrixSize, peak, linestyle="-", color="blue", label="% \of peak performance")
plt.plot(matrixSize, max, linestyle="-", color="Orange", label="100%")
plt.xlabel("Matrix size")
plt.ylabel("% of peak performance")
plt.title("%-utilization of hardware with CSCToBatchedDenseMatrices")
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.grid(axis = 'y')
plt.legend()
plt.show()


# Sequential Tests
size = ["10 x 10", "100 x 100", "1000 x 1000", "10000 x 10000"]
scipy = [None, 0.014879, 0.735263, 919.821114]
sequentialPy = [None, 13.778171, 7391.510661, None]
CuSolver = [None, 0.002633, 0.012987, 3.350172]
sequentialC = [None, 69.074035, 5023.625312, None]
plt.plot(size, scipy, linestyle="-", color="blue", label="Scipy")
plt.plot(size, sequentialPy, linestyle="-", color="Orange", label="Sequential SPAI in python")
plt.plot(size, CuSolver, linestyle="-", color="green", label="CuSolver")
plt.plot(size, sequentialC, linestyle="-", color="red", label="Sequential SPAI in C")
plt.xlabel("Matrix size")
plt.ylabel("Time (s)")
plt.title("Speed test Inverse Matrix \nSparsity = 0.1, tolerance = 0.01, max iterations = n - 1")
plt.legend()
plt.show()

scipy = [0.002563, 0.016799, 0.740422, 806.18316]
sequentialPy = [0.106887, 12.471273, 6602.605517, None]
CuSolver = [0.002086, 0.002700, 0.013012, 3.498521]
sequentialC = [0.018003, 92.412611, 5208.116384, None]
plt.plot(size, scipy, linestyle="-", color="blue", label="Scipy")
plt.plot(size, sequentialPy, linestyle="-", color="Orange", label="Sequential SPAI in python")
plt.plot(size, CuSolver, linestyle="-", color="green", label="CuSolver")
plt.plot(size, sequentialC, linestyle="-", color="red", label="Sequential SPAI in C")
plt.xlabel("Matrix size")
plt.ylabel("Time (s)")
plt.title("Speed test Inverse Matrix  \nSparsity = 0.3, tolerance = 0.01, max iterations = 100")
plt.legend()
plt.show()

scipy = [0.001761, 0.017183, 0.7267112, 811.357949]
sequentialPy = [0.078199, 12.866873, 7125.50547, None]
CuSolver = [0.002052, 0.002724, 0.013023, 3.951132]
sequentialC = [0.021173, 103.863722, 6198.732912, None]
plt.plot(size, scipy, linestyle="-", color="blue", label="Scipy")
plt.plot(size, sequentialPy, linestyle="-", color="Orange", label="Sequential SPAI in python")
plt.plot(size, CuSolver, linestyle="-", color="green", label="CuSolver")
plt.plot(size, sequentialC, linestyle="-", color="red", label="Sequential SPAI in C")
plt.xlabel("Matrix size")
plt.ylabel("Time (s)")
plt.title("Speed test Inverse Matrix \nSparsity = 0.5, tolerance = 0.01, max iterations = 100")
plt.legend()
plt.show()