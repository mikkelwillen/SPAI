import numpy as np
import matplotlib.pyplot as plt

matrixMul = ["10 x 10", "100 x 100", "200 x 200", "300 x 300", "400 x 400", "500 x 500", "1000 x 1000", "1500 x 1500", "2000 x 2000", "2500 x 2500", "3000 x 3000", "3500 x 3500", "4000 x 4000", "4500 x 4500", "5000 x 5000"]
sequential = [0.000011, 0.006522, 0.050982, 0.172691, 0.382062]
kernel = [0.000008, 0.000021, 0.000033, 0.000047, 0.000098]

plt.plot(matrixMul, sequential, linestyle="-", color="blue", label="Sequential")
plt.plot(matrixMul, kernel, linestyle="-", color="Orange", label="Parallel")
plt.xlabel("Matrix Size")
plt.ylabel("Time (s)")
plt.title("Speed test Matrix Multiplication")
plt.legend()
plt.show()

peak = [0.01, 1.25, 31.25, 100, 100]
max = [100, 100, 100, 100, 100]
plt.plot(matrixMul, peak, linestyle="-", color="blue", label="% \of peak performance")
plt.plot(matrixMul, max, linestyle="-", color="Orange", label="100%")
plt.xlabel("Matrix Size")
plt.ylabel("% of peak performance")
plt.title("%-utilization of hardware with Matrix Multiplication")
plt.legend()
plt.show()

setSecond = ["10 x 10", "100 x 100", "500 x 500", "1000 x 1000", "5000 x 5000"]
sequential = [0.000007, 0.000083, 0.00193, 0.007267,0.173633]
kernel = [0.000009, 0.000011, 0.000011, 0.00002, 0.000183]
plt.plot(setSecond, sequential, linestyle="-", color="blue", label="Sequential")
plt.plot(setSecond, kernel, linestyle="-", color="Orange", label="Parallel")
plt.xlabel("Matrix size")
plt.ylabel("Time (s)")
plt.title("Speed test Set Second Matrix")
plt.legend()
plt.show()

peak = [0, 0.45, 11.36, 25, 68.31]
max = [100, 100, 100, 100, 100]
plt.plot(setSecond, peak, linestyle="-", color="blue", label="% \of peak performance")
plt.plot(setSecond, max, linestyle="-", color="Orange", label="100%")
plt.xlabel("Matrix size")
plt.ylabel("% of peak performance")
plt.title("%-utilization of hardware with Set Second Matrix")
plt.legend()
plt.show()

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