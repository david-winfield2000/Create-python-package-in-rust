import time

import numpy as np
from my_rust_library import matmul


# Python implementation of matrix multiplication
def matmul_python(a, b):
    n = len(a)
    m = len(b[0])
    k = len(b)
    result = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            for l in range(k):
                result[i][j] += a[i][l] * b[l][j]

    return result


# Generate random matrices
def generate_matrix(rows, cols):
    return np.random.randint(0, 10, size=(rows, cols)).tolist()


# Benchmark
rust_times = []
python_times = []


def benchmark(func, a, b):
    start_time = time.time()
    func(a, b)
    end_time = time.time()
    return end_time - start_time


for _ in range(100):
    matrix_a = generate_matrix(200, 300)
    matrix_b = generate_matrix(300, 200)

    rust_times.append(benchmark(matmul, matrix_a, matrix_b))
    python_times.append(benchmark(matmul_python, matrix_a, matrix_b))


# Calculate average runtime for each function
python_average = sum(python_times) / len(python_times)
rust_average = sum(rust_times) / len(rust_times)

print(f"Python average runtime: {python_average:.6f} seconds")
print(f"Rust average runtime: {rust_average:.6f} seconds")


# Print how many times Rust was faster than Python
rust_better = 0
python_better = 0
for i in range(len(python_times)):
    if python_times[i] > rust_times[i]:
        rust_better += 1
    else:
        python_better += 1

print(f"Rust was faster than Python {rust_better} times")
print(f"Python was faster than Rust {python_better} times")
