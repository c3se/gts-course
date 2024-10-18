import numpy as np
from time import time
x = np.ones(1024 * 1024 * 100)  # vector of 100 * 1024^2 eleements, filled with the number 1

# Naive python version
naive_start_time = time()
naive_sum = 0
for e in x:
    naive_sum += e
naive_time_needed = time() - naive_start_time

# NumPy version
numpy_start_time = time()
numpy_sum = x.sum()
numpy_time_needed = time() - numpy_start_time

print(f'Naive time needed: {naive_time_needed:.2e}\n'
      f'Numpy time needed: {numpy_time_needed:.2e}')
