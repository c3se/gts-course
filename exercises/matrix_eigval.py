import numpy as np
from math import sqrt
from time import time

randgen = np.random.default_rng(seed=135893)

A = randgen.normal(2, 5, (2000, 2000)).clip(0, None)
b = randgen.normal(0, 1, (2000, 1))

"""These three functions constitute the computational load of this
   script. Which function requires the most time to carry out?
   How can we measure that? How can we optimize it?
   When you are done, the loop should run in
   less then one second."""


def get_vector_norm(v):
    norm = 0
    for j in range(v.size):
        norm += v[j] * v[j]
    norm = sqrt(norm)
    return norm


def divide_vector_by_scalar(v, scalar):
    x = np.zeros_like(v)
    for j in range(x.size):
        x[j] = v[j] / scalar
    return x


def matrix_vector_multiplication(M, v):
    x = np.zeros_like(v)
    for j in range(M.shape[0]):
        for k in range(M.shape[1]):
            x[j] += M[j, k] * v[k]

    return x


norm = get_vector_norm(b)
b = divide_vector_by_scalar(b, norm)
old_norm = 0
start_time = time()
for i in range(20):
    relative_difference = (abs(norm - old_norm) / (0.5 * (norm + old_norm)))
    print(f'Iteration {i}, eigenvalue approximation: {norm}, '
          f'relative difference: {relative_difference}')
    if relative_difference < 1e-8:
        print('Convergence!')
        break
    old_norm = norm


    b = matrix_vector_multiplication(A, b)

    norm = get_vector_norm(b)

    b = divide_vector_by_scalar(b, norm)
convergence_time = time() - start_time
print(f'Time to convergence: {convergence_time:.2f} seconds.')
if convergence_time > 0.5:
    raise ValueError('It should take less than half a second to run this optimization!')
