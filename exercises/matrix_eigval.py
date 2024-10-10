""" This code finds the largest eigenvalue in a matrix. """

# DO NOT MODIFY
import numpy as np
from math import sqrt
from time import time
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



""" The code may run too slowly to profile with N = 20000. Try setting N to a smaller
value first, and then change back to `20000` once you have optimized the code."""
N = 20000

""" READ BUT DO NOT MODFY THE CODE PAST THIS LINE, MODIFY INSIDE THE FUNCTIONS """
randgen = np.random.default_rng(seed=135893)
A = randgen.normal(2, 5, (N, N)).clip(0, None)
b = randgen.normal(0, 1, (N, 1))

norm = get_vector_norm(b)
b = divide_vector_by_scalar(b, norm)
old_norm = 0
start_time = time()
mv_time = 0.
vn_time = 0.
dv_time = 0.
for i in range(50):
    relative_difference = (abs(norm - old_norm) / (0.5 * (norm + old_norm)))
    print(f'Iteration {i}, eigenvalue approximation: {norm}, '
          f'relative difference: {relative_difference}')
    if relative_difference < 1e-8:
        print('Convergence!')
        break
    old_norm = norm


    temp_time = time()
    b = matrix_vector_multiplication(A, b)
    mv_time += time() - temp_time

    temp_time = time()
    norm = get_vector_norm(b)
    vn_time += time() - temp_time

    temp_time = time()
    b = divide_vector_by_scalar(b, norm)
    dv_time += time() - temp_time

print(
    f'Matrix-Vector time: {mv_time:.2e}\nVector Norm time: {vn_time:.2e}\nDivide-vector time: {dv_time:.2e}')
convergence_time = time() - start_time
print(f'Time to convergence: {convergence_time:.2f} seconds.')

if convergence_time > 0.5:
    raise ValueError('It should take less than half a'
                     f'second to run this optimization, but it took {convergence_time:2f}!')

if not np.isclose(norm, 63035.05409):
    raise ValueError(f'The computed eigenvalue is not correct! {norm:.5f} is not close to 6290.72419')
