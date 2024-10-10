""" This code finds the largest eigenvalue in a matrix. """

# DO NOT MODIFY
import cupy as cp
import  numpy as np
from math import sqrt
from time import time
"""These three functions constitute the computational load of this
   script. Which function requires the most time to carry out?
   How can we measure that? How can we optimize it?
   When you are done, the loop should run in
   less then one second."""


""" READ BUT DO NOT MODFY THE CODE PAST THIS LINE, MODIFY INSIDE THE FUNCTIONS """

randgen = np.random.default_rng(seed=135893)

N = 20000
A = (randgen.normal(2, 5, (N, N)).astype('float32')).clip(0, None)
b = randgen.normal(0, 1, (N, 1)).astype('float32')

norm = np.linalg.norm(b)
b /= norm
old_norm = 0
start_time = time()
for i in range(50):
    relative_difference = (abs(norm - old_norm) / (0.5 * (norm + old_norm)))
    print(f'Iteration {i}, eigenvalue approximation: {norm}, '
          f'relative difference: {relative_difference}')
    if relative_difference < 1e-8:
        print('Convergence!')
        break
    old_norm = norm


    b = (cp.array(A) @ cp.array(b)).get()
    norm = np.linalg.norm(b)
    b = b / norm

convergence_time = time() - start_time
print(f'Time to convergence: {convergence_time:.2f} seconds.')

if convergence_time > 0.5:
    raise ValueError('It should take less than half a'
                     f'second to run this optimization, but it took {convergence_time:2f}!')

if not cp.isclose(norm, 63035.05469):
    raise ValueError(f'The computed eigenvalue is not correct! {norm.get():.5f} is not close to 6290.72419')
