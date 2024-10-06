## Optimization with numpy

[NumPy](https://numpy.org/) is the backbone of all scientific computing in Python. It contains bindings to various standard numeric libraries, and allows a wide range of algebraic operations to be carried out on arrays with ease. For a simple example:

```python
import numpy as np
from time import time
x = np.ones(1024 * 1024 * 100) # vector of 100 * 1024^2 eleements, filled with the number 1

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
```

```bash
Naive time needed: 5.00e+00
Numpy time needed: 3.81e-02
```

Contrary to widely-held belief even among some computational scientists, most of the operations in numpy are _not_ parallelized, although they are typically much faster than standard Python operations, due to being implemented in `C` libraries.
The exceptions, which are parallelized, are the matrix multiplication operation `A @ B` or `np.matmul(A, B)` and the vector dot product operation `np.dot(x, y)`, and by extension functions that leverage these, such as `np.einsum`.
However, the limited amount of parallelization is often not a problem in NumPy, because the computational time basic arithmetic operations tend to be dominated by the amount of time it takes to load a vector into memory.
The lack of parallelization tends to become a problem only when vectors and matrices grow very large.

Broadly speaking, you should always strive to use `numpy`'s built-in operations; sometimes it may be necessary to break up a single large operation into several smaller ones for memory reasons, but for this `numpy` provides built-in functions like `np.array_split` which allows you to easily divide a large array into several smaller ones.

```python
import numpy as np
big_array = np.ones(1024 * 1024 * 1024 * 4) # multi-gigabyte array
sum = 0
for chunk in np.array_split(a, 100): # split into smaller chunks
    sum += chunk.sum() # sum each chunk
print(sum)
```

Thus, replacing or reducing loops in a sensible way is one of the chief ways to increase computation speed using `numpy`.

### Exercise: Optimize `maximum_eigenvalue.py`

The file `maximum_eigenvalue.py` contains Python code for computing the largest eigenvalue of a moderately sized matrix. The optimization should take several seconds but less than a minute to carry out in its current state. It is carried out by using three small functions, which carry out array arithmetic. How can you speed up this code by using NumPy operations? Which of the three functions take the most time to carry out?

Your task is to modify this code so that it runs in less than half a second while giving the same result. It should not be necessary to modify more than one of the three functions to achieve this, but you are welcome to do so, as long as you identify the most expensive function.
