import convolve_py
import convolve_cy
import numpy as np
import time

N = 200

f = np.arange(N*N, dtype=np.int).reshape((N,N))
g = np.arange(81, dtype=np.int).reshape((9,9))

s = time.time()
convolve_py.naive_convolve(f,g)
print(time.time() - s)

s = time.time()
convolve_cy.naive_convolve(f,g)
print(time.time() - s)