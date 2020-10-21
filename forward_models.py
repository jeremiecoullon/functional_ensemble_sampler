
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from BM_prior import dt, num_pt
import os

"""
Wrapper for the Euler solver in C

To compile: `gcc -fPIC -shared -o euler_solver.so src/euler_solver.c`
"""


# Load shared object
lib = ctypes.cdll.LoadLibrary(os.path.abspath("euler_solver.so"))


# Load Langevin dynamics function
euler_Langevin_C = lib.euler_Langevin
euler_Langevin_C.restype = None
euler_Langevin_C.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_size_t,
                ctypes.c_double,]


def solve_Langevin(W, alpha, sigma):
    """
    Euler discretisation of a Langevin dynamics on harmonic potential. Calls a C function

    Parameters:
    W: ndarray
        BM of size num_pt
    """
    xx = np.zeros(num_pt)
    pp = np.zeros(num_pt)
    euler_Langevin_C(W, xx, pp, alpha, sigma, num_pt, dt)
    return xx, pp
