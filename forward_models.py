
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from BM_prior import dt, num_pt

"""
Wrapper for the Euler solver in C

To compile: `gcc -fPIC -shared -o euler_solver.so src/euler_solver.c`
"""


# Load shared object
lib = ctypes.cdll.LoadLibrary("euler_solver.so")

# Load OU solver function
euler_OU_C = lib.euler_solver_fun
euler_OU_C.restype = None
euler_OU_C.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_size_t,
                ctypes.c_double,]

# Load double well function
euler_DW_C = lib.euler_double_well
euler_DW_C.restype = None
euler_DW_C.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
                ctypes.c_double,]

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



def solve_forward_Euler(W, θ, σ):
    """
    Solves Euler for Brownian Motion W. Calls a C function

    Parameters:
    W: ndarray
        BM of size num_pt
    """
    pp = np.zeros(num_pt)
    euler_OU_C(W, pp, θ, σ, num_pt, dt)
    return pp


def solve_DW(W):
    """
    Euler discretisation of a double well potential for Brownian Motion W. Calls a C function

    Parameters:
    W: ndarray
        BM of size num_pt
    """
    pp = np.zeros(num_pt)
    euler_DW_C(W, pp, num_pt, dt)
    return pp

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
