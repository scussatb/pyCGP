from pycgp import CGP
from pycgp.cgpfunctions import *
import numpy as np

def rand_arg():
    return np.random.rand() * 2.0 - 1

def build_func_lib():
    return [CGP.CGPFunc(f_sum, 'sum', 2),
            CGP.CGPFunc(f_aminus, 'aminus', 2),
            CGP.CGPFunc(f_mult, 'mult', 2),
            CGP.CGPFunc(f_exp, 'exp', 2),
            CGP.CGPFunc(f_abs, 'abs', 1),
            CGP.CGPFunc(f_sqrt, 'sqrt', 1),
            CGP.CGPFunc(f_sqrtxy, 'sqrtxy', 2),
            CGP.CGPFunc(f_squared, 'squared', 1),
            CGP.CGPFunc(f_pow, 'pow', 2),
            CGP.CGPFunc(f_one, 'one', 0),
            CGP.CGPFunc(f_zero, 'zero', 0),
            CGP.CGPFunc(f_inv, 'inv', 1),
            CGP.CGPFunc(f_gt, 'gt', 2),
            CGP.CGPFunc(f_asin, 'asin', 1),
            CGP.CGPFunc(f_acos, 'acos', 1),
            CGP.CGPFunc(f_atan, 'atan', 1),
            CGP.CGPFunc(f_min, 'min', 2),
            CGP.CGPFunc(f_max, 'max', 2),
            CGP.CGPFunc(f_round, 'round', 1),
            CGP.CGPFunc(f_floor, 'floor', 1),
            CGP.CGPFunc(f_ceil, 'ceil', 1)
            ]

def test_functions():
    fs = build_func_lib()
    assert len(fs) > 0
    for function in fs:
        f = function.function
        inputs = [rand_arg(), -1, 0, 1]
        for i in inputs:
            out = f([i for i in range(function.arity)])
            assert ~(np.isnan(out))
            assert out <= 1
            assert out >= -1
