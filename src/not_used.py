#! /usr/bin/env python3

import numpy as np
import polars as pl

from scipy.optimize import minimize
try:
    from src.m02 import (
        get_squared_error,
        )
except:
    from m02 import (
        get_squared_error,
        )

 
def get_squared_error_0(
    params: list[float], examples: pl.DataFrame) -> float:

    a_n = examples[:, 0].n_unique()
    b_n = examples[:, 1].n_unique()
    assert len(params) == 1 + a_n + b_n

    ap = params[0]
    bp = 1 - ap
    a = params[1:(1+a_n)]
    b = params[(1+a_n):]
    
    # Compute the sum of squared differences between predicted and actual v
    error = 0.0
    # print('\n')
    for i, j, v in examples.iter_rows():
        predicted_v = ap * a[i] + bp * b[j]
        error += (predicted_v - v) ** 2
        # print(i, a[i], ap, j, b[j], bp, 'pv,v,e', predicted_v, v, error)
    
    return error


def optimization_example():
    """
    Example modified from ChatGPT demonstrating how to solve linear system of 
        equations in Python
    """

    # i = index for A, j = index for B, v = observed weighted average
    examples = pl.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'v': [1.5, 2.0, 2.5, 3.0]})

    p_initial = 0.5         # Starting value for p
    a_initial = [1.0, 2.0]  # Initial guesses for a_i
    b_initial = [1.5, 2.5]  # Initial guesses for b_j
    initial_params = [p_initial] + a_initial + b_initial

    result = minimize(
        get_squared_error, initial_params, args=examples, method='Nelder-Mead')

    # Extract the optimized parameters
    optimized_params = result.x
    optimized_p = optimized_params[0]
    optimized_a = optimized_params[1:1+len(a_initial)]
    optimized_b = optimized_params[1+len(a_initial):]

    v0 = optimized_p * optimized_a[0] + (1 - optimized_p) * optimized_b[0]
    v1 = optimized_p * optimized_a[0] + (1 - optimized_p) * optimized_b[1]
    v2 = optimized_p * optimized_a[1] + (1 - optimized_p) * optimized_b[0]
    v3 = optimized_p * optimized_a[1] + (1 - optimized_p) * optimized_b[1]

    assert np.allclose(v0, examples[0, 2], atol=1e-2, rtol=1e-2)
    assert np.allclose(v1, examples[1, 2], atol=1e-2, rtol=1e-2)
    assert np.allclose(v2, examples[2, 2], atol=1e-2, rtol=1e-2)
    assert np.allclose(v3, examples[3, 2], atol=1e-2, rtol=1e-2)


