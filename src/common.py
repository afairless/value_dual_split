#! /usr/bin/env python3

import numpy as np


def calculate_average_of_averages_of_cross_product_pairs(
    vector_1: np.ndarray, vector_2: np.ndarray):
    """
    Use a formula to calculate the following average:
        Given two 1-dimensional vectors, pair every element of vector 1 with 
            every element of vector 2, average the pairs, then average the pair 
            averages
    """

    assert np.ndim(vector_1) == 1
    assert np.ndim(vector_2) == 1
    assert len(vector_1) > 1
    assert len(vector_2) > 1

    len_1 = len(vector_1)
    len_2 = len(vector_2)
    sum_1 = vector_1.sum()
    sum_2 = vector_2.sum()

    return ((len_2 * sum_1) + (len_1 * sum_2)) / (2 * len_1 * len_2)

 
