 
import pytest
import numpy as np
import polars as pl

from hypothesis import given, settings, reproduce_failure
import hypothesis.strategies as st

from src.common import (
    calculate_average_of_averages_of_cross_product_pairs,
    )


def test_calculate_average_of_averages_of_cross_product_pairs_01():
    """
    Test invalid input:  vectors of inadequate length
    """

    v1 = np.array([])
    v2 = np.array([])

    with pytest.raises(AssertionError):
        _ = calculate_average_of_averages_of_cross_product_pairs(v1, v2)


def test_calculate_average_of_averages_of_cross_product_pairs_02():
    """
    Test invalid input:  vectors of inadequate length
    """

    v1 = np.array([1])
    v2 = np.array([])

    with pytest.raises(AssertionError):
        _ = calculate_average_of_averages_of_cross_product_pairs(v1, v2)


def test_calculate_average_of_averages_of_cross_product_pairs_03():
    """
    Test invalid input:  vectors of inadequate length
    """

    v1 = np.array([1])
    v2 = np.array([1])

    with pytest.raises(AssertionError):
        _ = calculate_average_of_averages_of_cross_product_pairs(v1, v2)


def test_calculate_average_of_averages_of_cross_product_pairs_04():
    """
    Test invalid input:  vectors of inadequate length
    """

    v1 = np.array([1, 1])
    v2 = np.array([1])

    with pytest.raises(AssertionError):
        _ = calculate_average_of_averages_of_cross_product_pairs(v1, v2)


def test_calculate_average_of_averages_of_cross_product_pairs_05():
    """
    Test valid input
    """

    v1 = np.array([1, 1])
    v2 = np.array([1, 1])

    result = calculate_average_of_averages_of_cross_product_pairs(v1, v2)
    correct_result = 1
    assert result == correct_result


def test_calculate_average_of_averages_of_cross_product_pairs_06():
    """
    Test valid input
    """

    v1 = np.array([1, 3])
    v2 = np.array([5, 7])

    result = calculate_average_of_averages_of_cross_product_pairs(v1, v2)
    correct_result = 4
    assert result == correct_result


def test_calculate_average_of_averages_of_cross_product_pairs_07():
    """
    Test invalid input:  vectors with too many dimensions
    """

    v1 = np.array([1, 3]).reshape(-1, 1)
    v2 = np.array([5, 7])

    with pytest.raises(AssertionError):
        _ = calculate_average_of_averages_of_cross_product_pairs(v1, v2)


def test_calculate_average_of_averages_of_cross_product_pairs_08():
    """
    Test invalid input:  vectors with too many dimensions
    """

    v1 = np.array([1, 3])
    v2 = np.array([5, 7]).reshape(-1, 1)

    with pytest.raises(AssertionError):
        _ = calculate_average_of_averages_of_cross_product_pairs(v1, v2)


@given(
    size_1=st.integers(min_value=2, max_value=100),
    size_2=st.integers(min_value=2, max_value=100),
    seed=st.integers(min_value=0, max_value=100_000))
@settings(print_blob=True)
def test_calculate_average_of_averages_of_cross_product_pairs_09(
    size_1: int, size_2: int, seed: int):
    """
    Test valid input
    """

    rng = np.random.default_rng(seed)
    v1 = rng.integers(low=0, high=500, size=size_1)
    v2 = rng.integers(low=0, high=500, size=size_2)

    result = calculate_average_of_averages_of_cross_product_pairs(v1, v2)

    grid_a, grid_b = np.meshgrid(v1, v2)
    pairs = np.column_stack((grid_a.ravel(), grid_b.ravel()))
    correct_result = pairs.mean(axis=1).mean()

    assert result == correct_result





