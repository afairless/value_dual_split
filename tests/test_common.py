 
import pytest
import numpy as np

from hypothesis import given, settings, reproduce_failure
import hypothesis.strategies as st

from src.common import (
    calculate_average_of_averages_of_cross_product_pairs,
    predict_inverse_logistic,
    calculate_inverse_logistic_squared_error,
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


def test_predict_inverse_logistic_01():
    """
    Test invalid input:  array with too many dimensions
    """

    xs = np.array([0.10, 0.25, 0.5, 0.75, 0.90]).reshape(-1, 1)

    horizontal_bias_param = 0
    vertical_stretch_param = 1

    with pytest.raises(AssertionError):
        _ = predict_inverse_logistic(
            xs, horizontal_bias_param, vertical_stretch_param)


def test_predict_inverse_logistic_02():
    """
    Test invalid input:  an element in 'x' array not between 0 and 1
    """

    xs = np.array([0.10, 1.25, 0.5, 0.75, 0.90])

    horizontal_bias_param = 0
    vertical_stretch_param = 1

    with pytest.raises(AssertionError):
        _ = predict_inverse_logistic(
            xs, horizontal_bias_param, vertical_stretch_param)


def test_predict_inverse_logistic_03():
    """
    Test invalid input:  verticle stretch parameter not positive
    """

    xs = np.array([0.10, 1.25, 0.5, 0.75, 0.90])

    horizontal_bias_param = 0
    vertical_stretch_param = 0

    with pytest.raises(AssertionError):
        _ = predict_inverse_logistic(
            xs, horizontal_bias_param, vertical_stretch_param)


def test_predict_inverse_logistic_04():
    """
    Test valid input
    """

    xs = np.array([0.10, 0.25, 0.5, 0.75, 0.90])

    horizontal_bias_param = 0
    vertical_stretch_param = 1

    result = predict_inverse_logistic(
        xs, horizontal_bias_param, vertical_stretch_param)
    correct_result = np.array([-2.197225, -1.098612, 0, 1.098612, 2.197225])
    assert np.allclose(result, correct_result, atol=1e-6, rtol=1e-6)


def test_predict_inverse_logistic_05():
    """
    Test valid input
    """

    xs = np.array([0.10, 0.25, 0.5, 0.75, 0.90])

    horizontal_bias_param = 1
    vertical_stretch_param = 1

    result = predict_inverse_logistic(
        xs, horizontal_bias_param, vertical_stretch_param)
    correct_result = np.array([-1.197225, -0.098612, 1, 2.098612, 3.197225])
    assert np.allclose(result, correct_result, atol=1e-6, rtol=1e-6)


def test_predict_inverse_logistic_06():
    """
    Test valid input
    """

    xs = np.array([0.10, 0.25, 0.5, 0.75, 0.90])

    horizontal_bias_param = 0
    vertical_stretch_param = 2

    result = predict_inverse_logistic(
        xs, horizontal_bias_param, vertical_stretch_param)
    correct_result = 2 * np.array([-2.197225, -1.098612, 0, 1.098612, 2.197225])
    assert np.allclose(result, correct_result, atol=1e-6, rtol=1e-6)


def test_predict_inverse_logistic_07():
    """
    Test valid input
    """

    xs = np.array([0.10, 0.25, 0.5, 0.75, 0.90])

    horizontal_bias_param = 1
    vertical_stretch_param = 2

    result = predict_inverse_logistic(
        xs, horizontal_bias_param, vertical_stretch_param)
    correct_result = (
        1 + (2 * np.array([-2.197225, -1.098612, 0, 1.098612, 2.197225])))
    assert np.allclose(result, correct_result, atol=1e-6, rtol=1e-6)


def test_calculate_logit_squared_error_01():
    """
    Test invalid input:  array with too many dimensions
    """

    xs = np.array([0.10, 0.25, 0.5, 0.75, 0.90]).reshape(-1, 1)
    ys = np.array([-1, 0, 1, 2, 3])

    horizontal_bias_param = 0
    vertical_stretch_param = 1

    with pytest.raises(AssertionError):
        _ = calculate_inverse_logistic_squared_error(
            [horizontal_bias_param, vertical_stretch_param], xs, ys)


def test_calculate_logit_squared_error_02():
    """
    Test invalid input:  array with too many dimensions
    """

    xs = np.array([0.10, 0.25, 0.5, 0.75, 0.90])
    ys = np.array([-1, 0, 1, 2, 3]).reshape(-1, 1)

    horizontal_bias_param = 0
    vertical_stretch_param = 1

    with pytest.raises(AssertionError):
        _ = calculate_inverse_logistic_squared_error(
            [horizontal_bias_param, vertical_stretch_param], xs, ys)


def test_calculate_logit_squared_error_03():
    """
    Test invalid input:  arrays with different lengths
    """

    xs = np.array([0.10, 0.25, 0.5, 0.75])
    ys = np.array([-1, 0, 1, 2, 3])

    horizontal_bias_param = 0
    vertical_stretch_param = 1

    with pytest.raises(AssertionError):
        _ = calculate_inverse_logistic_squared_error(
            [horizontal_bias_param, vertical_stretch_param], xs, ys)


def test_calculate_logit_squared_error_04():
    """
    Test invalid input:  an element in 'x' array not between 0 and 1
    """

    xs = np.array([0.10, 1.25, 0.5, 0.75])
    ys = np.array([-1, 0, 1, 2, 3])

    horizontal_bias_param = 0
    vertical_stretch_param = 1

    with pytest.raises(AssertionError):
        _ = calculate_inverse_logistic_squared_error(
            [horizontal_bias_param, vertical_stretch_param], xs, ys)


def test_calculate_logit_squared_error_05():
    """
    Test invalid input:  verticle stretch parameter not positive
    """

    xs = np.array([0.10, 1.25, 0.5, 0.75])
    ys = np.array([-1, 0, 1, 2, 3])

    horizontal_bias_param = 0
    vertical_stretch_param = 0

    with pytest.raises(AssertionError):
        _ = calculate_inverse_logistic_squared_error(
            [horizontal_bias_param, vertical_stretch_param], xs, ys)


def test_calculate_logit_squared_error_06():
    """
    Test valid input
    """

    xs = np.array([0.10, 0.25, 0.5, 0.75, 0.90])
    ys = np.array([-1, 0, 1, 2, 3])

    horizontal_bias_param = 0
    vertical_stretch_param = 1

    result = calculate_inverse_logistic_squared_error(
        [horizontal_bias_param, vertical_stretch_param], xs, ys)
    correct_result = 5.097243834763624
    assert np.isclose(result, correct_result, atol=1e-6, rtol=1e-6)


def test_calculate_logit_squared_error_07():
    """
    Test valid input
    """

    xs = np.array([0.10, 0.25, 0.5, 0.75, 0.90])
    ys = np.array([-1, 0, 1, 2, 3])

    horizontal_bias_param = 1
    vertical_stretch_param = 1

    result = calculate_inverse_logistic_squared_error(
        [horizontal_bias_param, vertical_stretch_param], xs, ys)
    correct_result = 0.097243834763626
    assert np.isclose(result, correct_result, atol=1e-6, rtol=1e-6)


def test_calculate_logit_squared_error_08():
    """
    Test valid input
    """

    xs = np.array([0.10, 0.25, 0.5, 0.75, 0.90])
    ys = np.array([-1, 0, 1, 2, 3])

    horizontal_bias_param = 0
    vertical_stretch_param = 2

    result = calculate_inverse_logistic_squared_error(
        [horizontal_bias_param, vertical_stretch_param], xs, ys)
    correct_result = 19.333466885778893
    assert np.isclose(result, correct_result, atol=1e-6, rtol=1e-6)


def test_calculate_logit_squared_error_09():
    """
    Test valid input
    """

    xs = np.array([0.10, 0.25, 0.5, 0.75, 0.90])
    ys = np.array([-1, 0, 1, 2, 3])

    horizontal_bias_param = 1
    vertical_stretch_param = 2

    result = calculate_inverse_logistic_squared_error(
        [horizontal_bias_param, vertical_stretch_param], xs, ys)
    correct_result = 14.333466885778892
    assert np.isclose(result, correct_result, atol=1e-6, rtol=1e-6)


