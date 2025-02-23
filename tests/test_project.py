 
import pytest
import numpy as np
import polars as pl

from hypothesis import given, settings, reproduce_failure
import hypothesis.strategies as st

from src.m02 import (
    get_squared_error_0,
    get_squared_error,
    get_squared_error_lin_alg,
    calculate_inverse_logistic_squared_error,
    )

from src.m03 import (
    dummy_code_two_level_hierarchical_categories,
    )


def test_get_squared_error_0_01():
    """
    Test 'params' with incorrect length
    """

    df = pl.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'v': [1.5, 2.0, 2.5, 3.0]})

    params = [0.5]

    with pytest.raises(AssertionError):
        _ = get_squared_error_0(params, df)


def test_get_squared_error_0_02():
    """
    Test valid input
    """

    df = pl.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'v': [0., 0.5, 1., 1.5]})

    params = [0.5, 0.5, 0.5, 0.5, 0.5]

    result = get_squared_error_0(params, df)
    correct_result = 1.5
    assert result == correct_result


def test_get_squared_error_0_03():
    """
    Test valid input
    """

    df = pl.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'v': [-1., 0, 1., 2]})

    params = [0.5, 0.5, 0.5, 0.5, 0.5]

    result = get_squared_error_0(params, df)
    correct_result = 5.
    assert result == correct_result


def test_get_squared_error_0_04():
    """
    Test valid input
    """

    df = pl.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'v': [-1., 0, 1., 2]})

    params = [0.8, 0.1, 0.3, 0.5, 0.7]

    result = get_squared_error_0(params, df)
    correct_result = 4.5008
    assert result == correct_result


def test_get_squared_error_01():
    """
    Test 'params' with incorrect length
    """

    df = pl.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'v': [1.5, 2.0, 2.5, 3.0]})

    params = [0.5]

    with pytest.raises(AssertionError):
        _ = get_squared_error(params, df)


def test_get_squared_error_02():
    """
    Test valid input
    """

    df = pl.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'v': [0., 0.5, 1., 1.5]})

    params = [0.5, 0.5, 0.5, 0.5, 0.5]

    result = get_squared_error(params, df)
    correct_result = 1.5
    assert result == correct_result


def test_get_squared_error_03():
    """
    Test valid input
    """

    df = pl.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'v': [-1., 0, 1., 2]})

    params = [0.5, 0.5, 0.5, 0.5, 0.5]

    result = get_squared_error(params, df)
    correct_result = 5.
    assert result == correct_result


def test_get_squared_error_04():
    """
    Test valid input
    """

    df = pl.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'v': [-1., 0, 1., 2]})

    params = [0.8, 0.1, 0.3, 0.5, 0.7]

    result = get_squared_error(params, df)
    correct_result = 4.5008
    assert result == correct_result


def test_get_squared_error_lin_alg_01():
    """
    Test 'params' with incorrect length
    """

    df = pl.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'v': [1.5, 2.0, 2.5, 3.0]})

    params = [0.5]

    with pytest.raises(AssertionError):
        _ = get_squared_error_lin_alg(params, df)


def test_get_squared_error_lin_alg_02():
    """
    Test valid input
    """

    df = pl.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'v': [0., 0.5, 1., 1.5]})

    params = [0.5, 0.5, 0.5, 0.5, 0.5]

    result = get_squared_error_lin_alg(params, df)
    correct_result = 1.5
    assert result == correct_result


def test_get_squared_error_lin_alg_03():
    """
    Test valid input
    """

    df = pl.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'v': [-1., 0, 1., 2]})

    params = [0.5, 0.5, 0.5, 0.5, 0.5]

    result = get_squared_error_lin_alg(params, df)
    correct_result = 5.
    assert result == correct_result


def test_get_squared_error_lin_alg_04():
    """
    Test valid input
    """

    df = pl.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'v': [-1., 0, 1., 2]})

    params = [0.8, 0.1, 0.3, 0.5, 0.7]

    result = get_squared_error_lin_alg(params, df)
    correct_result = 4.5008
    assert result == correct_result


def test_dummy_code_two_level_hierarchical_categories_01():
    """
    Test invalid input
    """

    group_n = 0
    individual_n = 0
    with pytest.raises(AssertionError):
        _ = dummy_code_two_level_hierarchical_categories(group_n, individual_n)


def test_dummy_code_two_level_hierarchical_categories_02():
    """
    Test invalid input
    """

    group_n = 0
    individual_n = 1
    with pytest.raises(AssertionError):
        _ = dummy_code_two_level_hierarchical_categories(group_n, individual_n)


def test_dummy_code_two_level_hierarchical_categories_03():
    """
    Test invalid input
    """

    group_n = 1
    individual_n = 0
    with pytest.raises(AssertionError):
        _ = dummy_code_two_level_hierarchical_categories(group_n, individual_n)


def test_dummy_code_two_level_hierarchical_categories_04():
    """
    Test valid input
    """

    group_n = 1
    individual_n = 1
    result = dummy_code_two_level_hierarchical_categories(group_n, individual_n)

    correct_result = np.array([1, 1])
    assert (result == correct_result).all()


def test_dummy_code_two_level_hierarchical_categories_05():
    """
    Test valid input
    """

    group_n = 1
    individual_n = 2
    result = dummy_code_two_level_hierarchical_categories(group_n, individual_n)

    correct_result = np.array(([1, 1, 0], [1, 0, 1]))
    assert (result == correct_result).all()


def test_dummy_code_two_level_hierarchical_categories_06():
    """
    Test valid input
    """

    group_n = 2
    individual_n = 1
    result = dummy_code_two_level_hierarchical_categories(group_n, individual_n)

    correct_result = np.array(([1, 0, 1], [0, 1, 1]))
    assert (result == correct_result).all()


def test_dummy_code_two_level_hierarchical_categories_07():
    """
    Test valid input
    """

    group_n = 2
    individual_n = 2
    result = dummy_code_two_level_hierarchical_categories(group_n, individual_n)

    correct_result = np.array((
        [1, 0, 1, 0], 
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1]))
    assert (result == correct_result).all()


def test_dummy_code_two_level_hierarchical_categories_08():
    """
    Test valid input
    """

    group_n = 3
    individual_n = 2
    result = dummy_code_two_level_hierarchical_categories(group_n, individual_n)

    correct_result = np.array((
        [1, 0, 0, 1, 0], 
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1]))
    assert (result == correct_result).all()


def test_dummy_code_two_level_hierarchical_categories_09():
    """
    Test valid input
    """

    group_n = 2
    individual_n = 3
    result = dummy_code_two_level_hierarchical_categories(group_n, individual_n)

    correct_result = np.array((
        [1, 0, 1, 0, 0], 
        [1, 0, 0, 1, 0], 
        [1, 0, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1]))
    assert (result == correct_result).all()


def test_dummy_code_two_level_hierarchical_categories_10():
    """
    Test valid input
    """

    group_n = 3
    individual_n = 3
    result = dummy_code_two_level_hierarchical_categories(group_n, individual_n)

    correct_result = np.array((
        [1, 0, 0, 1, 0, 0], 
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1]))
    assert (result == correct_result).all()


@given(
    a_case_n=st.integers(min_value=2, max_value=10),
    b_case_n=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=999, max_value=100_000))
@settings(print_blob=True)
def test_get_squared_error_functions_01(
    a_case_n: int, b_case_n: int, seed: int,
        ):
    """
    Test valid input
    """

    case_n = a_case_n * b_case_n

    rng = np.random.default_rng(seed)
    a_vector = np.repeat(range(a_case_n), b_case_n)
    b_vector = np.array(list(range(b_case_n)) * a_case_n)
    v_vector = rng.random(size=case_n)

    df = pl.DataFrame({
        'a': a_vector,
        'b': b_vector,
        'v': v_vector})

    params = list(rng.random(size=1+a_case_n+b_case_n))

    result_1 = get_squared_error(params, df)
    result_2 = get_squared_error_lin_alg(params, df)
    assert np.isclose(result_1, result_2, atol=1e-6, rtol=1e-6)


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


def test_calculate_logit_squared_error_09():
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


