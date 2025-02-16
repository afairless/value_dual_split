 
import pytest
import numpy as np
import polars as pl

from src.m02 import (
    get_squared_error_0,
    get_squared_error,
    )

from src.bayes_stan.bayes_stan import (
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


