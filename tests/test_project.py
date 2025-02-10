 
import pytest
import polars as pl

from src.m02 import (
    get_squared_error_0,
    get_squared_error,
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


