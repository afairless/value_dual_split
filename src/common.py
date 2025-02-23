#! /usr/bin/env python3

import numpy as np
import polars as pl

from pathlib import Path
from dataclasses import dataclass


##################################################
# SAVE DATAFRAME
##################################################

def save_dataframe_to_csv_and_parquet(
    df: pl.DataFrame, filename_stem: str, output_path : Path) -> None:

    output_filename = filename_stem + '.csv'
    output_filepath = output_path / output_filename

    # df.drop(pl.col(pl.Struct))
    # filter out struct columns to save to 'csv'
    (df
     .select(
        [pl.col(c) for c in df.columns if df[c].dtype != pl.Struct])
     .write_csv(output_filepath))

    output_filename = filename_stem + '.parquet'
    output_filepath = output_path / output_filename
    df.write_parquet(output_filepath)


##################################################
# PARTITION VARIANCE
##################################################

@dataclass
class GroupVariance:

    btwn_group_var: float
    wthn_group_var: float
    total_var: float = np.nan
    btwn_var_prop: float = np.nan
    wthn_var_prop: float = np.nan

    def __post_init__(self):

        self.total_var = self.btwn_group_var + self.wthn_group_var 
        self.btwn_var_prop = self.btwn_group_var / self.total_var
        self.wthn_var_prop = self.wthn_group_var / self.total_var


def partition_group_variance(
    df: pl.DataFrame, group_colname: str, value_colname: str) -> GroupVariance:

    group_means_df = df.group_by(group_colname).mean()
    btwn_group_var = group_means_df[value_colname].var() 
    assert isinstance(btwn_group_var, float)

    group_var_df = df.group_by(group_colname).agg(pl.var(value_colname))
    wthn_group_var = group_var_df[value_colname].mean() 
    assert isinstance(wthn_group_var, float)

    group_variance = GroupVariance(
        btwn_group_var=btwn_group_var, wthn_group_var=wthn_group_var)

    return group_variance


##################################################
# CROSS JOIN STATISTICS
##################################################

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

 
##################################################
# INVERSE LOGISTICS CALCULATIONS
##################################################

def predict_inverse_logistic(
    xs: np.ndarray, horizontal_bias_param: float=0, 
    vertical_stretch_param: float=1) -> np.ndarray:
    """
    Predict the logit of the values for an inverse logistic
    """

    assert np.ndim(xs) == 1
    assert all([0 < e < 1 for e in xs])
    assert vertical_stretch_param > 0

    return (
        horizontal_bias_param + vertical_stretch_param * np.log(xs / (1 - xs)))


def calculate_inverse_logistic_squared_error(
    params: list[float], xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Calculate the sum of squared errors for an inverse logistic regression model
    """

    assert np.ndim(xs) == 1
    assert np.ndim(ys) == 1
    assert len(xs) == len(ys)
    assert all([0 < e < 1 for e in xs])
    assert params[1] > 0

    pred_y = predict_inverse_logistic(xs, params[0], params[1])

    error = np.sum(np.power(pred_y - ys, 2))

    return error


