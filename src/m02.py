#! /usr/bin/env python3

import random
import numpy as np
import polars as pl
import pandas as pd
import pingouin as pg

from pathlib import Path
from scipy.optimize import minimize, LinearConstraint
from typing import Callable, Iterable
from itertools import product as it_product
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import statsmodels.api as sm
import statsmodels.formula.api as smf


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


def calculate_factors(n: int):
    """
    Calculate the integer factors of 'n' and return them as pairs in a DataFrame
    """

    factors_1 = [e for e in range(2, int(np.sqrt(n)+1)) if n % e == 0]
    factors_2 = [n // e for e in factors_1]

    factor_df = pl.DataFrame({'factor_1': factors_1, 'factor_2': factors_2})
    
    df_products = (
        factor_df
        .select(pl.col('factor_1').mul(pl.col('factor_2')).alias('c'))['c'])
    assert df_products.n_unique() == 1 
    assert df_products[0] == n

    return factor_df


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


def get_squared_error(
    params: list[float], example_df: pl.DataFrame) -> float:
    """
    Calculate the sum of squared errors between the predicted and actual values
        for a linear model with a single predictor and a single outcome variable

    Assumes the following structure for 'example_df':
        - Column 0:  'a' (predictor variable)
        - Column 1:  'b' (predictor variable)
        - Column 2:  'v' (outcome variable)
    """

    a_n = example_df[:, 0].n_unique()
    b_n = example_df[:, 1].n_unique()
    assert len(params) == 1 + a_n + b_n

    ap = params[0]
    bp = 1 - ap
    a = params[1:(1+a_n)]
    b = params[(1+a_n):]

    a_v_dict = {i: a[i] for i in range(a_n)}
    b_v_dict = {i: b[i] for i in range(b_n)}

    calc_df = pl.DataFrame({
        'av': example_df[:, 0].cast(pl.Float64).replace(a_v_dict),
        'ap': [ap] * len(example_df),
        'bv': example_df[:, 1].cast(pl.Float64).replace(b_v_dict),
        'bp': [bp] * len(example_df)})

    calc_df = calc_df.with_columns(
        pl.col('av').mul(pl.col('ap')).add(pl.col('bv').mul(pl.col('bp')))
        .alias('dp'))
    
    calc_df = calc_df.with_columns(
        (pl.col('dp') - example_df[:, 2]).pow(2).alias('sq_err'))
    
    error = calc_df['sq_err'].sum()

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


def generate_sequential_integers(n: int) -> Iterable[float]:
    return range(1, n + 1)


def generate_uniform_random_floats(
    n: int, range_min: float=0.01, range_max: float=0.99, precision: int=2
    ) -> Iterable[float]:
    """
    Generate 'n' random floats in the range ['range_min', 'range_max'] with 
        'precision' decimal places
    """
    return [
        round(random.uniform(range_min, range_max), precision) 
        for _ in range(n)]


def get_all_value_combinations(
    g_n: int, i_n: int, g_prop: float, 
    get_g_v: Callable, get_i_v: Callable) -> pl.DataFrame:
    """
    Generate values separately for 'g_n' groups and 'i_n' individuals, then 
        generate every combination of groups and individuals with corresponding
        values that are a weighted sum of their separate values, where 'g_prop'
        denotes the group weight and the individual weight is 1 - 'g_prop'
        
    Abbreviations:
        'g':  group
        'i':  individual
        'v':  value
        'n':  number
        'prop':  proportion
        'cv':  'combination value'
    """

    assert g_n > 0
    assert i_n > 0
    assert 0 <= g_prop <= 1

    i_prop = 1 - g_prop
    g_id = range(g_n)
    i_id = range(i_n)
    # g_v = range(1, g_n + 1)
    g_v = get_g_v(g_n)
    # i_v = range(1, i_n + 1)
    i_v = get_i_v(i_n)
    g_id_v = zip(g_id, g_v)
    i_id_v = zip(i_id, i_v)

    pre_df = [e[0] + e[1] for e in it_product(g_id_v, i_id_v)]
    df = pl.DataFrame(pre_df, orient='row')
    colnames = ['g_id', 'g_v', 'i_id', 'i_v']
    df.columns = colnames
    df = df[['g_id', 'i_id', 'g_v', 'i_v']]
    df = df.with_columns(
        pl.col('g_v').mul(g_prop).add(pl.col('i_v').mul(i_prop)).alias('cv'))

    return df


def get_results_given_parameter_set(
    g_n: int, i_n: int, g_prop: float, df: pl.DataFrame) -> list[float]:

    group_variance = partition_group_variance(df, 'g_id', 'cv')

    icc_df = pg.intraclass_corr(
        data=df.to_pandas(), targets='i_id', raters='g_id', ratings='cv')

    icc1 = icc_df['ICC'][0]
    icc2 = icc_df['ICC'][1]
    icc3 = icc_df['ICC'][2]
    assert isinstance(icc1, float)
    assert isinstance(icc2, float)
    assert isinstance(icc3, float)
    icc1 = float(icc1)
    icc2 = float(icc2)
    icc3 = float(icc3)

    optimization_input_df = df[['g_id', 'i_id', 'cv']]

    p_initial = 0.5         # Starting value for p
    a_initial = [0.5 for _ in range(g_n)]
    b_initial = [0.5 for _ in range(i_n)]
    initial_params = [p_initial] + a_initial + b_initial
    bounds = [(0, 1) for _ in range(len(initial_params))]

    result = minimize(
        get_squared_error, initial_params, args=optimization_input_df, 
        # method='Nelder-Mead', 
        bounds=bounds, tol=1e-2)

    final_error = get_squared_error(result.x, df)

    result = [
        g_n, i_n, 
        g_prop, float(result.x[0]), 
        final_error, result.success,
        group_variance.btwn_group_var,
        group_variance.wthn_group_var,
        group_variance.total_var,
        group_variance.btwn_var_prop,
        group_variance.wthn_var_prop,
        icc1, icc2, icc3]

    return result


def get_results_given_parameters(
    factor_df: pl.DataFrame, g_props: list[float], output_path: Path
    ) -> pl.DataFrame:

    results = []
    for g_n, i_n in factor_df.iter_rows():
        for g_prop in g_props:
            print(g_n, i_n)
            print(f'Running g_n: {g_n}, i_n: {i_n}, g_prop: {g_prop}')

            df = get_all_value_combinations(
                g_n, i_n, g_prop, 
                generate_uniform_random_floats, generate_uniform_random_floats)

            filename_stem = (
                'gn_' + str(g_n) + 
                '_in_' + str(i_n) + 
                '_gprop_' + str(g_prop))
            save_dataframe_to_csv_and_parquet(df, filename_stem, output_path)

            result = get_results_given_parameter_set(g_n, i_n, g_prop, df)

            results.append(result)

    colnames = [
        'g_n', 'i_n', 'g_prop', 'g_prop_est', 'error', 'success',
        'btwn_group_var', 'wthn_group_var', 'total_var', 
        'btwn_var_prop', 'wthn_var_prop',
        'icc1', 'icc2', 'icc3']
    result_df = pl.DataFrame(results, orient='row')
    result_df.columns = colnames

    return result_df


def save_dataframe_to_csv_and_parquet(
    df: pl.DataFrame, filename_stem: str, output_path : Path) -> None:

    output_filename = filename_stem + '.csv'
    output_filepath = output_path / output_filename
    df.write_csv(output_filepath)

    output_filename = filename_stem + '.parquet'
    output_filepath = output_path / output_filename
    df.write_parquet(output_filepath)


def plot_group_proportion_estimates_by_group_individual_ratios(
    result_df: pl.DataFrame, output_path: Path, output_filename: str) -> None:

    plot_colnames = ['g_n', 'g_prop', 'g_prop_est', 'i_n_to_g_n']
    plot_df = (
        result_df[plot_colnames]
        .group_by('g_n')
        .agg([
            pl.col('g_prop'), 
            pl.col('g_prop_est'), 
            pl.col('i_n_to_g_n').first()])).sort('g_n')

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(plot_df)) for i in range(len(plot_df))]

    for i, row in enumerate(plot_df.iter_rows(named=True)):
        plt.scatter(
            row['g_prop'], row['g_prop_est'], label=round(row['i_n_to_g_n'], 2), 
            color=colors[i])
        plt.plot(
            row['g_prop'], row['g_prop_est'], linestyle='-', marker='o', 
            color=colors[i])

    plt.legend(title='i_n_to_g_n')
    plt.xlabel('True group proportion, g_prop')
    plt.ylabel('Estimated group proportion, g_prop_est')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    output_filepath = output_path / output_filename
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_metric_by_group_individual_ratios(
    result_df: pl.DataFrame, metric_colname: str, metric_label: str,
    output_path: Path, output_filename: str) -> None:

    plot_colnames = ['g_n', 'g_prop', metric_colname, 'i_n_to_g_n']
    plot_df = (
        result_df[plot_colnames]
        .group_by('g_n')
        .agg([
            pl.col('g_prop'), 
            pl.col(metric_colname), 
            pl.col('i_n_to_g_n').first()])).sort('g_n')

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(plot_df)) for i in range(len(plot_df))]

    for i, row in enumerate(plot_df.iter_rows(named=True)):
        plt.scatter(
            row['g_prop'], row[metric_colname], label=round(row['i_n_to_g_n'], 2), 
            color=colors[i])
        plt.plot(
            row['g_prop'], row[metric_colname], linestyle='-', marker='o', 
            color=colors[i])

    plt.legend(title='i_n_to_g_n')
    plt.xlabel('True group proportion, g_prop')
    plt.ylabel(metric_label)
    plt.xlim(0, 1)

    output_filepath = output_path / output_filename
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_metric_by_metric(
    result_df: pl.DataFrame, 
    metric_x_colname: str, metric_x_label: str,
    metric_y_colname: str, metric_y_label: str,
    output_path: Path, output_filename: str) -> None:

    plot_colnames = ['g_n', metric_x_colname, metric_y_colname, 'i_n_to_g_n']
    plot_df = (
        result_df[plot_colnames]
        .group_by('g_n')
        .agg([
            pl.col(metric_x_colname), 
            pl.col(metric_y_colname), 
            pl.col('i_n_to_g_n').first()])).sort('g_n')

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(plot_df)) for i in range(len(plot_df))]

    # plot horizontal line
    plt.axhline(0, color='black', linewidth=0.5)
    # plot vertical line
    plt.axvline(0, color='black', linewidth=0.5)

    for i, row in enumerate(plot_df.iter_rows(named=True)):
        plt.scatter(
            row[metric_x_colname], row[metric_y_colname], label=round(row['i_n_to_g_n'], 2), 
            color=colors[i])
        plt.plot(
            row[metric_x_colname], row[metric_y_colname], linestyle='-', marker='o', 
            color=colors[i])

    plt.legend(title='i_n_to_g_n')
    plt.xlabel(metric_x_label)
    plt.ylabel(metric_y_label)
    # plt.xlim(0, 1)

    output_filepath = output_path / output_filename
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def main():

    # pl.Config.set_tbl_cols(12)
    # pl.Config.set_tbl_rows(24)

    print('This is the saved main function.')

    # correspondences
    #   raters      targets     ratings
    #   group       x           y
    #   sellers     buyers      prices
    #
    # for our problem:
    #   raters:  fixed
    #   targets:  random

    # df = get_all_value_combinations(
    #     g_n, i_n, g_prop, 
    #     generate_sequential_integers, generate_sequential_integers)

    total = 72 ** 2
    output_path = Path.cwd() / 'output' / ('combo_n_' + str(total))
    output_path.mkdir(exist_ok=True, parents=True)

    factor_df = calculate_factors(total)
    g_props = [e/10 for e in range(1, 10, 2)]
    result_df = get_results_given_parameters(factor_df, g_props, output_path)

    filename_stem = 'results'
    save_dataframe_to_csv_and_parquet(result_df, filename_stem, output_path)
    # result_df = pl.read_parquet(output_path / 'results.parquet')


    # PLOT RESULTS
    ################################################## 

    result_df = result_df.with_columns(
        (pl.col('i_n') / (pl.col('g_n'))).alias('i_n_to_g_n'))
    # result_df.filter(pl.col('g_prop') == 0.5).sort('g_prop_est')

    output_filename = 'group_proportions_by_individual_to_group_ratios.png'
    plot_group_proportion_estimates_by_group_individual_ratios(
        result_df, output_path, output_filename)
    # shows that very imbalanced individual-to-group ratios produce less accurate 
    #   estimates

    output_filename = 'proportion_error_by_individual_to_group_ratios.png'
    metric_y_colname = 'error'
    metric_y_label = 'Total error, ' + metric_y_colname
    plot_metric_by_group_individual_ratios(
        result_df, metric_y_colname, metric_y_label, output_path, output_filename)
    # shows that error declines as group proportion increases; do I need to
    #   re-scale the error?


    metric_pairs = [
        ('g_prop', 'g_prop_est'), 
        ('g_prop', 'error'), 
        ('g_prop', 'btwn_var_prop'), 
        ('g_prop', 'wthn_var_prop'), 
        ('g_prop', 'total_var'), 
        ('g_prop', 'icc1'), 
        ('g_prop', 'icc2'), 
        ('btwn_var_prop', 'icc1'), 
        ('btwn_var_prop', 'icc2'), 
        ('wthn_var_prop', 'icc1'), 
        ('wthn_var_prop', 'icc2'), 
        ('icc1', 'icc2')]

    for metric_x_colname, metric_y_colname in metric_pairs:

        output_filename = metric_y_colname + '_by_' + metric_x_colname + '.png'
        plot_metric_by_metric(
            result_df, 
            metric_x_colname, metric_x_colname, 
            metric_y_colname, metric_y_colname, 
            output_path, output_filename)

    # metric_x_colname = 'btwn_var_prop'
    # metric_y_colname = 'icc2'
    # output_filename = metric_y_colname + '_by_' + metric_x_colname + '.png'
    # plot_metric_by_metric(
    #     result_df, 
    #     metric_x_colname, metric_x_colname, 
    #     metric_y_colname, metric_y_colname, 
    #     output_path, output_filename)


    '''
    metric_y_colname = 'btwn_var_prop'
    output_filename = metric_y_colname + '_by_individual_to_group_ratios.png'
    metric_y_label = 'Between-group variance proportion, ' + metric_y_colname
    plot_metric_by_group_individual_ratios(
        result_df, metric_y_colname, metric_y_label, output_path, output_filename)

    metric_y_colname = 'total_var'
    output_filename = metric_y_colname + '_by_individual_to_group_ratios.png'
    metric_y_label = 'Total variance, ' + metric_y_colname
    plot_metric_by_group_individual_ratios(
        result_df, metric_y_colname, metric_y_label, output_path, output_filename)

    metric_y_colname = 'icc1'
    output_filename = metric_y_colname + '_by_individual_to_group_ratios.png'
    metric_y_label = 'ICC1, ' + metric_y_colname
    plot_metric_by_group_individual_ratios(
        result_df, metric_y_colname, metric_y_label, output_path, output_filename)

    metric_y_colname = 'icc2'
    output_filename = metric_y_colname + '_by_individual_to_group_ratios.png'
    metric_y_label = 'ICC2, ' + metric_y_colname
    plot_metric_by_group_individual_ratios(
        result_df, metric_y_colname, metric_y_label, output_path, output_filename)

    metric_y_colname = 'icc3'
    output_filename = metric_y_colname + '_by_individual_to_group_ratios.png'
    metric_y_label = 'ICC3, ' + metric_y_colname
    plot_metric_by_group_individual_ratios(
        result_df, metric_y_colname, metric_y_label, output_path, output_filename)
    '''







if __name__ == '__main__':
    main()
