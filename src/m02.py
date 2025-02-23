#! /usr/bin/env python3

import json
import random
import numpy as np
import polars as pl
import pingouin as pg

from pathlib import Path
from scipy.optimize import minimize, OptimizeResult
from typing import Callable, Iterable
from itertools import product as it_product
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    from src.m03 import (
        dummy_code_two_level_hierarchical_categories,
        )
except:
    from m03 import (
        dummy_code_two_level_hierarchical_categories,
        )


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


def get_squared_error_lin_alg(
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

    dummy_arr = dummy_code_two_level_hierarchical_categories(a_n, b_n)
    a_dummy_arr = dummy_arr[:, :a_n]
    b_dummy_arr = dummy_arr[:, a_n:]

    a_params = np.array(params[1:(1+a_n)])
    b_params = np.array(params[(1+a_n):])
    av = a_dummy_arr @ a_params
    bv = b_dummy_arr @ b_params
    vs = np.concatenate((av.reshape(-1, 1), bv.reshape(-1, 1)), axis=1)
    ps = np.array([params[0], 1 - params[0]]).reshape(-1, 1)
    dp = vs @ ps
    error = np.pow(dp - example_df[:, 2].to_numpy().reshape(-1, 1), 2).sum()

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

    y_max = plot_df['g_prop_est'].explode().max()
    assert isinstance(y_max, float)
    if (0.8 < y_max) & (y_max < 1.2):
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.plot([0, 1], [1, 0], color='black', linestyle='--')

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(plot_df)) for i in range(len(plot_df))]

    for i, row in enumerate(plot_df.iter_rows(named=True)):
        plt.scatter(
            row['g_prop'], row['g_prop_est'], label=round(row['i_n_to_g_n'], 2), 
            color=colors[i])
        plt.plot(
            row['g_prop'], row['g_prop_est'], linestyle='-', marker='o', 
            color=colors[i])

    plt.legend(title='i_n_to_g_n', fontsize=8)
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

    y_max = plot_df[metric_colname].explode().max()
    assert isinstance(y_max, float)
    if (0.8 < y_max) & (y_max < 1.2):
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.plot([0, 1], [1, 0], color='black', linestyle='--')

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(plot_df)) for i in range(len(plot_df))]

    for i, row in enumerate(plot_df.iter_rows(named=True)):
        plt.scatter(
            row['g_prop'], row[metric_colname], label=round(row['i_n_to_g_n'], 2), 
            color=colors[i])
        plt.plot(
            row['g_prop'], row[metric_colname], linestyle='-', marker='o', 
            color=colors[i])

    plt.legend(title='i_n_to_g_n', fontsize=8)
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

    y_max = plot_df[metric_y_colname].explode().max()
    assert isinstance(y_max, float)
    if (0.8 < y_max) & (y_max < 1.2):
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.plot([0, 1], [1, 0], color='black', linestyle='--')

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

    plt.legend(title='i_n_to_g_n', fontsize=8)
    plt.xlabel(metric_x_label)
    plt.ylabel(metric_y_label)
    # plt.xlim(0, 1)

    output_filepath = output_path / output_filename
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_combined_values_by_group_proportion_for_groups(
    plot_df: pl.DataFrame, colnames: list[str], title: str, output_path: Path, 
    output_filename: str) -> None:
    """
    Plot the combined values for each group as a function of the group 
        proportions
    TODO:
        Add spacing between groups
    """

    df_groups = (
        plot_df
        .group_by([colnames[0], colnames[2]])
        .agg(pl.col(colnames[1]))
        .sort(colnames[0], colnames[2]))
    df_groups.columns = ['g_id', 'g_prop', 'cv']

    cmap = plt.get_cmap('viridis')
    g_prop_n = df_groups[colnames[2]].n_unique()
    colors = [cmap(i / g_prop_n) for i in range(g_prop_n)]

    if len(df_groups['cv'][0]) > 300:
        alpha = 0.01
    else:
        alpha = 0.2

    for i, row in enumerate(df_groups.iter_rows(named=True)):
        plt.scatter(
            [i] * len(row['cv']), row['cv'], label=row['g_prop'], 
            alpha=alpha, c=colors[i%g_prop_n])

    plt.title(title)

    output_filepath = output_path / output_filename
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def main_analysis(total: int, output_path: Path):

    # pl.Config.set_tbl_cols(14)
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

    factor_df = calculate_factors(total)
    g_props = [e/10 for e in range(1, 10, 2)]
    result_df = get_results_given_parameters(factor_df, g_props, output_path)
    # result_df.filter(pl.col('g_prop').eq(0.9)).select(pl.col('g_prop', 'g_prop_est', 'btwn_var_prop'))

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
        ('g_prop_est', 'btwn_var_prop'), 
        ('g_prop_est', 'wthn_var_prop'), 
        ('g_prop_est', 'total_var'), 
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


def plot_combined_values_by_group_proportion(output_path: Path):
    """
    Plot the combined values for each group as a function of the group 
        proportions
    """

    df_filepaths = output_path.glob('*.parquet')
    df_filepaths = [
        e for e in df_filepaths if e.stem[:3] == 'gn_' and 'gprop' in e.stem]

    df_filename_groups = list(
        set([e.stem.split('gprop')[0] for e in df_filepaths]))

    for df_filename_group in df_filename_groups:

        df_group_filepaths = [
            e for e in df_filepaths if df_filename_group in e.stem]

        colnames = ['g_id', 'cv', 'g_prop']
        dfs = [
            pl.read_parquet(e)
            .with_columns(pl.lit(e.stem.split('gprop_')[1]).alias(colnames[-1]))
            .select(colnames)
            for e in df_group_filepaths]

        # make sure that all the DataFrames have the same number of groups
        g_ns = pl.Series([e['g_id'].n_unique() for e in dfs])
        assert g_ns.n_unique() == 1

        df = pl.concat(dfs)
        g_props_str = ','.join(df['g_prop'].unique().sort())

        title = df_filename_group + 'g_prop_' + g_props_str
        output_filename = df_filename_group[:-1] + '.png'
        plot_combined_values_by_group_proportion_for_groups(
            df, colnames, title, output_path, output_filename)


def predict_inverse_logistic(
    xs: np.ndarray, horizontal_bias_param: float=0, 
    vertical_stretch_param: float=1) -> np.ndarray:
    """
    Predict the logit of the values for an inverse logistic
    """

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


def get_inverse_logistic_results_given_parameter_set(
    x: pl.Series, y: pl.Series) -> OptimizeResult:
    """
    Get the results for an inverse logistic regression model given a set of 
        parameters
    """

    horizontal_bias_param = 0
    vertical_stretch_param = 1
    xs = x.to_numpy()
    ys = y.to_numpy()
    initial_params = [horizontal_bias_param, vertical_stretch_param]
    bounds = [(None, None), (1e-6, None)]

    result = minimize(
        calculate_inverse_logistic_squared_error, initial_params, 
        args=(xs, ys), bounds=bounds)

    return result


def plot_inverse_logistic_results(
    x: pl.Series, y: pl.Series, bias_param: float, stretch_param: float,
    output_path: Path) -> None:
    """
    Plot the data and the predicted values for an inverse logistic regression
    """

    assert len(x) == len(y)

    xs = np.linspace(0.01, 0.99, 99)
    ys = predict_inverse_logistic(xs, bias_param, stretch_param)

    plt.scatter(xs, ys, color='red', alpha=0.5)
    plt.scatter(x.to_list(), y.to_list(), color='blue')

    plt.xlim(0, 1)

    output_filename = 'inverse_logistic_data_and_prediction.png'
    output_filepath = output_path / output_filename
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def predict_on_between_group_variance(output_path: Path):
    """
    Predict the group proportion from the between-group variance proportion 
        using an inverse logistic regression
    """

    df_filepath = output_path / 'results.parquet'

    df = pl.read_parquet(df_filepath)

    filter_df = df.filter(pl.col('g_n').eq(pl.col('i_n')))
    x_colname = 'btwn_var_prop'
    y_colname = 'g_prop'
    x = filter_df[x_colname]
    y = filter_df[y_colname]

    result = get_inverse_logistic_results_given_parameter_set(x, y)

    param_dict = {
        'horizontal_bias': result.x[0], 
        'vertical_stretch': result.x[1]}
    output_filename = 'inverse_logistic_params.json'
    output_filepath = output_path / output_filename
    with open(output_filepath, 'w') as f:
        json.dump(param_dict, f)

    plot_inverse_logistic_results(
        x, y, result.x[0], result.x[1], output_path)


if __name__ == '__main__':

    total = 72 ** 2
    output_path = Path.cwd() / 'output' / 'm02' / ('combo_n_' + str(total))
    output_path.mkdir(exist_ok=True, parents=True)

    main_analysis(total, output_path)
    plot_combined_values_by_group_proportion(output_path)
    predict_on_between_group_variance(output_path)
