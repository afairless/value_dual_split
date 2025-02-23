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

import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    from src.common import (
        save_dataframe_to_csv_and_parquet,
        partition_group_variance,
        calculate_inverse_logistic_squared_error,
        )
except:
    from common import (
        save_dataframe_to_csv_and_parquet,
        partition_group_variance,
        calculate_inverse_logistic_squared_error,
        )

try:
    from src.m03 import (
        dummy_code_two_level_hierarchical_categories,
        )
except:
    from m03 import (
        dummy_code_two_level_hierarchical_categories,
        )

try:
    from src.plot import (
        plot_group_proportion_estimates_by_group_individual_ratios,
        plot_metric_by_group_individual_ratios,
        plot_metric_by_metric,
        plot_combined_values_by_group_proportion_for_groups,
        plot_inverse_logistic_results,
        )
except:
    from plot import (
        plot_group_proportion_estimates_by_group_individual_ratios,
        plot_metric_by_group_individual_ratios,
        plot_metric_by_metric,
        plot_combined_values_by_group_proportion_for_groups,
        plot_inverse_logistic_results,
        )


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
    g_n: int, i_n: int, g_prop: float, df: pl.DataFrame
    ) -> tuple[list[float], list[float]]:

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

    compiled_result = [
        g_n, i_n, 
        g_prop, float(result.x[0]), 
        final_error, result.success,
        group_variance.btwn_group_var,
        group_variance.wthn_group_var,
        group_variance.total_var,
        group_variance.btwn_var_prop,
        group_variance.wthn_var_prop,
        icc1, icc2, icc3]

    individual_estimates = result.x[1:].tolist()

    return compiled_result, individual_estimates


def get_results_given_parameters(
    factor_df: pl.DataFrame, g_props: list[float], output_path: Path
    ) -> pl.DataFrame:

    metric_summaries = []
    individual_parameters = []
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

            metric_summary, individual_params = (
                get_results_given_parameter_set(g_n, i_n, g_prop, df))

            metric_summaries.append(metric_summary)
            g_param_names = ['g_' + str(e) for e in range(g_n)]
            i_param_names = ['i_' + str(e) for e in range(i_n)]
            individual_param_names = g_param_names + i_param_names
            individual_params_dict = dict(zip(
                individual_param_names, individual_params))
            individual_parameters.append(individual_params_dict)

    colnames = [
        'g_n', 'i_n', 'g_prop', 'g_prop_est', 'error', 'success',
        'btwn_group_var', 'wthn_group_var', 'total_var', 
        'btwn_var_prop', 'wthn_var_prop',
        'icc1', 'icc2', 'icc3']
    metric_df = pl.DataFrame(metric_summaries, orient='row')
    metric_df.columns = colnames

    # to convert a list of dictionaries to structs to be used as a Polars 
    #   DataFrame column, one must ensure that all the dictionary keys are
    #   present in each dictionary
    # (Actually, it might work if all keys are present only in the first
    #   dictionary, but I'll take the uniform approach)
    g_param_names = ['g_' + str(e) for e in range(factor_df[:, 0].max())]
    i_param_names = ['i_' + str(e) for e in range(factor_df[:, 1].max())]

    for param_dict in individual_parameters:
        for g_param_name in g_param_names:
            if g_param_name not in param_dict:
                param_dict[g_param_name] = None
        for i_param_name in i_param_names:
            if i_param_name not in param_dict:
                param_dict[i_param_name] = None

    metric_df = metric_df.with_columns(
        pl.Series(individual_parameters).alias('individual_parameters'))

    return metric_df


def main_analysis(total: int, output_path: Path):

    # pl.Config.set_tbl_cols(14)
    # pl.Config.set_tbl_rows(24)

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
    g_props = [0.05] + [e/100 for e in range(10, 100, 10)] + [0.95]
    result_df = get_results_given_parameters(factor_df, g_props, output_path)
    # result_df.filter(pl.col('g_prop').eq(0.9)).select(pl.col('g_prop', 'g_prop_est', 'btwn_var_prop'))

    filename_stem = 'results'
    save_dataframe_to_csv_and_parquet(result_df, filename_stem, output_path)


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

    total = 12 ** 2
    output_path = Path.cwd() / 'output' / 'm02' / ('combo_n_' + str(total))
    output_path.mkdir(exist_ok=True, parents=True)

    main_analysis(total, output_path)
    plot_combined_values_by_group_proportion(output_path)
    predict_on_between_group_variance(output_path)
