#! /usr/bin/env python3

import numpy as np
import pandas as pd
import polars as pl

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.cm as cm
 
try:
    from src.common import (
        predict_inverse_logistic,
        )
except:
    from common import (
        predict_inverse_logistic,
        )

 
##################################################
# PLOT BIVARIATE METRICS BY GROUP PROPORTION
##################################################
 
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

 
##################################################
# PLOT ESTIMATES AND TRUE VALUES FOR EACH PARAMETER
##################################################
 
def plot_parameter_estimates_vs_true_values_correlation(
    plot_case_df: pl.DataFrame, output_path: Path, output_filename: str
    ) -> None:
    """
    Plot the correlation between the estimated and true values of group and 
        individual parameters
    """

    true_params = plot_case_df['true_parameters'][0]
    est_params = plot_case_df['parameter_estimates'][0]

    for k in true_params:
        assert k in est_params.keys()

    true_df = pl.DataFrame(true_params).transpose(include_header=True)
    true_df.columns = ['param_name', 'true_value']
    est_df = pl.DataFrame(est_params).transpose(include_header=True)
    est_df.columns = ['param_name', 'estimate']

    plot_df = true_df.join(est_df, on='param_name')

    # filter out missing values
    mask = (
        pl.col(plot_df.columns[0]).is_not_null() &
        pl.col(plot_df.columns[1]).is_not_null() &
        pl.col(plot_df.columns[2]).is_not_null())
    plot_df = plot_df.filter(mask).with_row_index()

    plot_g_df = plot_df.filter(pl.col('param_name').str.starts_with('g_'))
    plot_i_df = plot_df.filter(pl.col('param_name').str.starts_with('i_'))

    plt.scatter(
        plot_g_df['true_value'], plot_g_df['estimate'], 
        color='blue', alpha=0.5)
    plt.scatter(
        plot_i_df['true_value'], plot_i_df['estimate'], 
        color='orange', alpha=0.5)

    plt.xlabel('True value')
    plt.ylabel('Estimated value')

    output_filepath = output_path / output_filename
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

 
def plot_parameter_estimates_vs_true_values_by_index(
    plot_case_df: pl.DataFrame, output_path: Path, output_filename: str
    ) -> None:
    """
    Plot the estimated and true values of group and individual parameters
    """

    true_params = plot_case_df['true_parameters'][0]
    est_params = plot_case_df['parameter_estimates'][0]

    for k in true_params:
        assert k in est_params.keys()

    true_df = pl.DataFrame(true_params).transpose(include_header=True)
    true_df.columns = ['param_name', 'true_value']
    est_df = pl.DataFrame(est_params).transpose(include_header=True)
    est_df.columns = ['param_name', 'estimate']

    plot_df = true_df.join(est_df, on='param_name')

    # filter out missing values
    mask = (
        pl.col(plot_df.columns[0]).is_not_null() &
        pl.col(plot_df.columns[1]).is_not_null() &
        pl.col(plot_df.columns[2]).is_not_null())
    plot_df = plot_df.filter(mask).with_row_index()

    plot_g_df = plot_df.filter(pl.col('param_name').str.starts_with('g_'))
    plot_i_df = plot_df.filter(pl.col('param_name').str.starts_with('i_'))

    plt.scatter(
        plot_g_df['index'], plot_g_df['true_value'], 
        color='green', alpha=0.5, label='g_true')
    plt.scatter(
        plot_g_df['index'], plot_g_df['estimate'], 
        color='blue', alpha=0.5, label='g_estimate')
    plt.scatter(
        plot_i_df['index'], plot_i_df['true_value'], 
        color='green', alpha=0.5, label='i_true')
    plt.scatter(
        plot_i_df['index'], plot_i_df['estimate'], 
        color='orange', alpha=0.5, label='i_estimate')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend(fontsize=8)

    output_filepath = output_path / output_filename
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

 
##################################################
# PLOT INVERSE LOGISTIC RESULTS
##################################################
 
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
    plt.title(
        'Use inverse logistic to adjust between-group variance proportion\n '
        'as a prediction of the group proportion')
    plt.xlabel('Between-group variance proportion, btwn_var_prop')
    plt.ylabel('True group proportion, g_prop')

    output_filename = 'inverse_logistic_data_and_prediction.png'
    output_filepath = output_path / output_filename
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


##################################################
# PLOT BAYESIAN RESULTS
##################################################
 
def save_plots(
    fit_df: pd.DataFrame, fit_model, output_path: Path):

    # plot parameters
    ##################################################

    az_stan_data = az.from_cmdstanpy(
        posterior=fit_model,
        # posterior_predictive='predicted_y_given_x',
        # observed_data=['y']
        )


    az.style.use('arviz-darkgrid')
    parameter_names = ['g_prob']
    show = False


    # plot chain autocorrelation
    ##################################################

    print('Plotting autocorrelation')
    az.plot_autocorr(az_stan_data, var_names=parameter_names, show=show)
    output_filepath = output_path / 'plot_autocorr.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_autocorr(
        az_stan_data, var_names=parameter_names, combined=True, show=show)
    output_filepath = output_path / 'plot_autocorr_combined_chains.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # plot parameter density
    ##################################################

    print('Plotting density')
    az.plot_density(
        az_stan_data, var_names=parameter_names, outline=False, shade=0.7,
        hdi_prob=0.9, point_estimate='mean', show=show)
    output_filepath = output_path / 'plot_density.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # plot distribution
    ##################################################

    print('Plotting distribution')
    az.plot_dist(
        fit_df[parameter_names[0]], rug=True,
        quantiles=[0.25, 0.5, 0.75], show=show)
    output_filepath = output_path / 'plot_distribution.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_dist(
        fit_df[parameter_names[0]], rug=True, cumulative=True,
        quantiles=[0.25, 0.5, 0.75], show=show)
    output_filepath = output_path / 'plot_distribution_cumulative.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # plot ESS across local parts of distribution
    ##################################################

    print('Plotting ESS')
    az.plot_ess(
        az_stan_data, var_names=parameter_names, kind='local', n_points=10,
        rug=True, extra_methods=True, show=show)
    output_filepath = output_path / 'plot_ess_local.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_ess(
        az_stan_data, var_names=parameter_names, kind='quantile', n_points=10,
        rug=True, extra_methods=True, show=show)
    output_filepath = output_path / 'plot_ess_quantile.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_ess(
        az_stan_data, var_names=parameter_names, kind='evolution', n_points=10,
        rug=True, extra_methods=True, show=show)
    output_filepath = output_path / 'plot_ess_evolution.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # forest plots
    ##################################################

    print('Plotting forest plots')

    # look at model estimations of parameters, r-hat, and ess
    az.plot_forest(
        az_stan_data, kind='forestplot', var_names=parameter_names,
        linewidth=6, markersize=8,
        hdi_prob=0.9, r_hat=True, ess=True, show=show)
    output_filepath = output_path / 'plot_forest.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    # look at model estimations of parameters, r-hat, and ess
    az.plot_forest(
        az_stan_data, kind='ridgeplot', var_names=parameter_names,
        hdi_prob=0.9, r_hat=True, ess=True,
        ridgeplot_alpha=0.5, ridgeplot_overlap=2, ridgeplot_kind='auto',
        show=show)
    output_filepath = output_path / 'plot_forest_ridge.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    """
    # HPD plot
    ##################################################

    print('Plotting HPD plots')

    # look at model estimations of parameters, r-hat, and ess
    predicted_y_colnames = [e for e in fit_df.columns if 'y_given_x' in e]
    predicted_y_df = fit_df[predicted_y_colnames]

    for x_col_idx in range(x.shape[1]):
        plt.scatter(x[:, x_col_idx], y)
        az.plot_hpd(
            x[:, x_col_idx], predicted_y_df, hdi_prob=0.5, show=show)
        az.plot_hpd(
            x[:, x_col_idx], predicted_y_df, hdi_prob=0.9, show=show)
        filename = 'plot_hpd_x' + str(x_col_idx) + '.png'
        output_filepath = output_path / filename
        plt.savefig(output_filepath)
        plt.clf()
        plt.close()
    """


    # plot KDE
    ##################################################

    print('Plotting KDE plots')

    az.plot_kde(
        fit_df[parameter_names[0]],
        contour=True, show=show)
    output_filepath = output_path / 'plot_kde_contour.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_kde(
        fit_df[parameter_names[0]],
        contour=False, show=show)
    output_filepath = output_path / 'plot_kde_no_contour.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # MCSE statistics and plots
    ##################################################

    print('Plotting MCSE plots')

    az.mcse(az_stan_data, var_names=parameter_names, method='mean')
    az.mcse(az_stan_data, var_names=parameter_names, method='sd')
    az.mcse(az_stan_data, var_names=parameter_names, method='quantile', prob=0.1)

    az.plot_mcse(
        az_stan_data, var_names=parameter_names, errorbar=True, n_points=10)
    output_filepath = output_path / 'plot_mcse_errorbar.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_mcse(
        az_stan_data, var_names=parameter_names, extra_methods=True,
        n_points=10)
    output_filepath = output_path / 'plot_mcse_extra_methods.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()



    """
    # plot pair
    ##################################################

    print('Plotting pair plots')

    az.plot_pair(
        az_stan_data, var_names=parameter_names, kind='scatter',
        divergences=True, show=show)
    output_filepath = output_path / 'plot_pair_scatter.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_pair(
        az_stan_data, var_names=parameter_names, kind='kde',
        divergences=True, show=show)
    output_filepath = output_path / 'plot_pair_kde.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # plot parameters in parallel
    ##################################################

    print('Plotting parallel plots')

    az.plot_parallel(
        az_stan_data, var_names=parameter_names, colornd='blue', show=show)
    output_filepath = output_path / 'plot_parallel.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    """


    # plot parameters in parallel
    ##################################################

    az.plot_posterior(
        az_stan_data, var_names=parameter_names, hdi_prob=0.9,
        point_estimate='mean', show=show)
    output_filepath = output_path / 'plot_posterior.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    """
    # plot predictive check
    ##################################################

    print('Plotting predictive checks')

    az.plot_ppc(
        az_stan_data, kind='kde', data_pairs={'y': 'predict_y_given_x'},
        random_seed=483742, show=show)
    output_filepath = output_path / 'plot_predictive_check_kde.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_ppc(
        az_stan_data, kind='cumulative', data_pairs={'y': 'predict_y_given_x'},
        random_seed=483742, show=show)
    output_filepath = output_path / 'plot_predictive_check_cumulative.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_ppc(
        az_stan_data, kind='scatter', data_pairs={'y': 'predict_y_given_x'},
        random_seed=483742, jitter=0.5, show=show)
    output_filepath = output_path / 'plot_predictive_check_scatter.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()
    """


    # plot chain rank order statistics
    ##################################################
    # each chain should show approximately a uniform distribution:
    #   https://arxiv.org/pdf/1903.08008
    #   Vehtari, Gelman, Simpson, Carpenter, Bürkner (2020)
    #   Rank-normalization, folding, and localization: An improved R for
    #       assessing convergence of MCMC

    az.plot_rank(
        az_stan_data, var_names=parameter_names, kind='bars', show=show)
    output_filepath = output_path / 'plot_rank_bars.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    az.plot_rank(
        az_stan_data, var_names=parameter_names, kind='vlines', show=show)
    output_filepath = output_path / 'plot_rank_vlines.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # plot traces
    ##################################################

    print('Plotting traces')

    az.plot_trace(
        az_stan_data, var_names=parameter_names, legend=False, show=show)
    output_filepath = output_path / 'plot_trace.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


    # plot distributions on violin plot
    ##################################################

    print('Plotting violin plots')

    az.plot_violin(
        az_stan_data, var_names=parameter_names, rug=True,
        hdi_prob=0.9, show=show)
    output_filepath = output_path / 'plot_violin.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_group_probability_true_vs_posterior(
    fit_df: pd.DataFrame, g_prop: float, title: str, output_filepath: Path
    ) -> None:
    """
    Plot the posterior distribution of the group probability against the true 
        group probability
    """

    plt.violinplot(fit_df['g_prob'], showmeans=True, showmedians=True)
    plt.scatter([1]*len(fit_df), fit_df['g_prob'], alpha=0.2)
    plt.scatter(1, g_prop, color='orange', s=100, alpha=0.9)
 
    plt.title(title)
    txt = 'orange dot marks true group probability'
    plt.suptitle(txt)

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


def plot_values_true_vs_posterior(
    fit_df: pd.DataFrame, colnames: list, true_g_v: pd.Series, title: str, 
    output_filepath: Path) -> None:
    """
    Plot the posterior distribution of the values against the true value 
    """

    assert len(colnames) == len(true_g_v)
    posteriors_df = fit_df[colnames]
    assert isinstance(posteriors_df, pd.DataFrame)
    true_g_v_xs = range(1, len(true_g_v)+1)
 
    fig_width = max(60, len(colnames))
    plt.figure(figsize=(fig_width, 6))
    plt.violinplot(posteriors_df.values, showmeans=True, showmedians=True)
    plt.scatter(true_g_v_xs, true_g_v, color='orange', s=100, alpha=0.9)
    plt.tight_layout()
 
    plt.title(title)

    plt.savefig(output_filepath)
    plt.clf()
    plt.close()


