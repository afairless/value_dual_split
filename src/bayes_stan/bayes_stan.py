#! /usr/bin/env python3

import pystan
import arviz as az
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def write_list_to_text_file(
    a_list: list, text_filename: str, overwrite: bool=False):
    """
    Writes a list of strings to a text file
    If 'overwrite' is 'True', any existing file by the name of 'text_filename'
        will be overwritten
    If 'overwrite' is 'False', list of strings will be appended to any existing
        file by the name of 'text_filename'

    :param a_list: a list of strings to be written to a text file
    :param text_filename: a string denoting the filepath or filename of text
        file
    :param overwrite: Boolean indicating whether to overwrite any existing text
        file or to append 'a_list' to that file's contents
    :return:
    """

    if overwrite:
        append_or_overwrite = 'w'
    else:
        append_or_overwrite = 'a'

    try:
        text_file = open(text_filename, append_or_overwrite, encoding='utf-8')
        for e in a_list:
            text_file.write(str(e))
            text_file.write('\n')

    finally:
        text_file.close()


def dummy_code_two_level_hierarchical_categories(
    group_n: int, individual_n: int) -> np.ndarray:
    """
    Create a dummy-coded (one-hot-encoded) Numpy array for two-level 
        hierarchical categories, so that every higher-level category is paired
        with every lower-level category
    """

    assert group_n > 0
    assert individual_n > 0

    group_eye_array = np.eye(group_n)
    group_array = np.repeat(group_eye_array, individual_n, axis=0)
    individual_eye_array = np.eye(individual_n)
    individual_array = np.tile(individual_eye_array, (group_n, 1))
    combined_array = np.concatenate((group_array, individual_array), axis=1)

    return combined_array


def get_stan_data_from_df(df: pd.DataFrame) -> dict:
    """
    Create a dictionary from DataFrame of data to be passed to a Stan model
    """

    # verify that all group IDs appear the same number of times
    g_id_srs = df['g_id']
    assert isinstance(g_id_srs, pd.Series)
    assert g_id_srs.value_counts().nunique() == 1

    i_id_srs = df['i_id']
    assert isinstance(i_id_srs, pd.Series)
    assert i_id_srs.value_counts().nunique() == 1

    # one-hot encode groups and individuals
    n = len(df)
    g_n = g_id_srs.value_counts().unique()[0]
    i_n = i_id_srs.value_counts().unique()[0]
    assert isinstance(g_n, np.int64)
    assert isinstance(i_n, np.int64)
    x = dummy_code_two_level_hierarchical_categories(g_n, i_n)
    assert x.shape[0] == n
    assert x.shape[1] == g_n + i_n
    g_x = x[:, :g_n]
    i_x = x[:, g_n:]

    cv_id_srs = df['i_id']
    assert isinstance(cv_id_srs, pd.Series)
    y = cv_id_srs.values

    stan_data = {'N': n, 'G': g_n, 'I': i_n, 'g_x': g_x, 'i_x': i_x, 'y': y}

    return stan_data


def save_summaries(fit_df: pd.DataFrame, fit_model, output_path: Path):

    filename = 'fit_df.csv'
    output_filepath = output_path / filename
    fit_df.to_csv(output_filepath)

    filename = 'fit_df.parquet'
    output_filepath = output_path / filename
    fit_df.to_parquet(output_filepath)

    # text summary of means, sd, se, and quantiles for parameters, n_eff, & Rhat
    fit_stansummary = fit_model.stansummary()
    output_filepath = output_path / 'stansummary.txt'
    write_list_to_text_file([fit_stansummary], output_filepath.as_posix(), True)

    # same summary as for 'stansummary', but in matrix/dataframe form instead of text
    fit_summary_df = pd.DataFrame(
        fit_model.summary()['summary'],
        index=fit_model.summary()['summary_rownames'],
        columns=fit_model.summary()['summary_colnames'])
    output_filepath = output_path / 'stansummary.csv'
    fit_summary_df.to_csv(output_filepath, index=True)
    output_filepath = output_path / 'stansummary.parquet'
    fit_summary_df.to_parquet(output_filepath, index=True)


def save_plots(
    fit_df: pd.DataFrame, fit_model, output_path: Path):

    # plot parameters
    ##################################################

    az_stan_data = az.from_pystan(
        posterior=fit_model,
        # posterior_predictive='predicted_y_given_x',
        observed_data=['y'])


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
        credible_interval=0.9, point_estimate='mean', show=show)
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
        credible_interval=0.9, r_hat=True, ess=True, show=show)
    output_filepath = output_path / 'plot_forest.png'
    plt.savefig(output_filepath)
    plt.clf()
    plt.close()

    # look at model estimations of parameters, r-hat, and ess
    az.plot_forest(
        az_stan_data, kind='ridgeplot', var_names=parameter_names,
        credible_interval=0.9, r_hat=True, ess=True,
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
            x[:, x_col_idx], predicted_y_df, credible_interval=0.5, show=show)
        az.plot_hpd(
            x[:, x_col_idx], predicted_y_df, credible_interval=0.9, show=show)
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
        az_stan_data, var_names=parameter_names, credible_interval=0.9,
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
        credible_interval=0.9, show=show)
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


def main():

    total = 72 ** 2
    input_path = Path.cwd() / 'output' / ('combo_n_' + str(total))
    stan_filename = 'bayes_stan.stan'
    stan_filepath = Path.cwd() / 'src' / 'stan_code' / stan_filename
    output_path = Path.cwd() / 'output'
    output_path.mkdir(exist_ok=True, parents=True)


    df_filepaths = input_path.glob('*.parquet')
    df_filepaths = [
        e for e in df_filepaths if e.stem[:3] == 'gn_' and 'gprop' in e.stem]

    df_filename_groups = list(
        set([e.stem.split('gprop')[0] for e in df_filepaths]))
    df_filename_group_stem = df_filename_groups[0]
    # df_filepath = [
    #     e for e in df_filepaths if 'gn_72_in_72_gprop_0.9' in e.stem][0]
    df_group_filepaths = [
        e for e in df_filepaths if df_filename_group_stem in e.stem]

    probability_estimation_metrics = []
    for df_filepath in df_group_filepaths:

        print('Processing:', df_filepath.stem)

        output_subpath = output_path / df_filepath.stem.replace('.', '_')
        output_subpath.mkdir(exist_ok=True, parents=True)

        df = pd.read_parquet(df_filepath)
        stan_data = get_stan_data_from_df(df)
        stan_model = pystan.StanModel(file=stan_filepath.as_posix())

        fit_model = stan_model.sampling(
           data=stan_data, iter=300, chains=1, warmup=150, thin=1, seed=708869)
        # fit_model = stan_model.sampling(
        #    data=stan_data, iter=2000, chains=4, warmup=1000, thin=1, seed=22074)
        # fit_model = stan_model.sampling(
        #     data=stan_data, iter=2000, chains=4, warmup=1000, thin=2, seed=22074)

        # all samples for all parameters, predicted values, and diagnostics
        #   number of rows = number of 'iter' in 'StanModel.sampling' call
        fit_df = fit_model.to_dataframe()

        save_summaries(fit_df, fit_model, output_subpath)
        save_plots(fit_df, fit_model, output_subpath)

        # fit_model.stansummary()
        true_g_v = df[['g_id', 'g_v']].groupby('g_id').mean()
        true_i_v = df[['i_id', 'i_v']].groupby('i_id').mean()
        g_prop = float(df_filepath.stem.split('gprop_')[1])
        mean_error = fit_df['g_prob'].mean() - g_prop
        median_error = fit_df['g_prob'].median() - g_prop
        probability_estimation_metrics.append(
            (df_filepath.stem, g_prop, mean_error, median_error))

        # fit_df.columns

        title = df_filepath.stem
        output_filename = 'group_probability_true_vs_posterior.png'
        output_filepath = output_subpath / output_filename
        plot_group_probability_true_vs_posterior(
            fit_df, g_prop, title, output_filepath)


        # limit number of columns to plot, so plot is readable
        v_n = 72
        colnames = [e for e in fit_df.columns if 'g_v' in e]
        colnames = colnames[:v_n]
        title = df_filepath.stem
        output_filename = 'group_values_true_vs_posterior.png'
        output_filepath = output_subpath / output_filename
        plot_values_true_vs_posterior(
            fit_df, colnames, true_g_v, title, output_filepath)

        colnames = [e for e in fit_df.columns if 'i_v' in e]
        colnames = colnames[:v_n]
        title = df_filepath.stem
        output_filename = 'individual_values_true_vs_posterior.png'
        output_filepath = output_subpath / output_filename
        plot_values_true_vs_posterior(
            fit_df, colnames, true_i_v, title, output_filepath)


    probability_estimation_df = pd.DataFrame(probability_estimation_metrics)
    colnames = ['filename', 'true_g_prop', 'mean_error', 'median_error']
    probability_estimation_df.columns = colnames 

    output_filename = df_filename_group_stem + 'g_prop_error.csv'
    output_filepath = output_path / output_filename
    probability_estimation_df.to_csv(output_filepath)

    output_filename = df_filename_group_stem + 'g_prop_error.parquet'
    output_filepath = output_path / output_filename
    probability_estimation_df.to_parquet(output_filepath)




 
    # matrix[N, 1] g_vs;
    # matrix[N, 1] i_vs;
    # vector[N] g_vs;
    # vector[N] i_vs;
    # g_vs = g_x * g_v;
    # i_vs = i_x * i_v;
    # vs = append_col(g_vs, i_vs);



if __name__ == '__main__':
    main()
