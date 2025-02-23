#! /usr/bin/env python3

import numpy as np
import polars as pl
import pingouin as pg

import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    from src.common import (
        partition_group_variance,
        )
except:
    from common import (
        partition_group_variance,
        )

#import rpy2
#from rpy2.robjects.packages import importr
#base = importr('base')
#utils = importr('utils')
# errors while trying to install R packages
#utils.install_packages('Matrix')
#utils.install_packages('MASS')
#utils.install_packages('reformulas')
#utils.install_packages('lme4')
#Matrix = importr('Matrix')
#MASS = importr('MASS')
#reformulas = importr('reformulas')
#lme4 = importr('lme4')

def toy_model_01():
    """
    This toy model represents a nested structure (with 'b's in 'a's) with some
        value 'v' associated with each 'a-b' pair
    The code shows how to apportion the variance between 'a' and 'b' so that
        new, hypothetical 'a-b' pairs can be created and assigned a hypothetical
        value
    Apportioning variance is the classical statistical way of thinking about 
        this problem, but the classical context is a model using some form of 
        squared error as a loss function; but one can presumably apply the same 
        logic to absolute error or other measures of variability, as well
    """

    df = pl.DataFrame({
        'a': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        'v': [1, 3, 5, 3, 5, 7, 3, 6, 9, 5, 7, 9],
        'b': range(12)})

    group_means_df = df.drop('b').group_by('a').mean()
    btwn_group_var = group_means_df['v'].var() 

    group_var_df = df.drop('b').group_by('a').agg(pl.var('v'))
    wthn_group_var = group_var_df['v'].mean() 

    assert isinstance(btwn_group_var, float)
    assert isinstance(wthn_group_var, float)
    total_var = btwn_group_var + wthn_group_var 
    
    btwn_var_prop = btwn_group_var / total_var
    wthn_var_prop = wthn_group_var / total_var

    # re-assign 'b11' to 'a0'
    ##################################################

    a_id = 0
    b_id = 11
    a_value = group_means_df.filter(pl.col('a') == a_id).select('v')[0, 0]
    b_value = df.filter(pl.col('b') == b_id).select('v')[0, 0]

    new_value = (btwn_var_prop * a_value) + (wthn_var_prop * b_value)


def toy_model_02():
    """
    These toy models represent a simple regression ('x' as a predictor; 'y' as 
        an outcome) with a nested structure ('group') as a grouping variable
    The groups have a common slope but different intercepts
    The code shows how different fits of a linear mixed model from 'statsmodels'
        are reported in the results
    """

    #data = sm.datasets.get_rdataset('dietox', 'geepack').data
    #md = smf.mixedlm('Weight ~ Time', data, groups=data['Pig'])
    #mdf = md.fit()
    #mdf.summary()


    # DATA SET 1
    ##################################################

    df = pl.DataFrame({
        #'group': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
        'group': [ 1,  1,  1,  2,  2,   2,  3,  3,  3],
        'x':     [ 1,  2,  3,  3,  4,   5,  5,  6,  7],
        'y':     [12, 14, 16, 15, 17,  19, 21, 23, 25],
        #'y':     [2, 4, 6, 5, 7,  9, 11, 13, 15],
        })

    # default model configuration, which includes a global intercept
    #   and random intercepts
    ##################################################

    md = smf.mixedlm('y ~ x', df, groups=df['group'])
    result = md.fit()

    # all groups have a common slope 2, which is correctly estimated for 'x'

    #>>> result.summary()
    #<class 'statsmodels.iolib.summary2.Summary'>
    #"""
    #          Mixed Linear Model Regression Results
    #=========================================================
    #Model:              MixedLM  Dependent Variable:  y      
    #No. Observations:   9        Method:              REML   
    #No. Groups:         3        Scale:               0.0000 
    #Min. group size:    3        Log-Likelihood:      46.7740
    #Max. group size:    3        Converged:           Yes    
    #Mean group size:    3.0                                  
    #---------------------------------------------------------
    #          Coef.   Std.Err.     z      P>|z| [0.025 0.975]
    #---------------------------------------------------------
    #Intercept 10.000     0.309     32.404 0.000  9.395 10.605
    #x          2.000     0.000 596244.843 0.000  2.000  2.000
    #Group Var  0.286 21992.822                               
    #=========================================================

    # the 3 groups have intercepts of 9, 10, and 11, which is correctly 
    #   estimated in the global intercept of 10 (above) plus the random effects 
    #   for groups 1, 2, and 3 (below) as -1, 0, and 1

    #>>> result.random_effects
    #{np.int64(1): Group   -0.000004
    #dtype: float64, np.int64(2): Group   -1.000007
    #dtype: float64, np.int64(3): Group    0.999999
    #dtype: float64}


    # model configuration suppresses global intercept, but retains random 
    #   intercepts
    ##################################################

    md = smf.mixedlm('y ~ x + 0', df, groups=df['group'])
    result = md.fit()

    # all groups have a common slope 2, which is correctly estimated for 'x'
    # without global intercept, 'Group Var' is much larger (37.75) for this 
    #   model (below) than for the prior model (0.286, above)
    # for both models, residual term (Scale) is so small as to show up as 0
    # without global intercept, 'Log-Likelihood' is much lower (27.28) for this 
    #   model (below) than for the prior model (46.77, above)

    #>>> result.summary()
    #<class 'statsmodels.iolib.summary2.Summary'>
    #"""
    #          Mixed Linear Model Regression Results
    #=========================================================
    #Model:              MixedLM  Dependent Variable:  y      
    #No. Observations:   9        Method:              REML   
    #No. Groups:         3        Scale:               0.0000 
    #Min. group size:    3        Log-Likelihood:      27.2767
    #Max. group size:    3        Converged:           Yes    
    #Mean group size:    3.0                                  
    #---------------------------------------------------------
    #          Coef.   Std.Err.      z     P>|z| [0.025 0.975]
    #---------------------------------------------------------
    #x          2.000      0.000 54870.323 0.000  2.000  2.000
    #Group Var 37.750 267410.668                              
    #=========================================================

    # the 3 groups have intercepts of 9, 10, and 11, which is correctly 
    #   estimated ) in the random effects for groups 1, 2, and 3 (below), unlike 
    #   the prior model, where these intercepts were apportioned between the
    #   global intercept and the random effects
    # my take-away from the lower log-likelihood for this model is that the
    #   global intercept doesn't help the model (given the cost of the extra
    #   parameter), because the random effects adequately estimate the random
    #   intercepts without the global intercept

    #"""
    #>>> result.random_effects
    #{np.int64(1): Group    10.000023
    #dtype: float64, np.int64(2): Group    9.000007
    #dtype: float64, np.int64(3): Group    11.00002
    #dtype: float64}


def toy_model_03():
    """
    These toy models represent a simple regression ('x' as a predictor; 'y' as 
        an outcome) with a nested structure ('group') as a grouping variable
    The groups have a common intercept but different slopes
    The code shows how different fits of a linear mixed model from 'statsmodels'
        are reported in the results
    """

    # DATA SET 1
    ##################################################

    df = pl.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
        'x':     [  0,   1,   2,   0,   2,   4,   0,   1,   2],
        'y':     [ 10,  11,  12,  10,   6,   2,  10,  13,  16],
        })

    # default model configuration, which includes a global intercept
    #   and random intercepts but not random slopes
    ##################################################

    md = smf.mixedlm('y ~ x', df, groups=df['group'])
    result = md.fit()

    #>>> result.summary()
    #<class 'statsmodels.iolib.summary2.Summary'>
    #"""
    #        Mixed Linear Model Regression Results
    #======================================================
    #Model:            MixedLM Dependent Variable: y       
    #No. Observations: 9       Method:             REML    
    #No. Groups:       3       Scale:              9.4133  
    #Min. group size:  3       Log-Likelihood:     -21.2852
    #Max. group size:  3       Converged:          Yes     
    #Mean group size:  3.0                                 
    #------------------------------------------------------
    #            Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    #------------------------------------------------------
    #Intercept   11.249    2.184  5.150 0.000  6.968 15.531
    #x           -0.937    0.921 -1.018 0.309 -2.742  0.868
    #Group Var    6.655    3.993                           
    #======================================================

    #"""
    #>>> result.random_effects
    #{np.str_('a'): Group    0.467318
    #dtype: float64, np.str_('b'): Group   -2.293805
    #dtype: float64, np.str_('c'): Group    1.826487
    #dtype: float64}


    # model configuration, which includes a global intercept, random 
    #   intercepts, and random slopes
    ##################################################

    md = smf.mixedlm('y ~ x', df, groups=df['group'], re_formula='~x')
    result = md.fit()

    # with random slopes, this model's residuals (below) sum to nearly zero
    #   ('Scale'), much lower than the prior model's substantial error (where
    #   'Scale' = 9.41)
    # however, the log-likelihood for ths model (18.55) is much higher than
    #   the prior model's log-likelihood (-21.29); I would expect it to be
    #   lower; I'm clearly misunderstanding/misremembering something

    #>>> result.summary()
    #<class 'statsmodels.iolib.summary2.Summary'>
    #"""
    #          Mixed Linear Model Regression Results
    #=========================================================
    #Model:              MixedLM  Dependent Variable:  y      
    #No. Observations:   9        Method:              REML   
    #No. Groups:         3        Scale:               0.0000 
    #Min. group size:    3        Log-Likelihood:      18.5493
    #Max. group size:    3        Converged:           Yes    
    #Mean group size:    3.0                                  
    #---------------------------------------------------------
    #              Coef.   Std.Err.   z    P>|z| [0.025 0.975]
    #---------------------------------------------------------
    #Intercept     10.000     0.286 35.000 0.000  9.440 10.560
    #x              0.667     0.782  0.852 0.394 -0.867  2.200
    #Group Var      0.245                                     
    #Group x x Cov -0.082 12900.224                           
    #x Var          1.837 52328.144                           
    #=========================================================

    #"""
    #>>> result.random_effects
    #{np.str_('a'): Group    7.164249e-07
    #x        3.333565e-01
    #dtype: float64, np.str_('b'): Group    1.576503e-07
    #x       -2.666638e+00
    #dtype: float64, np.str_('c'): Group    0.000001
    #x        2.333361
    #dtype: float64}

    # add global intercept (~10) and group intercept (~0 for all groups) to get
    #   each group's intercept (~10 for all groups)
    a_intercept = result.fe_params['Intercept'] + result.random_effects[np.str_('a')]['Group']
    b_intercept = result.fe_params['Intercept'] + result.random_effects[np.str_('b')]['Group']
    c_intercept = result.fe_params['Intercept'] + result.random_effects[np.str_('c')]['Group']
    assert np.allclose(10, a_intercept, atol=1e-6, rtol=1e-6)
    assert np.allclose(10, b_intercept, atol=1e-6, rtol=1e-6)
    assert np.allclose(10, c_intercept, atol=1e-6, rtol=1e-6)

    # add global slope (~2/3) and group slope to get each group's slope
    a_slope = result.fe_params['x'] + result.random_effects[np.str_('a')]['x']
    b_slope = result.fe_params['x'] + result.random_effects[np.str_('b')]['x']
    c_slope = result.fe_params['x'] + result.random_effects[np.str_('c')]['x']
    assert np.allclose( 1, a_slope, atol=1e-6, rtol=1e-6)
    assert np.allclose(-2, b_slope, atol=1e-4, rtol=1e-4)
    assert np.allclose( 3, c_slope, atol=1e-4, rtol=1e-4)


    # model configuration, which includes a global intercept and random slopes,
    #   but no random intercepts
    ##################################################

    md = smf.mixedlm('y ~ x', df, groups=df['group'], re_formula='~x-1')
    result = md.fit()

    # with random slopes, this model's residuals (below) sum to nearly zero
    #   ('Scale'), much lower than the prior model's substantial error (where
    #   'Scale' = 9.41) and on par with the second model (above)
    # however, the log-likelihood for ths model (39.19) is much higher than
    #   the first model's log-likelihood (-21.29) and that of the second one
    #   (18.55); I guess I assumed that it was the negative log-likelihood,
    #   but better models seem to be more positive; this one should be the
    #   better model, compared to the second one, because it has fewer 
    #   parameters with equivalent performance

    #>>> result.summary()
    #<class 'statsmodels.iolib.summary2.Summary'>
    #"""
    #          Mixed Linear Model Regression Results
    #=========================================================
    #Model:              MixedLM  Dependent Variable:  y      
    #No. Observations:   9        Method:              REML   
    #No. Groups:         3        Scale:               0.0000 
    #Min. group size:    3        Log-Likelihood:      39.1921
    #Max. group size:    3        Converged:           Yes    
    #Mean group size:    3.0                                  
    #---------------------------------------------------------
    #          Coef.   Std.Err.     z      P>|z| [0.025 0.975]
    #---------------------------------------------------------
    #Intercept 10.000     0.000 932861.972 0.000 10.000 10.000
    #x          0.667     0.777      0.858 0.391 -0.856  2.189
    #x Var      1.810 56268.076                               
    #=========================================================

    #"""
    #>>> result.random_effects
    #{np.str_('a'): x    0.333322
    #dtype: float64, np.str_('b'): x   -2.666674
    #dtype: float64, np.str_('c'): x    2.333316
    #dtype: float64}

    # add global slope (~2/3) and group slope to get each group's slope
    a_slope = result.fe_params['x'] + result.random_effects[np.str_('a')]['x']
    b_slope = result.fe_params['x'] + result.random_effects[np.str_('b')]['x']
    c_slope = result.fe_params['x'] + result.random_effects[np.str_('c')]['x']
    assert np.allclose( 1, a_slope, atol=1e-6, rtol=1e-6)
    assert np.allclose(-2, b_slope, atol=1e-4, rtol=1e-4)
    assert np.allclose( 3, c_slope, atol=1e-4, rtol=1e-4)


    # model configuration, which includes random slopes and intercepts, but no
    #   global intercept
    ##################################################

    md = smf.mixedlm('y ~ x + 0', df, groups=df['group'], re_formula='~x')
    result = md.fit()

    # like the prior two models with random slopes, this model performs 
    #   perfectly well with ~0 residuals/'Scale'
    # by removing the global intercept, the random intercepts are all correctly
    #   estimated at ~10

    #>>> result.summary()
    #<class 'statsmodels.iolib.summary2.Summary'>
    #"""
    #          Mixed Linear Model Regression Results
    #=========================================================
    #Model:               MixedLM  Dependent Variable:  y     
    #No. Observations:    9        Method:              REML  
    #No. Groups:          3        Scale:               0.0000
    #Min. group size:     3        Log-Likelihood:      2.9764
    #Max. group size:     3        Converged:           Yes   
    #Mean group size:     3.0                                 
    #---------------------------------------------------------
    #              Coef.   Std.Err.    z   P>|z| [0.025 0.975]
    #---------------------------------------------------------
    #x              0.882      2.962 0.298 0.766 -4.923  6.686
    #Group Var     59.665 284471.241                          
    #Group x x Cov -1.284  89276.137                          
    #x Var          4.290  24182.346                          
    #=========================================================

    #"""
    #>>> result.random_effects
    #{np.str_('a'): Group    10.000007
    #x         0.118093
    #dtype: float64, np.str_('b'): Group    10.000009
    #x        -2.881907
    #dtype: float64, np.str_('c'): Group    10.000021
    #x         2.118095
    #dtype: float64}

    # add global slope (0.88) and group slope to get each group's slope
    a_slope = result.fe_params['x'] + result.random_effects[np.str_('a')]['x']
    b_slope = result.fe_params['x'] + result.random_effects[np.str_('b')]['x']
    c_slope = result.fe_params['x'] + result.random_effects[np.str_('c')]['x']
    assert np.allclose( 1, a_slope, atol=1e-6, rtol=1e-6)
    assert np.allclose(-2, b_slope, atol=1e-4, rtol=1e-4)
    assert np.allclose( 3, c_slope, atol=1e-4, rtol=1e-4)


    # model configuration, which includes random slopes but no intercepts, 
    #   neither global nor random
    ##################################################

    md = smf.mixedlm('y ~ x + 0', df, groups=df['group'], re_formula='~x-1')
    result = md.fit()

    # removing all the intercepts (global and random) obliterates the prior
    #   models' perfect performance (below 'Scale' = 57.94 vs. ~0 previously), 
    #   as expected

    #>>> result.summary()
    #<class 'statsmodels.iolib.summary2.Summary'>
    #"""
    #        Mixed Linear Model Regression Results
    #======================================================
    #Model:            MixedLM Dependent Variable: y       
    #No. Observations: 9       Method:             REML    
    #No. Groups:       3       Scale:              57.9368 
    #Min. group size:  3       Log-Likelihood:     -30.2705
    #Max. group size:  3       Converged:          Yes     
    #Mean group size:  3.0                                 
    #-------------------------------------------------------
    #         Coef.   Std.Err.    z    P>|z|  [0.025  0.975]
    #-------------------------------------------------------
    #x         4.961     2.776  1.787  0.074  -0.481  10.402
    #x Var    13.353     2.879                              
    #======================================================

    #"""
    #>>> result.random_effects
    #{np.str_('a'): x    1.091881
    #dtype: float64, np.str_('b'): x   -3.254552
    #dtype: float64, np.str_('c'): x    2.162672
    #dtype: float64}


    #dir(result)
    #result.resid
    #result.params
    #result.k_fe
    #result.k_re
    #result.k_re2
    #result.fittedvalues
    #result.random_effects
    #result.fe_params
    #result.df_resid
    #result.df_modelwc
    #result.scale
    #result.llf


def toy_model_04():
    """
    These toy models represent a simple regression ('x' as a predictor; 'y' as 
        an outcome) with a nested structure ('group') as a grouping variable
    The groups have a common intercept but different slopes
    However, with the ICCs, these data are being analyzed as in an ANOVA model

    From 'pingouin' documentation:
    https://pingouin-stats.org/build/html/generated/pingouin.intraclass_corr.html

        Shrout and Fleiss (1979) [2] describe six cases of reliability of 
            ratings done by \(k\) raters on \(n\) targets. Pingouin returns all 
            six cases with corresponding F and p-values, as well as 95% 
            confidence intervals.

        From the documentation of the ICC function in the psych R package:

        ICC1: Each target is rated by a different rater and the raters are 
            selected at random. This is a one-way ANOVA fixed effects model.

        ICC2: A random sample of \(k\) raters rate each target. The measure is 
            one of absolute agreement in the ratings. ICC1 is sensitive to 
            differences in means between raters and is a measure of absolute 
            agreement.

        ICC3: A fixed set of \(k\) raters rate each target. There is no 
            generalization to a larger population of raters. ICC2 and ICC3 
            remove mean differences between raters, but are sensitive to 
            interactions. The difference between ICC2 and ICC3 is whether raters 
            are seen as fixed or random effects.

        Then, for each of these cases, the reliability can either be estimated 
            for a single rating or for the average of \(k\) ratings. The 1 
            rating case is equivalent to the average intercorrelation, while 
            the \(k\) rating case is equivalent to the Spearman Brown adjusted 
            reliability. ICC1k, ICC2k, ICC3K reflect the means of \(k\) raters.
    """

    # After looking at the R package 'irr' and the Wikipedia article,
    #
    #   https://en.wikipedia.org/wiki/Intraclass_correlation
    #
    #   I realized that I was entering the data incorrectly for this 'pingouin'
    #   package.  The worked example on the Wikipedia page mostly clarifies 
    #   which 'pingouin' ICC corresponds to which Shrout and Fleiss (1979) ICC 
    #   names

    df = pl.DataFrame({
        'group': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'],
        'x':     [  1,   2,   3,   4,   5,   1,   2,   3,   4,   5],
        'y':     [  1,   2,   3,   4,   5, 3+1, 3+2, 3+3, 3+4, 3+5]})

    group_var = partition_group_variance(df, 'group', 'y')
    icc_df = pg.intraclass_corr(
        data=df.to_pandas(), targets='x', raters='group', ratings='y')

    # Wikipedia answer for ICC(1)
    assert np.allclose(icc_df.iloc[0, 2], 0.053, atol=1e-3, rtol=1e-3)
    # Wikipedia answer for ICC(2,1),agreement and ICC(3,1),agreement
    assert np.allclose(icc_df.iloc[1, 2], 0.357, atol=1e-3, rtol=1e-3)
    # Wikipedia answer for ICC(2,1),consistency and ICC(3,1),consistency
    assert np.allclose(icc_df.iloc[2, 2], 1.00 , atol=1e-3, rtol=1e-3)

    # within-group variance proportion tracks with agreement, in this case
    assert np.allclose(group_var.wthn_var_prop, 0.357, atol=1e-3, rtol=1e-3)

    #>>> icc_df.iloc[:, :6]
    #    Type              Description       ICC         F  df1  df2
    #0   ICC1   Single raters absolute  0.052632  1.111111    4    5
    #1   ICC2     Single random raters  0.357143       inf    4    4
    #2   ICC3      Single fixed raters  1.000000       inf    4    4
    #3  ICC1k  Average raters absolute  0.100000  1.111111    4    5
    #4  ICC2k    Average random raters  0.526316       inf    4    4
    #5  ICC3k     Average fixed raters  1.000000       inf    4    4


    df = pl.DataFrame({
        'group': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'],
        'x':     [  1,   2,   3,   4,   5,   1,   2,   3,   4,   5],
        'y':     [  1,   2,   3,   4,   5, 2*1, 2*2, 2*3, 2*4, 2*5]})

    group_var = partition_group_variance(df, 'group', 'y')
    icc_df = pg.intraclass_corr(
        data=df.to_pandas(), targets='x', raters='group', ratings='y')

    # Wikipedia answer for ICC(1)
    assert np.allclose(icc_df.iloc[0, 2], 0.343, atol=1e-3, rtol=1e-3)
    # Wikipedia answer for ICC(2,1),agreement and ICC(3,1),agreement
    assert np.allclose(icc_df.iloc[1, 2], 0.476, atol=1e-3, rtol=1e-3)
    # Wikipedia answer for ICC(2,1),consistency and ICC(3,1),consistency
    assert np.allclose(icc_df.iloc[2, 2], 0.800, atol=1e-3, rtol=1e-3)

    # within-group variance proportion doesn't track agreement, in this case
    assert np.allclose(group_var.wthn_var_prop, 0.581, atol=1e-3, rtol=1e-3)

    #>>> icc_df.iloc[:, :6]
    #    Type              Description       ICC         F  df1  df2
    #0   ICC1   Single raters absolute  0.343284  2.045455    4    5
    #1   ICC2     Single random raters  0.476190  9.000000    4    4
    #2   ICC3      Single fixed raters  0.800000  9.000000    4    4
    #3  ICC1k  Average raters absolute  0.511111  2.045455    4    5
    #4  ICC2k    Average random raters  0.645161  9.000000    4    4
    #5  ICC3k     Average fixed raters  0.888889  9.000000    4    4


def toy_model_05():
    """
    These toy models represent a simple regression ('x' as a predictor; 'y' as 
        an outcome) with a nested structure ('group') as a grouping variable
    The groups have a common intercept but different slopes
    However, with the ICCs, these data are being analyzed as in an ANOVA model
    """

    df = pl.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
        'x':     [  0,   1,   2,   0,   1,   2,   0,   1,   2],
        'y':     [ 10,  11,  12,  10,  11,  12,  10,  11,  12]})

    group_variance = partition_group_variance(df, 'group', 'y')
    icc_df = pg.intraclass_corr(
        data=df.to_pandas(), targets='x', raters='group', ratings='y')

    # when all 'ratings'/'y' are the same, traditional intracluster cluster 
    #   coefficient of between-group variance divided by total variance = 0
    #   and all ICCs = 1
    assert group_variance.btwn_var_prop == 0
    assert np.allclose(icc_df['ICC'], 1, atol=1e-6, rtol=1e-6)


def toy_model_06():
    """
    These toy models represent a simple regression ('x' as a predictor; 'y' as 
        an outcome) with a nested structure ('group') as a grouping variable
    The groups have a common intercept but different slopes
    However, with the ICCs, these data are being analyzed as in an ANOVA model
    """

    df = pl.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
        'x':     [  0,   1,   2,   0,   1,   2,   0,   1,   2],
        'y':     [  2,   3,   4,   4,   5,   6,   6,   7,   8]})

    group_variance = partition_group_variance(df, 'group', 'y')
    icc_df = pg.intraclass_corr(
        data=df.to_pandas(), targets='x', raters='group', ratings='y')

    assert np.allclose(group_variance.btwn_var_prop, 0.80, atol=1e-2, rtol=1e-2)
    assert np.allclose(
        group_variance.wthn_var_prop, icc_df.iloc[1, 2], atol=1e-2, rtol=1e-2)

    #>>> icc_df.iloc[:, :6]
    #    Type              Description       ICC     F  df1  df2
    #0   ICC1   Single raters absolute -0.090909  0.75    2    6
    #1   ICC2     Single random raters  0.200000   inf    2    4
    #2   ICC3      Single fixed raters  1.000000   inf    2    4
    #3  ICC1k  Average raters absolute -0.333333  0.75    2    6
    #4  ICC2k    Average random raters  0.428571   inf    2    4
    #5  ICC3k     Average fixed raters  1.000000   inf    2    4


    # compared to the data set above, the data set below maintains the same
    #   means and "slopes" but spreads the within-group variability
    ##################################################

    df = pl.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
        'x':     [  0,   1,   2,   0,   1,   2,   0,   1,   2],
        'y':     [  1,   3,   5,   3,   5,   7,   5,   7,   9]})

    group_variance = partition_group_variance(df, 'group', 'y')
    icc_df = pg.intraclass_corr(
        data=df.to_pandas(), targets='x', raters='group', ratings='y')

    # when the within-group variability increases in 'ratings'/'y':
    #   intracluster correlation coefficient decreases
    #   1 - intracluster correlation coefficient seems to equal ICC2
    #       I think I mis-identified the intracluster correlation coefficient:
    #           since it's *intra*cluster, larger within-group variability would
    #           increase it, so it would be within-group variance / total 
    #           variance, not between-group variance / total variance 
    #   ICC1s increase
    #   ICC2s increase
    #   ICC3s stay at 1
    #   absolute ICCs are lowest in their respective groups of 3 ICCs
    assert np.allclose(group_variance.btwn_var_prop, 0.50, atol=1e-2, rtol=1e-2)
    assert np.allclose(
        group_variance.wthn_var_prop, icc_df.iloc[1, 2], atol=1e-2, rtol=1e-2)

    #>>> icc_df.iloc[:, :6]
    #    Type              Description       ICC    F  df1  df2
    #0   ICC1   Single raters absolute  0.400000  3.0    2    6
    #1   ICC2     Single random raters  0.500000  inf    2    4
    #2   ICC3      Single fixed raters  1.000000  inf    2    4
    #3  ICC1k  Average raters absolute  0.666667  3.0    2    6
    #4  ICC2k    Average random raters  0.750000  inf    2    4
    #5  ICC3k     Average fixed raters  1.000000  inf    2    4


    # unlike above, the "within-group intracluster correlation coefficient"
    #   does not equal the ICC2
    ##################################################

    df = pl.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
        'x':     [  0,   1,   2,   0,   1,   2,   0,   1,   2],
        'y':     [ 10,  11,  12,  12,  11,  10,  10,  11,  12]})

    group_variance = partition_group_variance(df, 'group', 'y')
    icc_df = pg.intraclass_corr(
        data=df.to_pandas(), targets='x', raters='group', ratings='y')

    assert np.allclose(group_variance.wthn_var_prop, 1.00, atol=1e-2, rtol=1e-2)
    assert not np.allclose(
        group_variance.wthn_var_prop, icc_df.iloc[1, 2], atol=1e-2, rtol=1e-2)


def main():

    toy_model_01()
    toy_model_02()
    toy_model_03()
    toy_model_04()
    toy_model_05()
    toy_model_06()
    print('Toy models have completed their runs.')

    # correspondences
    #   raters      targets     ratings
    #   group       x           y
    #   sellers     buyers      prices
    #   providers   patients    costs
    #
    # for our problem:
    #   raters:  fixed
    #   targets:  random


if __name__ == '__main__':
    main()
