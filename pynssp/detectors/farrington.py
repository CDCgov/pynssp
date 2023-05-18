from datetime import datetime
import numpy as np
import pandas as pd
from statsmodels.api import families, GLM, add_constant
from statsmodels.genmod.families.links import log
from scipy.stats import norm, nbinom, poisson
from pandas.api.types import is_datetime64_any_dtype as is_datetime


def seasonal_groups(B=4, g=27, w=3, p=10, base_length=None, base_weeks=None):
    """Return 10-level seasonal factor series

    :param B: Number of years to include in baseline (default is 4)
    :param g: Number of guardband weeks to separate the test week from the baseline (default is 27)
    :param w: Half the number of weeks included in reference window, before and after each reference date (default is 3)
    :param p: Number of seasonal periods for each year in baseline (default is 10)
    :param base_length: Total number of weeks included in baseline
    :param base_weeks: Indices of baseline weeks

    :return: A pandas Series
    """
    h = [1] + [base_weeks[i+1] - base_weeks[i] for i in range(len(base_weeks) - 1)]
    csum_h = np.cumsum(h) - 1
    fct_levels = np.zeros(base_length)

    for i in range(B):
        fct_levels[csum_h[i]:(csum_h[i] + 2 * w + 1)] = p

        delta_weeks = h[i + 1] - (2 * w + 1)

        quotient = delta_weeks // (p - 1)
        remainder = delta_weeks % (p - 1)

        idx_extra = np.arange(remainder)

        fct_lengths = np.full(p - 1, quotient)
        fct_lengths[idx_extra] += 1
        fct_lengths = np.concatenate(([0], fct_lengths))

        cum_lengths = np.cumsum(fct_lengths)

        for j in range(1, p):
            fct_levels[csum_h[i] + 2 * w + 1 + cum_lengths[j-1]:
                    csum_h[i] + 2 * w + 1 + cum_lengths[j]] = j

    # Trim extra components outside of baseline
    # fct_levels = pd.Series(fct_levels).astype('category')

    return fct_levels.astype(int)


def farrington_modified(df, t='date', y='count', B=4, g=27, w=3, p=10):
    """Modified Farrington Algorithm

    :param df: A pandas dataframe
    :param t: A column containing date values (Default value = "date")
    :param y: A column containing time series counts (Default value = "count")
    :param B: Number of years to include in baseline (default is 4)
    :param g: Number of guardband weeks to separate the test date 
        from the baseline (default is 27)
    :param w: Half the number of weeks included in reference window, before and 
        after each reference date (default is 3)
    :param p: Number of seasonal periods for each year in baseline
    :return: A pandas dataframe
    """
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df[t])
    y_obs = df[y]
    N = len(df)

    # Minimum number of observations needed
    min_obs = 52 * B + w + 1

    # Initialize result vectors
    predicted = np.repeat(np.nan, N)
    time_coefficient = np.repeat(np.nan, N)
    include_time_term = [None] * N #np.repeat(np.nan, N)
    upper = np.repeat(np.nan, N)
    alert_score = np.repeat(np.nan, N)
    alert = [None] * N

    for i in range(min_obs, N):
        current_date = dates[i]
        ref_dates = pd.date_range(current_date, periods=B+1, freq='-1Y')
        ref_dates = [date.replace(month=current_date.month, day=current_date.day) for date in ref_dates]
        wday_gaps = [date.isoweekday() % 7 - current_date.isoweekday() % 7 for date in ref_dates] 
        ref_dates_shifted = np.array([date - pd.to_timedelta(wkd, unit='d') for date, wkd in zip(ref_dates, wday_gaps)])
        floor_ceiling_dates = np.where(ref_dates_shifted > ref_dates, ref_dates_shifted - pd.Timedelta(7, 'd'), ref_dates_shifted + pd.Timedelta(7, 'd'))
        center_dates = np.sort(np.where(abs(ref_dates - floor_ceiling_dates) < abs(ref_dates - ref_dates_shifted), floor_ceiling_dates, ref_dates_shifted))
        base_start = np.sort((center_dates - pd.Timedelta(7*w, 'd'))[:B])
        base_end = np.concatenate((np.sort(center_dates - pd.to_timedelta(7*B, unit='D'))[1:B], np.array([max(center_dates) - pd.to_timedelta(7*g, unit='D')])))
        base_dates = pd.date_range(start=min(base_start), end=max(base_end), freq='1W')
        base_length = len(base_dates)
        base_weeks = np.where(dates.isin(center_dates))[0]
        fct_levels = seasonal_groups(B, g, w, p, base_length, base_weeks)
        
        idx = np.where(np.in1d(dates, base_dates))[0]
        min_date = np.min(dates[idx])
        base_dates = np.asarray((dates[idx] - min(dates[idx])) / np.timedelta64(1, 'W'), dtype=np.float64)
        base_counts = y_obs[idx]
        mod_data = pd.DataFrame({"base_counts": base_counts, "base_dates": base_dates, "fct_levels": fct_levels})
        mod_data = pd.get_dummies(mod_data, columns=['fct_levels'], drop_first=True)
        mod_data.columns = [str(col).replace(".0","") for col in mod_data]
        mod_formula = "base_counts ~ 1 + base_dates + " + " + ".join([f"fct_levels_{i+1}" for i in range(1, p)])
        mod = GLM.from_formula(mod_formula, mod_data, family=families.Poisson(link=log())).fit()
        if not mod.converged:
            mod_formula = "base_counts ~ 1 + " + " + ".join([f"fct_levels_{i+1}" for i in range(1, p)])
            mod = GLM.from_formula(mod_formula, mod_data, family=families.Poisson(link=log()))
            include_time = False
        else:
            include_time = True
        if not mod.converged:
            continue
        # mod_formula = mod.model.formula
        if include_time:
            time_coeff = mod.params['base_dates']
        else:
            time_coeff = np.nan
            # time_p_val = np.nan
        y_observed = mod._endog
        y_fit = mod.fittedvalues
        phi = np.maximum(mod.pearson_chi2 / mod.df_resid, 1)
        diag = mod.get_influence().hat_matrix_diag
        ambscombe_resid = ((3 / 2) * (y_observed**(2 / 3) * (y_fit**(-1 / 6)) - np.sqrt(y_fit))) / (np.sqrt(phi * (1 - diag)))
        scaled = np.where(ambscombe_resid > 2.58, 1 / (ambscombe_resid**2), 1)
        gamma = len(ambscombe_resid) / np.sum(scaled)
        omega = np.where(ambscombe_resid > 2.58, gamma / (ambscombe_resid**2), gamma)
        mod_weighted = GLM.from_formula(
            formula=mod_formula, 
            data=mod_data, 
            family=families.Poisson(link=log()), 
            freq_weights=omega
        ).fit()
        phi_weighted = max(mod_weighted.pearson_chi2 / mod_weighted.df_resid, 1)
        # mod_weighted.week_time = base_dates
        # mod_weighted.data_count = base_counts
        # mod_weighted.phi = phi_weighted
        # mod_weighted.weights = omega
        if include_time:
            time_pval_weighted = mod_weighted.pvalues['base_dates']
        else:
            time_pval_weighted = np.nan
        pred_week_time = ((current_date - min_date) / 7).days
        dummy_vars = {'fct_levels_{}'.format(i): int(i == p) for i in range(1, p+1)}
        pred_results = mod_weighted.get_prediction(
            pd.DataFrame({'base_dates': [pred_week_time], 'dispersion': [phi_weighted], **dummy_vars})
        ).summary_frame(alpha=0.05)

        pred, _, _, upper_ci = pred_results.values.T
        # Check 2 conditions
        # > 1: p-value for time term significant at 0.05 level
        time_significant = (pred <= np.nanmax(base_counts) and time_pval_weighted < 0.05)
        # > 2: Prediction less than or equal to maximum observation in baseline
        pred_ok = pred <= np.nanmax(base_counts)
        trend = include_time and time_significant and pred_ok

        if not trend:
            mod_formula = "base_counts ~ 1 + " + " + ".join([f"fct_levels_{i+1}" for i in range(1, p)])
            mod = GLM.from_formula(mod_formula, mod_data, family=families.Poisson(link=log())).fit()
            if not mod.converged:
                continue
            else:
                mod_formula = mod.model.formula
                y_observed = mod._endog
                y_fit = mod.fittedvalues
                phi = np.maximum(mod.pearson_chi2 / mod.df_resid, 1)
                diag = mod.get_influence().hat_matrix_diag
                ambscombe_resid = ((3 / 2) * (y_observed**(2 / 3) * (y_fit**(-1 / 6)) - np.sqrt(y_fit))) / (np.sqrt(phi * (1 - diag)))
                scaled = np.where(ambscombe_resid > 2.58, 1 / (ambscombe_resid**2), 1)
                gamma = len(ambscombe_resid) / sum(scaled)
                omega = np.where(ambscombe_resid > 2.58, gamma / (ambscombe_resid**2), gamma)
                mod_weighted = GLM.from_formula(
                    mod_formula, mod_data, family=families.Poisson(link=log()), freq_weights=omega
                ).fit()
                phi_weighted = max(mod_weighted.pearson_chi2 / mod_weighted.df_resid, 1)
                # mod_weighted.phi = phi_weighted
                # mod_weighted.weights = omega
                # mod_weighted.week_time = base_dates
                # mod_weighted.data_count = base_counts
                pred_results = mod_weighted.get_prediction(
                    pd.DataFrame({
                        'base_dates': [pred_week_time], 'population': [1], 
                        'dispersion': [phi_weighted], **dummy_vars
                    })
                ).summary_frame(alpha=0.05)

                pred, _, _, upper_ci = pred_results.values.T

                include_time_term[i] = False
        else:
            # pred = mod_weighted.predict(
            #     add_constant(
            #         pd.DataFrame(
            #             {'base_dates': pred_week_time, 'population': [1], 
            #              'dispersion': phi_weighted, 'fct_levels': pd.factorize(p)[0]}
            #         )
            #     ), 
            #     return_se=True, linear=True
            # )
            include_time_term[i] = True
        predicted[i] = pred
        time_coefficient[i] = time_coeff
        # include_time_term[i] = True
        # Temporary result vectors
        eta = predicted[i]
        mu_q = np.exp(eta)
        # dispersion = phi_weighted
        upper[i] = np.nan if mu_q == np.inf else (nbinom.ppf(0.95, mu_q / (phi_weighted - 1), 1 / phi_weighted) if phi_weighted > 1 else poisson.ppf(0.95, mu_q))
        alert_score[i] = (y_obs[i] - predicted[i]) / (upper[i] - predicted[i]) if not np.isnan(upper[i]) else np.nan
        recent_counts = np.sum(y_obs[(i - 4):i+1]) 
        alert[i] = 'red' if alert_score[i] > 1 and recent_counts > 5 else 'blue'
        upper[i] = upper[i] if recent_counts > 5 else np.nan
        predicted[i] = np.exp(predicted[i])
    
    return pd.concat([df, pd.DataFrame(
        {
            'predicted': predicted,
            'time_coefficient': time_coefficient, 
            'include_time_term': include_time_term, 
            'upper': upper, 
            'alert_score': alert_score, 
            'alert': alert
         }
       ).assign(alert=lambda x: x['alert'].fillna('grey'))
    ], axis=1)



def farrington_original(df, t='date', y='count', B=4, w=3):
    """Original Farrington Algorithm

    :param df: A pandas dataframe
    :param t: A column containing date values (Default value = "date")
    :param y: A column containing time series counts (Default value = "count")
    :param B: Number of years to include in baseline (default is 4)
    :param w: Half the number of weeks included in reference window,
        before and after each reference date (default is 3)
    :return: A pandas dataframe
    """
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df[t])
    y_obs = df[y]
    N = len(df)
    
    # Minimum number of observations needed
    min_obs = 52 * B + w + 1
    
    # Initialize result vectors
    predicted = np.repeat(np.nan, N)
    time_coefficient = np.repeat(np.nan, N)
    include_time_term = [None] * N #np.repeat(np.nan, N)
    upper = np.repeat(np.nan, N)
    alert_score = np.repeat(np.nan, N)
    alert = [None] * N
    
    for i in range(min_obs, N):
        current_date = dates[i]
        ref_dates = pd.date_range(current_date, periods=B+1, freq='-1Y')[1:]
        ref_dates = [date.replace(month=current_date.month, day=current_date.day) for date in ref_dates]
        # current_week = (current_date - datetime(current_date.year,1,1)).days // 7
        wday_gaps = [date.isoweekday() % 7 - current_date.isoweekday() % 7 for date in ref_dates] 
        ref_dates_shifted = np.array([date - pd.to_timedelta(wkd, unit='d') for date, wkd in zip(ref_dates, wday_gaps)])
        floor_ceiling_dates = np.where(ref_dates_shifted > ref_dates, ref_dates_shifted - pd.Timedelta(7, 'd'), ref_dates_shifted + pd.Timedelta(7, 'd'))
        center_dates = np.sort(np.where(abs(ref_dates - floor_ceiling_dates) < abs(ref_dates - ref_dates_shifted), floor_ceiling_dates, ref_dates_shifted))
        base_start = np.sort((center_dates - pd.Timedelta(7*w, 'd'))[:B])
        idx_start = np.where(dates.isin(base_start))[0]
        idx = np.add(np.repeat(idx_start, 7), np.tile(np.arange(7), len(idx_start)))
        min_date = dates.iloc[idx].min()
        base_dates = ((dates.iloc[idx] - min_date).dt.days / 7).astype(int).values
        base_counts = y_obs.iloc[idx].values
        mod = GLM(base_counts, add_constant(base_dates), family=families.Poisson(link=log())).fit()
        if not mod.converged:
            mod = GLM(base_counts, np.ones_like(base_dates), family=families.Poisson(link=log())).fit()
            include_time = False
            mod_formula = "base_counts ~ 1"
        else:
            mod_formula = "base_counts ~ 1 + base_dates"
            include_time = True
        # if not mod.converged:
        #     continue
        # mod_formula = mod.model.formula
        if include_time:
            time_coeff = mod.params[1]
        else:
            time_coeff = np.nan
            # time_p_val = np.nan
        y_observed = mod.model.endog
        y_fit = mod.fittedvalues
        phi = np.maximum(mod.pearson_chi2 / mod.df_resid, 1)
        diag = mod.get_influence().hat_matrix_diag

        ambscombe_resid = ((3 / 2) * (np.power(y_observed, 2 / 3) * np.power(y_fit, -1 / 6) - np.sqrt(y_fit))) / (np.sqrt(phi * (1 - diag)))
        
        scaled = np.where(ambscombe_resid > 1, 1 / (ambscombe_resid**2), 1)
        gamma = len(ambscombe_resid) / np.sum(scaled)
        omega = np.where(ambscombe_resid > 1, gamma / (ambscombe_resid**2), gamma)
        base_df = pd.DataFrame({"base_counts": base_counts, "base_dates": base_dates, "omega": omega})
        mod_weighted = GLM.from_formula(formula=mod_formula, data=base_df, family=families.Poisson(link=log()), freq_weights=omega).fit()
        # phi_weighted = max(mod_weighted.scale, 1)
        # mod_weighted.phi = phi_weighted
        # mod_weighted.weights = omega
        # mod_weighted.week_time = base_dates
        # mod_weighted.data_count = base_counts
        if include_time:
            time_pval_weighted = mod_weighted.pvalues['base_dates']
        pred_week_time = int((current_date - min_date).days / 7)
        # pred = mod_weighted.predict(pd.DataFrame({'base_dates': pred_week_time, 'dispersion': phi_weighted}), 
        #                     type="response", 
        #                     se=True)
        pred_results = mod_weighted.get_prediction(
            pd.DataFrame({'base_dates': [pred_week_time]})
        ).summary_frame(alpha=0.05)
        pred, _, _, upper_ci = pred_results.values.T
        time_significant = (pred <= np.nanmax(base_counts) and time_pval_weighted < 0.05)
        # Check 2 conditions
        # > 1: p-value for time term significant at 0.05 level
        time_significant = time_pval_weighted < 0.05
        # > 2: Prediction less than or equal to maximum observation in baseline
        pred_ok = pred <= np.nanmax(base_counts)
        trend = include_time & time_significant & pred_ok

        if not trend:
            mod = GLM.from_formula(
                "base_counts ~ 1",
                data=base_df,
                family=families.Poisson(link=log())
            ).fit()
    
            if not mod.converged:
                continue
            else:
                mod_formula = mod.model.formula
        
                y_observed = mod._endog
                y_fit = mod.mu
                phi = max(mod.pearson_chi2 / mod.df_resid, 1)
                diag = mod.get_influence().hat_matrix_diag
        
                ambscombe_resid = ((3 / 2) * (np.power(y_observed, 2 / 3) * np.power(y_fit, -1 / 6) - 
                                              np.sqrt(y_fit))) / (np.sqrt(phi * (1 - diag)))
        
                scaled = np.where(ambscombe_resid > 1, 1 / np.power(ambscombe_resid, 2), 1)
                gamma = len(ambscombe_resid) / np.sum(scaled)
                omega = np.where(ambscombe_resid > 1, gamma / np.power(ambscombe_resid, 2), gamma)
        
                mod_weighted = GLM.from_formula(
                    mod_formula, family=families.Poisson(link=log()), 
                    freq_weights=omega,
                    data=pd.DataFrame({'base_dates': base_dates, 'base_counts': base_counts, 'weights': omega})
                ).fit()
                # mod_weighted_res = mod_weighted.fit(method="bfgs", maxiter=100, disp=0)
                phi_weighted = max(mod_weighted.pearson_chi2 / mod_weighted.df_resid, 1)
        
                pred_results = mod_weighted.get_prediction(
                    pd.DataFrame({'base_dates': [pred_week_time], 'population': [1], 'dispersion': [phi_weighted]})
                ).summary_frame(alpha=0.05)

                pred, _, _, upper_ci = pred_results.values.T
        
                include_time_term[i] = False
        else:
            include_time_term[i] = True
        
        predicted[i] = pred 
        time_coefficient[i] = time_coeff
        include_time_term[i] = include_time
        
        upper[i] = upper_ci

        alert_score[i] = np.where(~np.isnan(upper[i]), (y_obs[i] - predicted[i]) / (upper[i] - predicted[i]), np.nan)

        recent_counts = np.sum(y_obs[(i - 4):i+1])
        alert[i] = np.where(alert_score[i] > 1 and recent_counts > 5, "red", "blue").item()
  
    return pd.concat([
        df, 
        pd.DataFrame({
            'predicted': predicted, 'time_coefficient': time_coefficient, 
            'include_time_term': include_time_term, 'upper': upper, 
            'alert_score': alert_score, 'alert': alert
        })
    ], axis=1)


def alert_farrington(df, t='date', y='count', B=4, g=27, w=3, p=10, method='original'):
    """Farrington Temporal Detector

        The Farrington algorithm is intended for weekly time series of counts 
        spanning multiple years. Original Farrington Algorithm: Quasi-Poisson 
        generalized linear regression models are fit to baseline counts associated 
        with reference dates in the B previous years, including w weeks before 
        and after each reference date. The algorithm checks for convergence with a 
        time term and refits a model with only an intercept term in the scenario 
        the model does not converge. The inclusion of high baseline counts associated 
        with past outbreaks or public health events is known to result in alerting 
        thresholds that are too high and a reduction in sensitivity. 
        An empirically derived weighting function is used to calculate weights 
        from Anscombe residuals that assign low weight to baseline observations 
        with large residuals. A 2/3rds transformation is applied to account for 
        skewness common to time series with lower counts, after which expected 
        value and variance estimates are used to derive upper and lower bounds for 
        the prediction interval. The alert score is defined as the current observation 
        minus the forecast value divided by the upper prediction interval bound minus 
        the forecast value. If this score exceeds 1, an alert (red value) is raised 
        given that the number of counts in the last 4 days is above 5. 
        This algorithm requires that the number of years included in the baseline is 3 
        or higher. Blue values are returned if an alert does not occur. 
        Grey values represent instances where anomaly detection did not apply 
        (i.e., observations for which baseline data were unavailable). 
        
        Modified Farrington Algorithm: In 2012, Angela Noufaily developed a modified 
        implementation of the original Farrington algorithm that improved performance 
        by including more historical data in the baseline. 
        The modified algorithm includes all weeks from the beginning of the first 
        reference window to the last week proceeding a 27-week guardband period used 
        to separate the test week from the baseline. 
        A 10-level factor is used to account for seasonality throughout the baseline. 
        Additionally, the modified algorithm assumes a negative binomial distribution 
        on the weekly time series counts, where thresholds are computed as quantiles 
        of the negative binomial distribution with plug-in estimates for mu and phi.

        :param df: A pandas dataframe
        :param t: A column containing date values (Default value = "date")
        :param y: A column containing time series counts (Default value = "count")
        :param B: Number of years to include in baseline (default is 4)
        :param g: Number of guardband weeks to separate the test date 
            from the baseline (default is 27)
        :param w: Half the number of weeks included in reference window, 
            before and after each reference date (default is 3)
        :param p: Number of seasonal periods for each year in baseline
        :param method: A string of either "original" (default) or "modified" 
            to specify the version of the Farrington algorithm (original vs modified).

        :return: A dataframe
    """
    # Ensure that df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Argument 'df' must be a DataFrame.")

    # Check baseline length argument
    if B < 4:
        raise ValueError("Baseline length argument 'B' must be greater than or equal to 4. Farrington algorithm requires a baseline of four or more years.")

    # Check guardband length argument
    if g < 0:
        raise ValueError("Guardband length argument 'g' cannot be negative.")

    # Check half-week baseline argument
    if w < 0:
        raise ValueError("Half-week baseline argument 'w' cannot be negative.")

    # Check seasonal periods baseline argument
    if p < 2:
        raise ValueError("Seasonal periods baseline argument 'p' cannot be less than 2.")
    
    # Check for grouping variables
    grouped_df = isinstance(df, pd.core.groupby.DataFrameGroupBy)

    if not grouped_df:
        df_size = df.size
    else:
        df_size = df.size()[0]

    # Check for sufficient baseline data
    if df_size < 52 * B + w + 2:
        raise ValueError("Not enough historical data to form baseline.")

    
    if method not in ["original", "modified"]:
        raise ValueError("Argument 'method' must be 'original' or 'modified'.")

    if grouped_df:

        if method == 'modified':
            alert_tbl = df.apply(lambda x: farrington_modified(x, t, y, B, g, w, p))
        elif method == 'original':
            alert_tbl = df.apply(lambda x: farrington_original(x, t, y, B, w))
    else:
        if method == 'modified':
            alert_tbl = farrington_modified(df, t=t, y=y, B=B, g=g, w=w, p=p)
        elif method == 'original':
            alert_tbl = farrington_original(df, t=t, y=y, B=B, g=g, w=w, p=p)
    
    return alert_tbl
