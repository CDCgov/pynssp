import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

def adaptive_regression(df, t, y, B, g):
    """Adaptive Regression

    Adaptive Regression helper function for Adaptive Multiple Regression

    :param df: A pandas data frame containing time series data
    :param t: Name of the column of type Date containing the dates
    :param y: Name of the column containing the response variable data
    :param B: Baseline parameter. The baseline length is the number of days to
            which each liner model is fit.
    :param g: Guardband parameter. The guardband length is the number of days
            separating the baseline from the current date in consideration for alerting.
    :returns: A pandas data frame with p-values and test statistics
    
    """
    df = df.reset_index(drop=True).sort_values(by=t)
    base_tbl = df
    base_tbl["dow"] = pd.to_datetime(df[t]).dt.strftime("%A").str[:3]
    base_tbl["dummy"] = 1
    df = pd.concat([
        df, base_tbl.pivot_table(index=t, columns="dow", values="dummy").\
            fillna(0).\
            reset_index(drop=True)
        ], axis=1)
    
    dates = pd.to_datetime(base_tbl[t]).tolist()
    y_obs = base_tbl[y].tolist()

    N = len(df)

    # Populate algorithm parameters
    min_df = 3
    min_baseline = 11
    max_baseline = B
    df_range = np.arange(1, B - min_df + 1)

    # ucl_alert = np.round(stats.t.ppf(1 - 0.01, df=df_range), 5)
    ucl_warning = np.round(stats.t.ppf(1 - 0.05, df=df_range), 5)

    # Bound standard error of regression
    min_sigma = 1.01 / ucl_warning

    # Initialize result vectors
    test_stat = np.repeat(np.nan, N)
    p_val = np.repeat(np.nan, N)
    expected = np.repeat(np.nan, N)
    sigma = np.repeat(np.nan, N)
    r_sqrd_adj = np.repeat(np.nan, N)

    # Initialize baseline indices
    ndx_baseline = np.arange(1, min_baseline)

    # Adaptive multiple regression loop
    for i in range(min_baseline + g, N):

        # Pad baseline until full baseline is obtained
        if ndx_baseline[-1] < max_baseline:
            ndx_baseline = np.insert(ndx_baseline, 0, 0)

        # Advance baseline for current iteration
        ndx_baseline += 1

        # Indices for baseline and test date
        if ndx_baseline[-1] < max_baseline:
            ndx_time = np.arange(1, len(ndx_baseline) + 1)
            ndx_test = ndx_baseline[-1] + g + 1
        else:
            ndx_time = np.arange(1, B + 1)
            ndx_test = B + g + 1

        # Set number of degrees of freedom
        n_df = len(ndx_baseline) - 8

        # Baseline and current data
        baseline_data = df.iloc[ndx_baseline - 1, :]

        B_length = len(ndx_baseline)

        # Baseline observed values
        baseline_obs = baseline_data[y]

        # Form regression matrix
        X = np.column_stack(
            [
                ndx_time,
                baseline_data[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]]
            ]
        )

        # Fit regression model with linear regression
        X = sm.add_constant(X)
        lr_fit = sm.OLS(baseline_obs, X).fit()

        # Extract model components
        beta = lr_fit.params.tolist()
        mse = lr_fit.mse_resid

        # Compute adjusted R-squared value
        r2_adj = lr_fit.rsquared_adj
        r_sqrd_adj[i] = r2_adj if not np.isnan(r2_adj) else 0

        # Calculate bounded standard error of regression with derived formula for efficiency
        sigma[i] = max(
            np.sqrt(mse) * np.sqrt(((B_length + 7) * (B_length - 4)) / (B_length * (B_length - 7))),
            min_sigma[n_df],
        )

        # Day of week for test date
        dow_test = int(dates[i].strftime("%u"))

        # Calculate forecast on test date
        if dow_test < 7:
            expected[i] = max(0, beta[0] + ndx_test * beta[1] + beta[dow_test + 1])
        else:
            expected[i] = max(0, beta[0] + ndx_test * beta[1])

        # Calculate test statistic
        test_stat[i] = (y_obs[i] - expected[i]) / sigma[i]

        # Calculate p-value
        p_val[i] = 1 - stats.t.cdf(test_stat[i], n_df)

    return df.assign(
        baseline_expected = expected,
        test_statistic = test_stat,
        p_value = p_val,
        sigma = sigma,
        adjusted_r_squared = r_sqrd_adj
    )


def alert_regression(df, t="date", y="count", B=28, g=2):
    """Adaptive Multiple Regression

    The adaptive multiple regression algorithm fits a linear model to a baseline
    of counts or percentages of length B, and forecasts a predicted value g + 1
    days later (guard-band). This value is compared to the current observed value
    and divided by the standard error of prediction in the test-statistic.
    The model includes terms to account for linear trends and day-of-week effects.
    Note that this implementation does NOT account for federal holidays as in the
    Regression 1.2 algorithm in ESSENCE. An alert (red value) is signaled if
    the statistical test (student"s t-test) applied to the test statistic yields
    a p-value less than 0.01. If the p-value is greater than or equal to 0.01
    and strictly less than 0.05, a warning (yellow value) is signaled.
    Blue values are returned if an alert or warning does not occur.
    Grey values represent instances where anomaly detection did not apply
    (i.e., observations for which baseline data were unavailable).

    :param df: A pandas data frame containing time series data
    :param t: The name of the column in df that contains the dates or times of
            observations. Defaults to "date".
    :param y: The name of the column in df that contains the values of the
            time series. Defaults to "count".
    :param B: The length of the baseline period (in days). Must be a multiple
            of 7 and at least 7. Defaults to 28.
    :param g: The length of the guard band (in days). Must be non-negative.
            Defaults to 2.
    :returns: Original pandas data frame with detection results.
    :examples:
    
        >>> from pynssp import alert_regression
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> df = pd.DataFrame({
        ...    "date": pd.date_range("2020-01-01", "2020-12-31"),
        ...    "count": np.random.randint(0, 101, size=366)
        ... })
        >>> 
        >>> df_regression = alert_regression(df)
        >>> df_regression.head()

    """
    
    # Check baseline length argument
    if B < 7:
        raise ValueError(f"Error in alert_regression: baseline length argument B={B} must be greater than or equal to 7")
    
    if B % 7 != 0:
        raise ValueError(f"Error in alert_regression: baseline length argument B={B} must be a multiple of 7")
    
    # Check guardband length argument
    if g < 0:
        raise ValueError(f"Error in alert_regression: guardband length argument g={g} cannot be negative")
    
    # Check for sufficient baseline data
    grouped_df = isinstance(df, pd.core.groupby.DataFrameGroupBy)

    if not grouped_df:
        df_size = df.size
    else:
        df_size = df.size()[0]

    # Check for sufficient baseline data
    if df_size < B + g + 1:
        raise ValueError("Error in alert_regression: not enough historical data")
        
    if grouped_df:

        df_result = df\
            .apply(lambda data: adaptive_regression(data, t=t, y=y, B=B, g=g))
        
        df_result = df_result.reset_index(drop=True)

        df_result["alert"] = np.select([(df_result["p_value"] < 0.01),
                                        (df_result["p_value"] >= 0.01) & (df_result["p_value"] < 0.05),
                                        (df_result["p_value"] >= 0.05)],
                                        ["red", "yellow", "blue"], default="grey")
        df_result = df_result.drop(columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    else:
        unique_dates = df[t].unique()
        if len(unique_dates) != len(df):
            raise ValueError("Error in alert_regression: Number of unique dates does not equal the number of rows. Should your dataframe be grouped?")

        df_result = adaptive_regression(df, t=t, y=y, B=B, g=g)
        df_result = df_result.reset_index()
        df_result["alert"] = np.select([(df_result["p_value"] < 0.01),
                                        (df_result["p_value"] >= 0.01) & (df_result["p_value"] < 0.05),
                                        (df_result["p_value"] >= 0.05)],
                                        ["red", "yellow", "blue"], default="grey")
        df_result = df_result.drop(columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        
    return df_result
