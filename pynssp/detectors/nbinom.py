import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.api import families, formula
from statsmodels.genmod.families import links

def nb_model(df, t, y, baseline_end, include_time):
    """Negative binomial regression model for weekly counts

    Negative binomial model helper function for monitoring weekly
    count time series with seasonality

    :param df: A pandas data frame
    :param t: Name of the column of type Date containing the dates
    :param y: Name of the column of type Numeric containing counts
    :param baseline_end: Object of type Date defining the end of the baseline/training period
    :param include_time: Logical indicating whether or not to include time term in regression model
    :returns: A pandas data frame.
    """
    df = df.reset_index(drop=True).sort_values(by=t)

    df[t] = pd.to_datetime(df[t])
    df[y] = pd.to_numeric(df[y])

    # Check baseline length and for sufficient historical data
    baseline = df if baseline_end is None else df[df[t] <= pd.to_datetime(baseline_end)]
    baseline_n_wks = baseline[t].nunique()
    baseline_n_yrs = baseline_n_wks / 52
    if baseline_n_yrs < 2:
        raise ValueError("Baseline length must be greater than or equal to 2 years")

    baseline_dates = baseline[t].unique()
    if len(pd.date_range(start=min(baseline_dates), end=max(baseline_dates), freq="W")) != baseline_n_wks:
        raise ValueError("Not all weeks in intended baseline date range were found")

    # Check that time series observations are non-negative integer counts
    ts_obs = df[y]
    ts_obs_int = ts_obs.astype(int)
    if not all(ts_obs == ts_obs_int) or not all(ts_obs >= 0):
        raise ValueError("Time series observations must be non-negative integer counts")

    df["obs"] = np.arange(1, len(df)+1)
    df["cos"] = np.cos(2 * np.pi * df["obs"] / 52.18)
    df["sin"] = np.sin(2 * np.pi * df["obs"] / 52.18)
    df["split"] = np.where(df[t] <= pd.to_datetime(baseline_end), "Baseline Period", "Prediction Period")

    baseline_data = df[df["split"] == "Baseline Period"]
    predict_data = df[df["split"] == "Prediction Period"]

    if include_time:
        responses = ["obs", "cos", "sin"] 
        formula_str = y + " ~ obs + cos + sin"
    else:
        responses = ["obs", "sin"] 
        formula_str = y + " ~ cos + sin"
    
    # Calculate the dispersion parameter `theta` using the inverse link function
    alpha = sm.NegativeBinomial(
        endog=baseline_data[y], 
        exog=sm.add_constant(baseline_data[responses])
    ).fit().params[-1]

    theta = 1/alpha

    # Fit the baseline negative binomial model
    baseline_model = formula.glm(
        formula_str, 
        data=baseline_data, 
        family=families.NegativeBinomial(link=links.log(), alpha=alpha)
    ).fit()

    # Predictions
    baseline_fit = baseline_data.copy()
    baseline_preds = baseline_model.get_prediction(baseline_fit)
    baseline_pred_ci = baseline_preds.summary_frame(alpha=0.05)
    baseline_fit["estimate"], _, _, baseline_fit["upper_ci"] = baseline_pred_ci.values.T
    baseline_fit["threshold"] = stats.nbinom.ppf(
        1 - 0.05, 
        n=theta, 
        p=theta/(theta+baseline_fit['upper_ci'])
    )

    predict_fit = predict_data.copy()
    predict_preds = baseline_model.get_prediction(predict_fit)
    predict_pred_ci = predict_preds.summary_frame(alpha=0.05) 
    predict_fit["estimate"], _, _, predict_fit["upper_ci"] = predict_pred_ci.values.T
    predict_fit["threshold"] = stats.nbinom.ppf(
        1 - 0.05, 
        n=theta, 
        p=theta/(theta+predict_fit['upper_ci'])
    )

    result = pd.concat([baseline_fit, predict_fit])
    result.sort_values(by=t, inplace=True)
    result.reset_index(drop=True, inplace=True)
    result["split"] = pd.Categorical(result["split"], categories=["Baseline Period", "Prediction Period"])
    result["alarm"] = np.where(result[y] > result["threshold"], True, False)
    result["time_term"] = include_time
    result.drop(columns=["obs", "cos", "sin", "upper_ci"], inplace=True)

    return result


def alert_nbinom(df, baseline_end, t="date", y="count", include_time=True):
    """Negative binomial detection algorithm for weekly counts
    
    The negative binomial regression algorithm fits a negative binomial regression
    model with a time term and order 1 Fourier terms to a baseline period that
    spans 2 or more years. Inclusion of Fourier terms in the model is intended
    to account for seasonality common in multi-year weekly time series of counts.
    Order 1 sine and cosine terms are included to account for annual seasonality
    that is common to syndromes and diseases such as influenza, RSV, and norovirus.
    Each baseline model is used to make weekly forecasts for all weeks following
    the baseline period. One-sided upper 95% prediction interval bounds are
    computed for each week in the prediction period. Alarms are signaled for
    any week during for which weekly counts fall above the upper bound of
    the prediction interval.

    :param df: A pandas data frame containing time series data
    :param t: Name of the column of type Date containing the dates (Default value = "date")
    :param y: Name of the column of type Numeric containing counts or percentages (Default value = "count")
    :param baseline_end: date of the end of the baseline/training period (in date or string class)
    :param include_time: Indicate whether or not to include time term in regression model (default is True)

    :returns: Original pandas dataframe with model estimates, upper prediction interval bounds,
        a binary alarm indicator field, and a binary indicator field of
        whether or not a time term was included.
    :examples:
    
        >>> from pnssp import alert_nbinom
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> df = pd.DataFrame({
        ...    "date": pd.date_range(start="2014-01-05", end="2022-02-05", freq="W"),
        ...    "count": np.random.poisson(lam=25, size=(len(pd.date_range(start="2014-01-05", end="2022-02-05", freq="W")),))
        ... })
        >>> 
        >>> df_nbinom = alert_nbinom(df, baseline_end = "2020-03-01")
        >>> df_nbinom.head()
    """
    grouped_df = isinstance(df, pd.core.groupby.DataFrameGroupBy)

    # Check for grouping variables
    if grouped_df:
        df = df\
            .apply(lambda data: nb_model(data, t=t, y=y, baseline_end=baseline_end, include_time=include_time))\
            .reset_index(drop=True)
    else:
        unique_dates = df[t].unique()
        if len(unique_dates) != len(df):
            raise ValueError("Error in alert_nbinom: Number of unique dates does not equal the number of rows. Should your dataframe be grouped?")

        df = nb_model(df, t=t, y=y, baseline_end=baseline_end, include_time=include_time)
        df = df.reset_index(drop=True)

    return df