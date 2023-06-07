import pandas as pd
import numpy as np
import statsmodels.api as sm

def serfling_model(df, t, y, baseline_end):
    """Original Serfling method for weekly time series

    Serfling model helper function for monitoring weekly time series with seasonality

    :param df: A pandas data frame
    :param t: Name of the column of type Date containing the dates
    :param y: Name of the column of type Numeric containing counts
    :param baseline_end: date of the end of the baseline/training period (date or string class)
    :returns: A pandas data frame.
    """
    df = df.reset_index(drop=True).sort_values(by=t)
    input_data = df.copy()
    input_data[t] = pd.to_datetime(input_data[t])
    input_data[y] = pd.to_numeric(input_data[y])

    # Check baseline length and for sufficient historical data
    baseline = input_data[input_data[t] <= pd.to_datetime(baseline_end)]
    baseline_dates = baseline[t]
    baseline_n_wks = len(baseline_dates.unique())
    baseline_n_yrs = baseline_n_wks / 52
    
    if baseline_n_yrs < 2:
        raise ValueError("Baseline length must be greater than or equal to 2 years.")
    
    if len(pd.date_range(start=min(baseline_dates), end=max(baseline_dates), freq="W")) != baseline_n_wks:
        raise ValueError("Not all weeks in intended baseline date range were found.")

    input_data["obs"] = input_data.index + 1
    input_data["cos"] = np.cos((2 * np.pi * input_data["obs"]) / 52.18)
    input_data["sin"] = np.sin((2 * np.pi * input_data["obs"]) / 52.18)
    input_data["split"] = np.where(input_data[t] <= baseline_end, "Baseline Period", "Prediction Period")
    input_data["epidemic_period"] = np.where((input_data[t].dt.month >= 10) | (input_data[t].dt.month <= 5), True, False)

    baseline_data = input_data.loc[(input_data["split"] == "Baseline Period") & (~input_data["epidemic_period"])]

    X = sm.add_constant(baseline_data[["obs", "cos", "sin"]])
    Y = baseline_data[y]

    baseline_model = sm.OLS(Y, X).fit()

    input_data["estimate"], _, _, _, _, input_data["threshold"] = \
        baseline_model.get_prediction(sm.add_constant(input_data[["obs", "cos", "sin"]]))\
                      .summary_frame(alpha=0.1).values.T
    input_data = input_data.drop(columns=["obs", "cos", "sin"])
    input_data["split"] = pd.Categorical(input_data["split"], categories=["Baseline Period", "Prediction Period"])
    input_data["alarm"] = np.where(input_data[y] > input_data["threshold"], True, False)

    return input_data


def alert_serfling(df, baseline_end, t="date", y="count"):
    """Original Serfling method for weekly time series

    The original Serfling algorithm fits a linear regression model with
    a time term and order 1 Fourier terms to a baseline period that ideally spans
    5 or more years. Inclusion of Fourier terms in the model is intended to
    account for seasonality common in multi-year weekly time series. Order 1 sine
    and cosine terms are included to account for annual seasonality that is
    common to syndromes and diseases such as influenza, RSV, and norovirus. Each
    baseline model is used to make weekly forecasts for all weeks following the
    baseline period. One-sided upper 95% prediction interval bounds are computed
    for each week in the prediction period. Alarms are signaled for any week
    during for which weekly observations fall above the upper bound of the
    prediction interval. This implementation follows the approach of the original
    Serfling method in which weeks between October of the starting year of a
    season and May of the ending year of a season are considered to be in the
    epidemic period. Weeks in the epidemic period are removed from the baseline
    prior to fitting the regression model.
    
    :param df: A pandas data frame containing time series data
    :param t: Name of the column of type Date containing the dates (Default value = "date")
    :param y: Name of the column of type Numeric containing counts or percentages (Default value = "count")
    :param baseline_end: date of the end of the baseline/training period (in date or string class)
    :returns: Original pandas dataframe with model estimates, upper prediction interval bounds,
        a binary alarm indicator field, and a binary indicator
    :examples:
    
        >>> from pynssp import alert_serfling
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> df = pd.DataFrame({
        ...    "date": pd.date_range(start="2014-01-05", end="2022-02-05", freq="W"),
        ...    "count": np.random.poisson(lam=25, size=(len(pd.date_range(start="2014-01-05", end="2022-02-05", freq="W")),))
        ... })
        >>> 
        >>> df_serfling  = alert_serfling(df, baseline_end = "2020-03-01")
        >>> df_serfling.head()
    """
    grouped_df = isinstance(df, pd.core.groupby.DataFrameGroupBy)
    
    if grouped_df:
        return df.apply(lambda x: serfling_model(x, t, y, baseline_end)).reset_index(drop=True)
    else:
        return serfling_model(df, t, y, baseline_end).reset_index(drop=True)
