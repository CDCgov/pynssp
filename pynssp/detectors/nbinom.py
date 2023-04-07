import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.api import families, formula
from statsmodels.genmod.families import links

def nb_model(df, t, y, baseline_end, include_time):
    """
    Fits a negative binomial model to predict the count variable y over time t,
    and computes upper confidence bounds and alarm thresholds for baseline and
    prediction periods.

    :param df: pandas DataFrame with columns 't' and 'y', representing the time
        variable and the count variable, respectively.
    :type df: pandas.DataFrame
    :param t: name of the time variable column in df.
    :type t: str
    :param y: name of the count variable column in df.
    :type y: str
    :param baseline_end: date at which the baseline period ends.
    :type baseline_end: datetime.date
    :param include_time: whether to include a time variable (in weeks) in the model.
    :type include_time: bool
    :return: pandas DataFrame with columns 'obs', 'cos', 'sin', 'split',
        'estimate', 'threshold', 'alarm', 'time_term', representing the
        observation number, the cosine and sine of the week number, the split
        period (baseline or prediction), the predicted count estimate, the
        upper alarm threshold, the alarm status, and whether time was included
        in the model.
    """
    df = df.reset_index(drop=True)

    df[t] = pd.to_datetime(df[t])
    df[y] = pd.to_numeric(df[y])

    # Check baseline length and for sufficient historical data
    baseline = df if baseline_end is None else df[df[t] <= pd.to_datetime(baseline_end)]
    baseline_n_wks = baseline[t].nunique()
    baseline_n_yrs = baseline_n_wks / 52
    if baseline_n_yrs < 2:
        raise ValueError("Baseline length must be greater than or equal to 2 years")

    baseline_dates = baseline[t].unique()
    if len(pd.date_range(start=min(baseline_dates), end=max(baseline_dates), freq='W')) != baseline_n_wks:
        raise ValueError("Not all weeks in intended baseline date range were found")

    # Check that time series observations are non-negative integer counts
    ts_obs = df[y]
    ts_obs_int = ts_obs.astype(int)
    if not all(ts_obs == ts_obs_int) or not all(ts_obs >= 0):
        raise ValueError("Time series observations must be non-negative integer counts")

    df['obs'] = np.arange(1, len(df)+1)
    df['cos'] = np.cos(2 * np.pi * df['obs'] / 52.18)
    df['sin'] = np.sin(2 * np.pi * df['obs'] / 52.18)
    df['split'] = np.where(df[t] <= pd.to_datetime(baseline_end), "Baseline Period", "Prediction Period")

    baseline_data = df[df['split'] == "Baseline Period"]
    predict_data = df[df['split'] == "Prediction Period"]

    if include_time:
        formula_str = y + ' ~ obs + cos + sin'
    else:
        formula_str = y + ' ~ cos + sin'

    baseline_model = formula.glm(formula_str, data=baseline_data, family=families.NegativeBinomial(link=links.log()))\
        .fit()
    # df_residual = baseline_model.df_resid
    # theta = baseline_model.scale

    baseline_fit = baseline_data.copy()
    baseline_preds = baseline_model.get_prediction(baseline_fit)
    baseline_pred_ci = pd.DataFrame(baseline_preds.conf_int(alpha=0.05), columns=['lower_ci', 'upper_ci'])
    baseline_fit['estimate'] = baseline_preds.predicted_mean
    # baseline_fit['se_link'] = baseline_model.bse
    # baseline_fit['fit_link'] = np.log(baseline_fit['fit_link'])
    baseline_fit['upper_ci'] = baseline_pred_ci['upper_ci'].tolist() #np.exp(baseline_fit['fit_link'] + stats.anglit.ppf(1 - 0.05/2, df_residual) * baseline_fit['se_link'])
    # baseline_fit['threshold'] = stats.nbinom.ppf(1 - 0.05, n=theta, p=theta/(theta+baseline_fit['estimate']))

    predict_fit = predict_data.copy()
    predict_preds = baseline_model.get_prediction(predict_fit)
    predict_pred_ci = pd.DataFrame(predict_preds.conf_int(alpha=0.05), columns=['lower_ci', 'upper_ci'])
    predict_fit['estimate'] = predict_preds.predicted_mean
    # predict_fit['se_link'] = baseline_model.get_prediction(predict_data).se_mean
    # predict_fit['fit_link'] = np.exp(predict_fit['fit_link'])
    predict_fit['upper_ci'] = predict_pred_ci['upper_ci'].tolist() #np.exp(predict_fit['fit_link'] + stats.t.ppf(1 - 0.05/2, df_residual) * predict_fit['se_link'])
    # predict_fit['threshold'] = stats.nbinom.ppf(1 - 0.05, n=theta, p=theta/(theta+predict_fit['estimate']))

    result = pd.concat([baseline_fit, predict_fit])
    result.sort_values(by=t, inplace=True)
    result.reset_index(drop=True, inplace=True)
    result['split'] = pd.Categorical(result['split'], categories=["Baseline Period", "Prediction Period"])
    result['alarm'] = np.where(result[y] > result['threshold'], True, False)
    result['time_term'] = include_time
    result.drop(columns=['obs', 'cos', 'sin', 'upper_ci'], inplace=True)

    return result


def alert_nbinom(df, baseline_end, t='date', y='count', include_time=True):
    """
    :examples:
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            'date': pd.date_range(start='2014-01-05', end='2022-02-05', freq='W'),
            'count': np.random.poisson(lam=25, size=(len(pd.date_range(start='2014-01-05', end='2022-02-05', freq='W')),))
        })

        df_nbinom = alert_nbinom(df, baseline_end = "2020-03-01")
    """
    grouped_df = isinstance(df, pd.core.groupby.DataFrameGroupBy)

    # df[t] = pd.to_datetime(df[t])
    # df[y] = pd.to_numeric(df[y])

    # # Check baseline length and for sufficient historical data
    # baseline = df if baseline_end is None else df[df[t] <= pd.to_datetime(baseline_end)]
    # baseline_n_wks = baseline[t].nunique()
    # baseline_n_yrs = baseline_n_wks / 52
    # if baseline_n_yrs < 2:
    #     raise ValueError("Error in {.fn alert_nbinom}: baseline length must be greater than or equal to 2 years")

    # baseline_dates = baseline[t].unique()
    # if len(pd.date_range(start=min(baseline_dates), end=max(baseline_dates), freq='W')) != baseline_n_wks:
    #     raise ValueError("Error in alert_nbinom: not all weeks in intended baseline date range were found")

    # # Check that time series observations are non-negative integer counts
    # ts_obs = df[y]
    # ts_obs_int = ts_obs.astype(int)
    # if not all(ts_obs == ts_obs_int) or not all(ts_obs >= 0):
    #     raise ValueError("Error in {.fn alert_nbinom}: time series observations must be non-negative integer counts")

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