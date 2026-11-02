import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning, SingularMatrixWarning


def get_trends(df, t, data_count, all_count, B):
    """Trend Classification Helper

    Fits rolling binomial models to a daily time series
    of percentages or proportions in order to classify the overall
    trend during the baseline period as significantly increasing,
    significantly decreasing, or stable.

    :param df: A pandas data frame
    :param t: Name of the column of type Date containing the dates
    :param data_count: Name of the column with counts for positive encounters
    :param all_count: Name of the column with total counts of encounters
    :param B: Baseline parameter. The baseline length is the number of days to
        which each binomial model is fit
    :return: A pandas data frame
    """

    df = df.reset_index(drop=True).sort_values(by=t)
    df[t] = pd.to_datetime(df[t])
    statistic = []
    p_value = []

    dates = np.array([(date - df[t][0]).days for date in df[t]])
    data_count = df[data_count].values
    all_count = df[all_count].values

    for j in range(0, len(df)):
        i = j - B
        if i < 0: i = 0
        date_window = dates[i:j]
        data_count_rol = data_count[i:j]
        all_count_rol = all_count[i:j]
        fail_count_rol = all_count_rol - data_count_rol

        if (
            len(date_window) < 3
            or np.unique(date_window).size < 2
            or np.sum(data_count_rol) <= 10
            or np.any(all_count_rol <= 0)
            or np.any(data_count_rol < 0)
            or np.any(fail_count_rol < 0)
            or np.sum(data_count_rol) == 0
            or np.sum(fail_count_rol) == 0
        ):
            statistic.append(np.nan)
            p_value.append(np.nan)
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", category=SingularMatrixWarning)
                    warnings.simplefilter("error", category=PerfectSeparationWarning)
                    warnings.simplefilter("error", category=RuntimeWarning)
                    mod = sm.GLM(
                        np.column_stack([data_count_rol, fail_count_rol]),
                        sm.add_constant(date_window),
                        family=sm.families.Binomial()
                    ).fit()
                statistic.append(mod.params[1])
                p_value.append(mod.pvalues[1])
            except (SingularMatrixWarning, PerfectSeparationWarning, RuntimeWarning, np.linalg.LinAlgError, ValueError):
                statistic.append(np.nan)
                p_value.append(np.nan)

    df["statistic"] = statistic
    df["p_value"] = p_value

    conditions = [
        (df['p_value'] < 0.01) & (df['statistic'] > 0),
        (df['p_value'] < 0.01) & (df['statistic'] < 0),
        df['p_value'].isna(),
    ]

    trends = ["Significant Increase", "Significant Decrease", "Insufficient Data"]

    df["trend_classification"] = np.select(conditions, trends, default="Stable")

    return df


def classify_trend(df, t='date', data_count='dataCount', all_count='allCount', B=12):
    """Trend Classification for Proportions/Percentages

    The algorithm fits rolling binomial models to a daily time series
    of percentages or proportions in order to classify the overall
    trend during the baseline period as significantly increasing,
    significantly decreasing, or stable.

    :param df: A pandas data frame
    :param t: Name of the column of type Date containing the dates
        (Default value = "date")
    :param data_count: Name of the column with counts for positive encounters
        (Default value = "dataCount")
    :param all_count: Name of the column with total counts of encounters
        (Default value = "allCount")
    :param B: Baseline parameter. The baseline length is the number of days to
        which each binomial model is fit (Default value = 12)
    :return: A pandas data frame
    :examples:
        >>> from pynssp import classify_trend
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> df = pd.DataFrame({
        ...     "date": pd.date_range("2020-01-01", "2020-12-31"),
        ...     "dataCount": np.random.randint(0, 101, size=366),
        ...     "allCount": np.random.randint(101, 500, size=366)
        ... })
        >>>
        >>> df_trend = classify_trend(df)
        >>> df_trend.head()
    """

    # Check for grouping variables
    grouped_df = isinstance(df, pd.core.groupby.DataFrameGroupBy)

    if grouped_df:
        group_cols = [df.keys] if isinstance(df.keys, str) else list(df.keys)

        trends_tbl = df\
            .apply(lambda x: get_trends(x, t, data_count=data_count, all_count=all_count, B=B))\
            .reset_index(level=-1, drop=True)\
            .reset_index(drop=False)
        return trends_tbl[group_cols + [col for col in trends_tbl.columns if col not in group_cols]]
    else:
        trends_tbl = get_trends(df, t, data_count=data_count, all_count=all_count, B=B)

    return trends_tbl
