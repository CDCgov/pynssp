import pandas as pd
import numpy as np
from .ewma import *
from .regression import *

def alert_switch(df, t="date", y="count", B=28, g=2, w1=0.4, w2=0.9):
    """Regression/EWMA Switch

    The NSSP-ESSENCE Regression/EWMA Switch algorithm generalized the Regression and
    EWMA algorithms by applying the most appropriate algorithm for the data in the
    baseline. First, multiple adaptive regression is applied where the adjusted R
    squared value of the model is examined to see if it meets a threshold of 0.60. If
    this threshold is not met, then the model is considered to not explain the data well.
    In this case, the algorithm switches to the EWMA algorithm, which is more appropriate
    for sparser time series that are common with county level trends. The smoothing
    coefficient for the EWMA algorithm is fixed to 0.4.

    :param df: A dataframe containing the time series data.
    :param t: The name of the column in `df` containing the time information.
            Defaults to "date".
    :param y: The name of the column in `df` containing the values to be analyzed.
            Defaults to "count".
    :param B: The length of the baseline period in days, must be a multiple of 7 and
            greater than or equal to 7. Defaults to 28.
    :param g: The length of the guardband period in days. Must be non-negative.
            Defaults to 2.
    :param w1: Smoothing coefficient for sensitivity to gradual events. Must be between
            0 and 1 and is recommended to be between 0.3 and 0.5 to account for gradual
            effects. Defaults to 0.4 to match NSSP-ESSENCE implementation.
    :param w2: Smoothed coefficient for sensitivity to sudden events. Must be between
            0 and 1 and is recommended to be above 0.7 to account for sudden events.
            Defaults to 0.9 to match NSSP-ESSENCE implementation and approximate the C2 algorithm.
    :returns: A dataframe containing the results of the analysis.
    :examples:
    
        >>> from pynssp import alert_switch
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> df = pd.DataFrame({
        ...    "date": pd.date_range("2020-01-01", "2020-12-31"),
        ...    "count": np.random.randint(0, 101, size=366)
        ... })
        >>> 
        >>> df_switch = alert_switch(df)
        >>> df_switch.head()
    """
  
    # Check baseline length argument
    if B < 7:
      raise ValueError("Error in alert_switch: baseline length argument B must be greater than or equal to 7")
    
    if B % 7 != 0:
      raise ValueError("Error in alert_switch: baseline length argument B must be a multiple of 7")
    
    # Check guardband length argument
    if g < 0:
      raise ValueError("Error in alert_switch: guardband length argument g cannot be negative")
    
    # Check for sufficient baseline data
    grouped_df = isinstance(df, pd.core.groupby.DataFrameGroupBy)

    if not grouped_df:
        df_size = df.size
    else:
        df_size = df.size()[0]

    # Check for sufficient baseline data
    if df_size < B + g + 1:
      raise ValueError("Error in alert_switch: not enough historical data")
    
    # Check for grouping variables
    base_tbl = df
    s_cols = ["dow", "dummy", "sigma"] if grouped_df else ["dow", "dummy", "sigma", "index"]

    alert_tbl_reg = alert_regression(base_tbl, t, y, B, g)\
        .drop(columns=s_cols)\
        .reset_index(drop=True)
    alert_tbl_reg["detector"] = "Adaptive Multiple Regression"

    alert_tbl_ewma = alert_ewma(base_tbl, t, y, B, g, w1, w2)\
        .reset_index(drop=True)
    alert_tbl_ewma["detector"] = "EWMA"

    stats_cols = [
       "baseline_expected", "test_statistic", "p_value", 
       "adjusted_r_squared", "alert", "detector"
    ]

    join_cols = list(set(alert_tbl_reg.columns) - set(stats_cols))

    replace_dates = pd.merge(
       alert_tbl_reg\
        .query("(adjusted_r_squared.isna()) | (adjusted_r_squared < 0.60)", engine="python")\
        .drop(columns=stats_cols),
        alert_tbl_ewma,
        on=join_cols,
        how="inner"
    )

    if grouped_df:
       groups = list(df.grouper.names)
       combined_out = pd.concat([
         alert_tbl_reg.query("(adjusted_r_squared >= 0.60)", engine="python"),
         replace_dates
        ]).drop(columns="adjusted_r_squared")\
          .sort_values(by=groups + [t])
       
       combined_out["detector"] = np.where(
         combined_out["test_statistic"].isna(), 
         np.nan, 
         combined_out["detector"]
        )
    else:
       combined_out = pd.concat([
          alert_tbl_reg\
          .query("(adjusted_r_squared >= 0.60)", engine="python"),
          replace_dates
        ]).drop(columns="adjusted_r_squared")\
          .sort_values(by=t)
       
       combined_out["detector"] = np.where(
          combined_out["test_statistic"].isna(), 
          np.nan, 
          combined_out["detector"]
        )

    return combined_out