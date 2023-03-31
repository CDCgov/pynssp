import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from tqdm.auto import tqdm
from scipy import stats


def ewma_loop(df, t, y, B, g, w1, w2):
    """Loop for EWMA
    Loop for EWMA and adjustment of outlying smoothed values

    :param df: A pandas data frame
    :param t: Name of the column of type Date containing the dates
    :param y: Numeric vector of counts or percentages
    :param mu: Numeric vector of baseline averages
    :param B: Baseline parameter. The baseline length is the number of days used to
        calculate rolling averages, standard deviations, and exponentially weighted
        moving averages. Defaults to 28 days to match ESSENCE implementation.
    :param g: Guardband parameter. The guardband length is the number of days separating
        the baseline from the current test date. Defaults to 2 days to match ESSENCE
        implementation.
    :param w1: Smoothing coefficient for sensitivity to gradual events. Must be between
        0 and 1 and is recommended to be between 0.3 and 0.5 to account for gradual
        effects. Defaults to 0.4 to match ESSENCE implementation.
    :param w2: Smoothed coefficient for sensitivity to sudden events. Must be between
        0 and 1 and is recommended to be above 0.7 to account for sudden events.
        Defaults to 0.9 to match ESSENCE implementation and approximate the C2 algorithm.
    :returns: A pandas data frame with p-values and test statistics

    """
    # Vector of observations
    y = df[y].tolist()

    N = len(df)

    # Populate algorithm parameters
    min_baseline = 11
    max_baseline = B

    # Initialize result vectors
    z = z1 = z2 = \
        sigma1 = sigma2 = \
        test_stat = test_stat1 = test_stat2 = \
        p_val = pval1 = pval2 = \
        expected = np.repeat(np.nan, N)

    # Initialize EWMA values
    z1[0] = z2[0] = y[0]

    for i_0 in range(1, min_baseline + g):
        z1[i_0] = w1 * y[i_0] + (1 - w1) * z1[i_0 - 1]
        z2[i_0] = w2 * y[i_0] + (1 - w2) * z2[i_0 - 1]

    # Initialize baseline indices
    ndx_baseline = np.arange(0, min_baseline)

    # EWMA loop
    for i in range(min_baseline + g, N):

        # Pad baseline until full baseline is obtained
        if ndx_baseline[-1] < max_baseline - 1:
            ndx_baseline = np.insert(ndx_baseline, 0, -1)

        # Advance baseline for current iteration
        ndx_baseline += 1

        # Set number of degrees of freedom
        n_df = len(ndx_baseline) - 1

        # Baseline and current data
        try:
            y_baseline = y[ndx_baseline]
        except:
            y_baseline = pd.Series(y)[ndx_baseline]

        expected[i] = np.mean(y_baseline)
        sigma = np.std(y_baseline, ddof=1)

        sigma_correction1 = np.sqrt(
            (w1 / (2 - w1))
            + (1 / len(ndx_baseline))
            - 2 * (1 - w1) ** (g + 1)
            * ((1 - (1 - w1) ** len(ndx_baseline)) / len(ndx_baseline))
        )
        sigma_correction2 = np.sqrt(
            (w2 / (2 - w2))
            + (1 / len(ndx_baseline))
            - 2 * (1 - w2) ** (g + 1)
            * ((1 - (1 - w2) ** len(ndx_baseline)) / len(ndx_baseline))
        )

        ucl_alert = np.round(stats.t.ppf(1 - 0.01, df=n_df), 5)
        ucl_warning = np.round(stats.t.ppf(1 - 0.025, df=n_df), 5)

        min_sigma1 = (w1 / ucl_warning) * (1 + 0.5 * (1 - w1)**2)
        min_sigma2 = (w2 / ucl_warning) * (1 + 0.5 * (1 - w2)**2)

        constant1 = (0.1289 - (0.2414 - 0.1826 * (1 - w1)**4) *
                    np.log(10 * 0.05)) * (w1 / ucl_warning)
        constant2 = (0.1289 - (0.2414 - 0.1826 * (1 - w2)**4) *
                    np.log(10 * 0.05)) * (w2 / ucl_warning)

        sigma1[i] = max(min_sigma1, sigma * sigma_correction1 + constant1)
        sigma2[i] = max(min_sigma2, sigma * sigma_correction2 + constant2)

        # EWMA values
        z1[i] = w1 * y[i] + (1 - w1) * z1[i - 1]
        z2[i] = w2 * y[i] + (1 - w2) * z2[i - 1]

        # Calculate test statistics
        test_stat1[i] = (z1[i] - expected[i]) / sigma1[i]
        test_stat2[i] = (z2[i] - expected[i]) / sigma2[i]

        if abs(test_stat1[i]) > ucl_alert:
            z1[i] = expected[i] + np.sign(test_stat1[i]) * ucl_alert * sigma1[i]

        if abs(test_stat2[i]) > ucl_alert:
            z2[i] = expected[i] + np.sign(test_stat2[i]) * ucl_alert * sigma2[i]

        # Compute p-values
        pval1[i] = 1 - stats.t.cdf(test_stat1[i], df=n_df)
        pval2[i] = 1 - stats.t.cdf(test_stat2[i], df=n_df)

        # Determine minimum p-value
        if pval1[i] < pval2[i]:
            p_val[i] = pval1[i]
            test_stat[i] = test_stat1[i]
            z[i] = z1[i]
        else:
            p_val[i] = pval2[i]
            test_stat[i] = test_stat2[i]
            z[i] = z2[i]

    return pd.concat([df, pd.DataFrame({
        'baseline_expected': expected,
        'test_statistic': test_stat,
        'p_value': p_val
    })], axis=0)


def alert_ewma(df, t='date', y='count', B=28, g=2, w1=0.4, w2=0.9):
    """Exponentially Weighted Moving Average (EWMA)
    
    The EWMA compares a weighted average of the most recent visit counts
    to a baseline expectation. For the weighted average to be tested, an exponential
    weighting gives the most influence to the most recent observations.
    This algorithm is appropriate for daily counts that do not have the
    characteristic features modeled in the regression algorithm. It is more applicable
    for Emergency Department data from certain hospital groups and for time series with
    small counts (daily average below 10) because of the limited case definition or
    chosen geographic region. An alert (red value) is signaled if the statistical test
    (student's t-test) applied to the test statistic yields a p-value less than 0.01.
    If the p-value is greater than or equal to 0.01 and strictly less than 0.05, a warning
    (yellow value) is signaled. Blue values are returned if an alert or warning does not
    occur. Grey values represent instances where anomaly detection did not apply
    (i.e., observations for which baseline data were unavailable).

    :param df: A pandas data frame
    :param t: Name of the column of type Date containing the dates (Default value = 'date')
    :param y: Name of the column of type Numeric containing counts or percentages (Default value = 'count')
    :param B: Baseline parameter. The baseline length is the number of days used to
        calculate rolling averages, standard deviations, and exponentially weighted
        moving averages. Defaults to 28 days to match ESSENCE implementation.
    :param g: Guardband parameter. The guardband length is the number of days separating
        the baseline from the current test date. Defaults to 2 days to match ESSENCE
        implementation.
    :param w1: Smoothing coefficient for sensitivity to gradual events. Must be between
        0 and 1 and is recommended to be between 0.3 and 0.5 to account for gradual
        effects. Defaults to 0.4 to match ESSENCE implementation.
    :param w2: Smoothed coefficient for sensitivity to sudden events. Must be between
        0 and 1 and is recommended to be above 0.7 to account for sudden events.
        Defaults to 0.9 to match ESSENCE implementation and approximate the C2 algorithm.
    :returns: Original pandas data frame with detection results.
    :examples:
        # Example 1
        import pandas as pd
        import numpy as np
        from pynssp.detectors.ewma import *
    
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', '2020-12-31'),
            'count': np.random.randint(0, 101, size=366)
        })

        df_ewma = alert_ewma(df)

        df_ewma.head()

    """
    
    # Check baseline length argument
    if B < 7:
        raise ValueError("Error in alert_ewma: baseline length argument 'B' must be greater than or equal to 7")
        
    # Check guardband length argument
    if g < 0:
        raise ValueError("Error in alert_ewma: guardband length argument 'g' cannot be negative")
    
    # Check for sufficient baseline data
    if df.size()[0] < B + g + 1:
        raise ValueError("Error in alert_ewma: not enough historical data")
        
    # Check for grouping variables
    # grouped_df = isinstance(df.index, pd.MultiIndex)
    try:
        grouped_df = df.ngroups > 1
    except:
        grouped_df = False
    
    if grouped_df:
        # groups = df.grouper.names
        # groups = list(df.groups.keys())
        # base_tbl = df.apply(lambda x: x.reset_index(drop=True)).explode(groups)
        
        alert_tbl = df\
            .apply(lambda x: ewma_loop(x, t, y, B, g, w1, w2))
        
        alert_tbl = alert_tbl.reset_index(drop=True)

        alert_tbl['alert'] = np.select(
            [
                alert_tbl['p_value'] < 0.01,
                (alert_tbl['p_value'] >= 0.01) & (alert_tbl['p_value'] < 0.05),
                alert_tbl['p_value'] >= 0.05
            ], 
            ['red', 'yellow', 'blue'], 
            default='grey'
        )

        # alert_tbl = pd.concat([base_tbl, alert_tbl], axis=0)
        
    else:
        base_tbl = df.copy()
        
        if not isinstance(base_tbl[t], pd.DatetimeIndex):
            base_tbl[t] = pd.to_datetime(base_tbl[t])
        
        unique_dates = base_tbl[t].unique()
        
        if len(unique_dates) != df.shape[0]:
            raise ValueError("Error in alert_ewma: Number of unique dates does not equal the number of rows. Should your dataframe be grouped?")
            
        alert_tbl = ewma_loop(base_tbl, t=t, y=y, B=B, g=g, w1=w1, w2=w2)
        
        alert_tbl = alert_tbl.reset_index(drop=True)

        alert_tbl['alert'] = np.select(
            [
                alert_tbl['p_value'] < 0.01,
                (alert_tbl['p_value'] >= 0.01) & (alert_tbl['p_value'] < 0.05),
                alert_tbl['p_value'] >= 0.05
            ], 
            ['red', 'yellow', 'blue'], 
            default='grey'
        )

        # alert_tbl = pd.concat([df, alert_tbl], axis=0)
    
    return alert_tbl