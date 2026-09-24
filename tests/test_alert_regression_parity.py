"""Rnssp parity regression test for adaptive_regression.

CDCgov/pynssp issue #8: "Match the implementation of pynssp's
alert_regression() with Rnssp's version of the function."

The R package CDCgov/Rnssp (R/alert_regression.R) corrected two things in
commit ae5ba22 that pynssp never picked up:

1. ``min_sigma <- 0.01 / ucl_warning`` (pynssp had ``1.01 / ucl_warning``),
   a ~100x inflation of the standard-error floor that shrinks test
   statistics and hides genuine alerts on smooth (low-noise) series.
2. The R code indexes its 1-based ``min_sigma`` vector as ``min_sigma[n_df]``
   (the t-quantile at df == n_df); the 0-based Python equivalent is
   ``min_sigma[n_df - 1]``. pynssp used ``min_sigma[n_df]``, silently picking
   the quantile for df == n_df + 1.

The oracle below is translated directly from the R source of
Rnssp::adaptive_regression (not from pynssp), so agreement between the two
independent implementations is the parity check.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats

from pynssp.detectors.regression import adaptive_regression


def rnssp_adaptive_regression(df, t, y, B, g):
    """Independent Python translation of Rnssp::adaptive_regression (R).

    Mirrors CDCgov/Rnssp R/alert_regression.R on master:
      df_range <- 1:(B - min_df); ucl_warning <- round(qt(1 - .05, df_range), 5)
      min_sigma <- 0.01 / ucl_warning
      for (i in (min_baseline + g + 1):N)  # R 1-based indexing
      mse <- (1 / n_df) * sum(res^2); n_df <- length(ndx_baseline) - 8
      sigma[i] <- max(sqrt(mse) * sqrt(((B_length + 7) * (B_length - 4)) /
                                       (B_length * (B_length - 7))),
                      min_sigma[n_df])     # 1-based lookup
    """
    df = df.reset_index(drop=True).sort_values(by=t)
    work = df.copy()
    work["dow"] = pd.to_datetime(work[t]).dt.strftime("%A").str[:3]
    work["dummy"] = 1
    work = pd.concat(
        [work,
         work.pivot_table(index=t, columns="dow", values="dummy")
             .fillna(0).reset_index(drop=True)],
        axis=1,
    )
    dates = pd.to_datetime(work[t]).tolist()
    y_obs = work[y].tolist()
    N = len(work)

    min_df = 3
    min_baseline = 11
    max_baseline = B
    df_range = np.arange(1, B - min_df + 1)
    ucl_warning = np.round(stats.t.ppf(1 - 0.05, df=df_range), 5)
    min_sigma = 0.01 / ucl_warning

    test_stat = np.repeat(np.nan, N)
    p_val = np.repeat(np.nan, N)
    expected = np.repeat(np.nan, N)
    sigma = np.repeat(np.nan, N)
    r_sqrd_adj = np.repeat(np.nan, N)

    ndx_baseline = np.arange(1, min_baseline)

    for i in range(min_baseline + g, N):  # R 1-based (min_baseline + g + 1):N
        if ndx_baseline[-1] < max_baseline:
            ndx_baseline = np.insert(ndx_baseline, 0, 0)
        ndx_baseline = ndx_baseline + 1

        if ndx_baseline[-1] < max_baseline:
            ndx_time = np.arange(1, len(ndx_baseline) + 1)
            ndx_test = int(ndx_baseline[-1] + g + 1)
        else:
            ndx_time = np.arange(1, B + 1)
            ndx_test = B + g + 1

        n_df = len(ndx_baseline) - 8
        baseline_data = work.iloc[ndx_baseline - 1, :]
        B_length = len(ndx_baseline)
        baseline_obs = baseline_data[y].to_numpy(dtype=float)

        X = np.column_stack([
            ndx_time,
            baseline_data[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]]
                .to_numpy(dtype=float),
        ])
        X = np.column_stack([np.ones(len(X)), X])
        beta, _, _, _ = np.linalg.lstsq(X, baseline_obs, rcond=None)
        res = baseline_obs - X @ beta
        mse = (1.0 / n_df) * np.sum(res ** 2)

        fit_vals = X @ beta
        mss = np.sum((fit_vals - fit_vals.mean()) ** 2)
        rss = np.sum(res ** 2)
        r2 = mss / (rss + mss)
        r2_adj = 1 - (1 - r2) * ((X.shape[0] - 1) / (X.shape[0] - X.shape[1]))
        r_sqrd_adj[i] = 0 if np.isnan(r2_adj) else r2_adj

        sigma[i] = max(
            np.sqrt(mse) * np.sqrt(((B_length + 7) * (B_length - 4)) /
                                   (B_length * (B_length - 7))),
            min_sigma[n_df - 1],  # R 1-based min_sigma[n_df]
        )

        dow_test = int(dates[i].strftime("%u"))
        if dow_test < 7:
            expected[i] = max(0.0, beta[0] + ndx_test * beta[1] + beta[dow_test + 1])
        else:
            expected[i] = max(0.0, beta[0] + ndx_test * beta[1])

        test_stat[i] = (y_obs[i] - expected[i]) / sigma[i]
        p_val[i] = 1 - stats.t.cdf(test_stat[i], n_df)

    return pd.DataFrame({
        "baseline_expected": expected,
        "test_statistic": test_stat,
        "p_value": p_val,
        "sigma": sigma,
        "adjusted_r_squared": r_sqrd_adj,
    })


def _smooth_series_with_spike():
    """Deterministic smooth series with one genuine modest spike.

    The low-noise baseline makes the OLS residuals tiny, so the sigma floor
    (the term this test targets) binds on every fitted row.
    """
    n = 120
    dates = pd.date_range("2021-01-01", periods=n)
    count = (20.0 + 0.5 * np.arange(n)
             + 0.05 * np.sin(2 * np.pi * np.arange(n) / 7.0))
    count[100] += 0.6  # genuine spike on 2021-04-11
    return pd.DataFrame({"date": dates, "count": count})


def test_adaptive_regression_matches_rnssp():
    """pynssp.adaptive_regression must agree with the R-faithful oracle."""
    df = _smooth_series_with_spike()
    ref = rnssp_adaptive_regression(df, "date", "count", 28, 2)
    got = adaptive_regression(df, "date", "count", 28, 2)

    for col in ["sigma", "test_statistic", "p_value",
                "baseline_expected", "adjusted_r_squared"]:
        assert np.allclose(ref[col].to_numpy(), got[col].to_numpy(),
                           equal_nan=True), (
            f"column {col!r} diverges from Rnssp reference "
            f"(max |diff| = {np.nanmax(np.abs(ref[col] - got[col])):.3e})"
        )


def test_spike_day_alert_level_matches_rnssp():
    """The spike day must reach red-alert significance, as in Rnssp.

    With the old 1.01 floor the standard error was inflated ~100x and the
    spike's p-value came out ~0.16 (blue); Rnssp reports p ~ 0 (red).
    """
    df = _smooth_series_with_spike()
    got = adaptive_regression(df, "date", "count", 28, 2)
    assert got["p_value"].iloc[100] < 0.01
