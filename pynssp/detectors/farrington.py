import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import nbinom, norm, poisson, t as student_t


def _to_date_series(values):
	"""Parse values as normalized pandas datetimes.

	:param values: Values to convert to datetimes.
	:returns: Datetime series normalized to midnight.
	:raises ValueError: If any value cannot be parsed as a date.
	"""
	try:
		return pd.to_datetime(values, errors="raise").dt.normalize()
	except Exception as exc:
		raise ValueError(
			"Date argument `t` is not in a standard unambiguous format. "
			"Dates must be in `%Y-%m-%d` format."
		) from exc


def _weekday_sunday_zero(dates):
	"""Return weekday numbers using Sunday as ``0``.

	:param dates: Date-like values.
	:returns: Weekday indices aligned to R's ``%w`` convention.
	"""
	idx = pd.DatetimeIndex(dates)
	return ((idx.dayofweek + 1) % 7).to_numpy(dtype=int)


def _fit_glm_poisson(y, X, weights=None):
	"""Fit a Poisson GLM and suppress fitting warnings.

	:param y: Response counts.
	:param X: Design matrix.
	:param weights: Optional observation weights.
	:returns: Fitted GLM result or ``None`` when fitting fails.
	"""
	try:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			model = sm.GLM(
				y,
				X,
				family=sm.families.Poisson(),
				freq_weights=weights,
			)
			result = model.fit()
		return result
	except Exception:
		return None


def _dispersion(result):
	"""Compute a quasi-Poisson style dispersion estimate.

	:param result: Fitted GLM result.
	:returns: Pearson chi-square divided by residual degrees of freedom.
	"""
	if result is None:
		return np.nan
	if result.df_resid > 0:
		return result.pearson_chi2 / result.df_resid
	return 1.0


def _hatvalues(result):
	"""Return leverage values for a fitted GLM.

	:param result: Fitted GLM result.
	:returns: Diagonal hat matrix values.
	"""
	try:
		influence = result.get_influence()
		return np.asarray(influence.hat_matrix_diag, dtype=float)
	except Exception:
		return np.zeros(len(result.model.endog), dtype=float)


def _quasi_p_value(result, param_name, phi):
	"""Estimate two-sided p-values for quasi-Poisson coefficients.

	:param result: Fitted GLM result.
	:param param_name: Parameter to evaluate.
	:param phi: Dispersion estimate.
	:returns: Two-sided p-value for the specified coefficient.
	"""
	if result is None or param_name not in result.params.index:
		return np.nan

	cov = np.asarray(result.normalized_cov_params, dtype=float)
	idx = result.params.index.get_loc(param_name)
	var = phi * cov[idx, idx]
	if not np.isfinite(var) or var <= 0:
		return np.nan

	se = math.sqrt(var)
	t_value = float(result.params.iloc[idx] / se)

	if result.df_resid > 0:
		return float(2 * student_t.sf(np.abs(t_value), df=result.df_resid))
	return float(2 * norm.sf(np.abs(t_value)))


def _predict_glm(result, row_dict, prediction_type="response"):
	"""Predict from a fitted GLM on link or response scale.

	:param result: Fitted GLM result.
	:param row_dict: Mapping of predictor name to value.
	:param prediction_type: ``"response"`` or ``"link"``.
	:returns: Predicted value and standard error for the chosen scale.
	"""
	param_names = list(result.params.index)
	x = np.array([float(row_dict.get(name, 0.0)) for name in param_names], dtype=float)
	beta = result.params.to_numpy(dtype=float)

	eta = float(np.dot(x, beta))

	cov = np.asarray(result.normalized_cov_params, dtype=float)
	var_eta = float(x @ cov @ x.T)
	if not np.isfinite(var_eta) or var_eta < 0:
		var_eta = np.nan

	se_eta = math.sqrt(var_eta) if np.isfinite(var_eta) else np.nan

	if prediction_type == "link":
		return eta, se_eta

	mu = float(math.exp(eta))
	se_response = abs(mu) * se_eta if np.isfinite(se_eta) else np.nan
	return mu, se_response


def _compute_weights(result, phi, threshold):
	"""Compute robust Farrington weights from Anscombe residuals.

	:param result: Fitted GLM result.
	:param phi: Dispersion estimate.
	:param threshold: Residual threshold used to down-weight outliers.
	:returns: Observation weights.
	"""
	y_observed = np.asarray(result.model.endog, dtype=float)
	y_fit = np.asarray(result.fittedvalues, dtype=float)
	diag = _hatvalues(result)

	with np.errstate(divide="ignore", invalid="ignore"):
		ambscombe_resid = (
			(3.0 / 2.0)
			* (np.power(y_observed, 2.0 / 3.0) * np.power(y_fit, -1.0 / 6.0) - np.sqrt(y_fit))
		) / np.sqrt(phi * (1.0 - diag))

		scaled = np.where(ambscombe_resid > threshold, 1.0 / (ambscombe_resid ** 2), 1.0)

	denom = np.nansum(scaled)
	gamma = (len(ambscombe_resid) / denom) if np.isfinite(denom) and denom > 0 else 1.0

	with np.errstate(divide="ignore", invalid="ignore"):
		omega = np.where(ambscombe_resid > threshold, gamma / (ambscombe_resid ** 2), gamma)

	omega = np.where(np.isfinite(omega), omega, gamma)
	return omega.astype(float)


def _build_modified_design(base_dates_num, fct_levels, include_time):
	"""Build the modified Farrington regression design matrix.

	:param base_dates_num: Baseline week offsets.
	:param fct_levels: Seasonal factor levels.
	:param include_time: Whether to include the linear time term.
	:returns: Design matrix and ordered seasonal levels.
	"""
	cat = pd.Categorical(fct_levels)
	levels = list(cat.categories)

	X = pd.DataFrame({"intercept": np.ones(len(cat), dtype=float)})
	if include_time:
		X["base_dates"] = np.asarray(base_dates_num, dtype=float)

	for lvl in levels[1:]:
		X[f"fct_{lvl}"] = (cat == lvl).astype(float)

	return X, levels


def _matches_level(level, target):
	"""Compare seasonal levels allowing numeric/string equivalence.

	:param level: Candidate level.
	:param target: Target level.
	:returns: ``True`` when both levels represent the same value.
	"""
	try:
		return float(level) == float(target)
	except Exception:
		return str(level) == str(target)


def _build_modified_prediction_row(pred_week_time, levels, p_value, include_time):
	"""Construct a single-row predictor mapping for modified Farrington.

	:param pred_week_time: Week index of the prediction date.
	:param levels: Seasonal levels used by the fitted model.
	:param p_value: Seasonal level to predict against.
	:param include_time: Whether to include the time predictor.
	:returns: Row mapping suitable for prediction.
	"""
	row = {"intercept": 1.0}
	if include_time:
		row["base_dates"] = float(pred_week_time)

	for lvl in levels[1:]:
		row[f"fct_{lvl}"] = 1.0 if _matches_level(lvl, p_value) else 0.0

	return row


def _r_index_sequence(start_idx, end_idx):
	"""Return an inclusive index sequence matching R's ``:`` behavior.

	:param start_idx: First index.
	:param end_idx: Last index.
	:returns: Inclusive sequence ascending or descending.
	"""
	if start_idx <= end_idx:
		return list(range(start_idx, end_idx + 1))
	return list(range(start_idx, end_idx - 1, -1))


def _assign_with_extension(values, start_idx, end_idx, value):
	"""Assign values over an inclusive 1-based range, extending storage as needed.

	:param values: Mutable sequence receiving assignments.
	:param start_idx: Start index (1-based, inclusive).
	:param end_idx: End index (1-based, inclusive).
	:param value: Value to assign.
	:returns: ``None``
	"""
	idx_seq = _r_index_sequence(int(start_idx), int(end_idx))
	positive_idx = [x for x in idx_seq if x >= 1]
	if not positive_idx:
		return

	max_idx = max(positive_idx)
	if max_idx > len(values):
		values.extend([np.nan] * (max_idx - len(values)))

	for idx in positive_idx:
		values[idx - 1] = value


def seasonal_groups(B=4, g=27, w=3, p=10, base_length=None, base_weeks=None):
	"""Return the seasonal factor vector for modified Farrington.

	This helper reproduces the 10-level seasonality construction used by the
	R implementation.

	:param B: Number of baseline years.
	:param g: Guardband weeks separating baseline from test week.
	:param w: Half-window width around each reference date.
	:param p: Number of seasonal periods in each baseline year.
	:param base_length: Total number of baseline weeks.
	:param base_weeks: Baseline week indices for reference windows.
	:returns: Seasonal factor levels.
	"""
	h = np.concatenate(([1], np.diff(base_weeks)))
	csum_h = np.cumsum(h)

	fct_levels = [0.0] * int(base_length)

	for i in range(1, B + 1):
		_assign_with_extension(fct_levels, csum_h[i - 1], csum_h[i - 1] + 2 * w, p)

		delta_weeks = int(h[i]) - (2 * w + 1)
		quotient = delta_weeks // (p - 1)
		remainder = delta_weeks % (p - 1)

		fct_lengths = [quotient] * (p - 1)
		for k in range(remainder):
			fct_lengths[k] += 1

		fct_lengths = [0] + fct_lengths
		cum_lengths = np.cumsum(fct_lengths)

		for j in range(1, p):
			start_idx = csum_h[i - 1] + 2 * w + 1 + cum_lengths[j - 1]
			end_idx = csum_h[i - 1] + 2 * w + cum_lengths[j]
			_assign_with_extension(fct_levels, start_idx, end_idx, j)

	trim_len = len(fct_levels) - (g - 1) + w
	if trim_len < 0:
		trim_len = 0
	fct_trimmed = fct_levels[:trim_len]
	return pd.Categorical(fct_trimmed)


def farrington_original(df, t="date", y="count", B=4, w=3):
	"""Original Farrington algorithm for weekly count surveillance.

	Quasi-Poisson models are fitted on historical reference windows from
	previous years, with robust weighting to reduce the influence of outliers.
	Prediction intervals are generated from a variance-stabilized transform and
	used to produce alert scores and color labels.

	:param df: Input dataframe containing weekly observations.
	:param t: Name of the date column.
	:param y: Name of the count column.
	:param B: Number of years to include in the baseline.
	:param w: Half the number of weeks in each reference window.
	:returns: A pandas data frame with Farrington detection outputs.
	"""
	N = len(df)
	min_obs = 52 * B + w + 2

	predicted = np.full(N, np.nan)
	time_coefficient = np.full(N, np.nan)
	include_time_term = np.full(N, np.nan, dtype=object)
	upper = np.full(N, np.nan)
	alert_score = np.full(N, np.nan)
	alert = np.full(N, np.nan, dtype=object)

	dates = pd.to_datetime(df[t]).dt.normalize().to_numpy(dtype="datetime64[ns]")
	y_obs = df[y].astype(float).to_numpy()

	for i in range(min_obs - 1, N):
		current_date = pd.Timestamp(dates[i])

		ref_dates = np.array(
			[current_date - pd.DateOffset(years=year_back) for year_back in range(1, B + 1)],
			dtype="datetime64[ns]",
		)

		wday_gaps = _weekday_sunday_zero(ref_dates) - int((current_date.dayofweek + 1) % 7)
		ref_dates_shifted = ref_dates - wday_gaps.astype("timedelta64[D]")

		floor_ceiling_dates = np.where(
			ref_dates_shifted > ref_dates,
			ref_dates_shifted - np.timedelta64(7, "D"),
			ref_dates_shifted + np.timedelta64(7, "D"),
		)

		center_dates = np.sort(
			np.where(
				np.abs(ref_dates - floor_ceiling_dates) < np.abs(ref_dates - ref_dates_shifted),
				floor_ceiling_dates,
				ref_dates_shifted,
			)
		)

		base_start = np.sort(center_dates - np.timedelta64(7 * w, "D"))[:B]

		idx_start = np.flatnonzero(np.isin(dates, base_start))
		idx = np.concatenate([start + np.arange(7) for start in idx_start])
		idx = idx[(idx >= 0) & (idx < N)]
		if len(idx) == 0:
			continue

		min_date = dates[idx].min()
		base_dates_num = ((dates[idx] - min_date) / np.timedelta64(7, "D")).astype(float)
		base_counts = y_obs[idx]

		X_time = pd.DataFrame(
			{
				"intercept": np.ones(len(base_counts), dtype=float),
				"base_dates": base_dates_num,
			}
		)
		X_no_time = pd.DataFrame({"intercept": np.ones(len(base_counts), dtype=float)})

		mod = _fit_glm_poisson(base_counts, X_time)
		include_time = bool(mod is not None and getattr(mod, "converged", False))
		X_current = X_time

		if not include_time:
			mod = _fit_glm_poisson(base_counts, X_no_time)
			include_time = False
			X_current = X_no_time

		if mod is None or not getattr(mod, "converged", False):
			continue

		phi = max(_dispersion(mod), 1.0)

		if include_time:
			time_coeff = float(mod.params.get("base_dates", np.nan))
			time_p_val = _quasi_p_value(mod, "base_dates", phi)
		else:
			time_coeff = np.nan
			time_p_val = np.nan

		omega = _compute_weights(mod, phi, threshold=1.0)

		mod_weighted = _fit_glm_poisson(base_counts, X_current, weights=omega)
		if mod_weighted is None or not getattr(mod_weighted, "converged", False):
			continue

		phi_weighted = max(_dispersion(mod_weighted), 1.0)
		time_pval_weighted = _quasi_p_value(mod_weighted, "base_dates", phi_weighted)

		pred_week_time = float((dates[i] - min_date) / np.timedelta64(7, "D"))

		pred_row = {"intercept": 1.0}
		if "base_dates" in X_current.columns:
			pred_row["base_dates"] = pred_week_time

		pred_fit, pred_se = _predict_glm(mod_weighted, pred_row, prediction_type="response")

		time_significant = time_pval_weighted < 0.05
		pred_ok = pred_fit <= np.nanmax(base_counts)
		trend = include_time and pred_ok

		if not trend:
			mod = _fit_glm_poisson(base_counts, X_no_time)
			if mod is None or not getattr(mod, "converged", False):
				continue

			phi = max(_dispersion(mod), 1.0)
			omega = _compute_weights(mod, phi, threshold=1.0)

			mod_weighted = _fit_glm_poisson(base_counts, X_no_time, weights=omega)
			if mod_weighted is None or not getattr(mod_weighted, "converged", False):
				continue

			phi_weighted = max(_dispersion(mod_weighted), 1.0)

			pred_fit, pred_se = _predict_glm(
				mod_weighted,
				{"intercept": 1.0},
				prediction_type="response",
			)

			include_time_term[i] = False

		predicted[i] = pred_fit
		time_coefficient[i] = time_coeff
		include_time_term[i] = True

		tau = phi_weighted + ((pred_se ** 2) / predicted[i])
		se_pred = math.sqrt((4.0 / 9.0) * (predicted[i] ** (1.0 / 3.0)) * tau)

		upper[i] = max(0.0, (predicted[i] ** (2.0 / 3.0) + norm.ppf(0.95) * se_pred) ** (3.0 / 2.0))

		if np.isfinite(upper[i]):
			alert_score[i] = (y_obs[i] - predicted[i]) / (upper[i] - predicted[i])
		else:
			alert_score[i] = np.nan

		recent_counts = np.sum(y_obs[(i - 4) : (i + 1)])
		alert[i] = "red" if (alert_score[i] > 1 and recent_counts > 5) else "blue"

	return pd.DataFrame(
		{
			"predicted": predicted,
			"time_coefficient": time_coefficient,
			"include_time_term": include_time_term,
			"upper": upper,
			"alert_score": alert_score,
			"alert": alert,
		}
	)


def farrington_modified(df, t="date", y="count", B=4, g=27, w=3, p=10):
	"""Modified Farrington algorithm with seasonal factor adjustment.

	This variant extends baseline coverage, applies a multi-level seasonal
	factor, and computes thresholds from Poisson or negative-binomial quantiles
	based on the weighted dispersion estimate.

	:param df: Input dataframe containing weekly observations.
	:param t: Name of the date column.
	:param y: Name of the count column.
	:param B: Number of years to include in the baseline.
	:param g: Guardband weeks separating baseline and test week.
	:param w: Half the number of weeks in each reference window.
	:param p: Number of seasonal periods in the baseline.
	:returns: A pandas data frame with Modified Farrington detection outputs.
	"""
	N = len(df)
	min_obs = 52 * B + w + 2

	predicted = np.full(N, np.nan)
	time_coefficient = np.full(N, np.nan)
	include_time_term = np.full(N, np.nan, dtype=object)
	upper = np.full(N, np.nan)
	alert_score = np.full(N, np.nan)
	alert = np.full(N, np.nan, dtype=object)

	dates = pd.to_datetime(df[t]).dt.normalize().to_numpy(dtype="datetime64[ns]")
	y_obs = df[y].astype(float).to_numpy()

	for i in range(min_obs - 1, N):
		current_date = pd.Timestamp(dates[i])

		ref_dates = np.array(
			[current_date - pd.DateOffset(years=year_back) for year_back in range(0, B + 1)],
			dtype="datetime64[ns]",
		)

		wday_gaps = _weekday_sunday_zero(ref_dates) - int((current_date.dayofweek + 1) % 7)
		ref_dates_shifted = ref_dates - wday_gaps.astype("timedelta64[D]")

		floor_ceiling_dates = np.where(
			ref_dates_shifted > ref_dates,
			ref_dates_shifted - np.timedelta64(7, "D"),
			ref_dates_shifted + np.timedelta64(7, "D"),
		)

		center_dates = np.sort(
			np.where(
				np.abs(ref_dates - floor_ceiling_dates) < np.abs(ref_dates - ref_dates_shifted),
				floor_ceiling_dates,
				ref_dates_shifted,
			)
		)

		base_start = np.sort(center_dates - np.timedelta64(7 * w, "D"))[:B]
		base_end = np.concatenate(
			[
				np.sort(center_dates - np.timedelta64(7 * B, "D"))[1:B],
				np.array([center_dates.max() - np.timedelta64(7 * g, "D")]),
			]
		)

		base_dates_range = pd.date_range(
			start=pd.Timestamp(base_start.min()),
			end=pd.Timestamp(base_end.max()),
			freq="7D",
		).to_numpy(dtype="datetime64[ns]")

		base_length = len(base_dates_range)
		base_weeks = np.flatnonzero(np.isin(dates, center_dates)) + 1

		fct_levels = seasonal_groups(B=B, g=g, w=w, p=p, base_length=base_length, base_weeks=base_weeks)

		idx = np.flatnonzero(np.isin(dates, base_dates_range))
		if len(idx) == 0:
			continue

		if len(fct_levels) != len(idx):
			min_len = min(len(fct_levels), len(idx))
			idx = idx[:min_len]
			fct_levels = fct_levels[:min_len]

		min_date = dates[idx].min()
		base_dates_num = ((dates[idx] - min_date) / np.timedelta64(7, "D")).astype(float)
		base_counts = y_obs[idx]

		X_time, levels = _build_modified_design(base_dates_num, fct_levels, include_time=True)
		X_no_time, _ = _build_modified_design(base_dates_num, fct_levels, include_time=False)

		mod = _fit_glm_poisson(base_counts, X_time)
		include_time = bool(mod is not None and getattr(mod, "converged", False))
		X_current = X_time

		if not include_time:
			mod = _fit_glm_poisson(base_counts, X_no_time)
			include_time = False
			X_current = X_no_time

		if mod is None or not getattr(mod, "converged", False):
			continue

		phi = max(_dispersion(mod), 1.0)

		if include_time:
			time_coeff = float(mod.params.get("base_dates", np.nan))
			time_p_val = _quasi_p_value(mod, "base_dates", phi)
		else:
			time_coeff = np.nan
			time_p_val = np.nan

		omega = _compute_weights(mod, phi, threshold=2.58)

		mod_weighted = _fit_glm_poisson(base_counts, X_current, weights=omega)
		if mod_weighted is None or not getattr(mod_weighted, "converged", False):
			continue

		phi_weighted = max(_dispersion(mod_weighted), 1.0)
		time_pval_weighted = _quasi_p_value(mod_weighted, "base_dates", phi_weighted)

		pred_week_time = float((dates[i] - min_date) / np.timedelta64(7, "D"))

		pred_row_current = _build_modified_prediction_row(
			pred_week_time=pred_week_time,
			levels=levels,
			p_value=p,
			include_time=("base_dates" in X_current.columns),
		)

		pred_response, _ = _predict_glm(
			mod_weighted,
			pred_row_current,
			prediction_type="response",
		)

		time_significant = time_pval_weighted < 0.05
		pred_ok = pred_response <= np.nanmax(base_counts)
		trend = include_time and pred_ok

		if not trend:
			mod = _fit_glm_poisson(base_counts, X_no_time)
			if mod is None or not getattr(mod, "converged", False):
				continue

			phi = max(_dispersion(mod), 1.0)
			omega = _compute_weights(mod, phi, threshold=2.58)

			mod_weighted = _fit_glm_poisson(base_counts, X_no_time, weights=omega)
			if mod_weighted is None or not getattr(mod_weighted, "converged", False):
				continue

			phi_weighted = max(_dispersion(mod_weighted), 1.0)

			pred_row_no_time = _build_modified_prediction_row(
				pred_week_time=pred_week_time,
				levels=levels,
				p_value=p,
				include_time=False,
			)

			eta, _ = _predict_glm(
				mod_weighted,
				pred_row_no_time,
				prediction_type="link",
			)

			include_time_term[i] = False
		else:
			eta, _ = _predict_glm(
				mod_weighted,
				pred_row_current,
				prediction_type="link",
			)

			include_time_term[i] = True

		predicted[i] = eta
		time_coefficient[i] = time_coeff
		include_time_term[i] = True

		mu_q = math.exp(eta)

		if np.isinf(mu_q):
			upper[i] = np.nan
		elif phi_weighted > 1:
			size = mu_q / (phi_weighted - 1)
			prob = 1 / phi_weighted
			upper[i] = nbinom.ppf(0.95, size, prob)
		else:
			upper[i] = poisson.ppf(0.95, mu_q)

		if np.isfinite(upper[i]):
			alert_score[i] = (y_obs[i] - predicted[i]) / (upper[i] - predicted[i])
		else:
			alert_score[i] = np.nan

		recent_counts = np.sum(y_obs[(i - 4) : (i + 1)])
		alert[i] = "red" if (alert_score[i] > 1 and recent_counts > 5) else "blue"
		upper[i] = upper[i] if recent_counts > 5 else np.nan
		predicted[i] = math.exp(predicted[i])

	return pd.DataFrame(
		{
			"predicted": predicted,
			"time_coefficient": time_coefficient,
			"include_time_term": include_time_term,
			"upper": upper,
			"alert_score": alert_score,
			"alert": alert,
		}
	)


def alert_farrington(df, t="date", y="count", B=4, g=27, w=3, p=10, method="original"):
	"""Run Farrington temporal detection on weekly count time series.

	The Farrington algorithm family is intended for multi-year weekly count
	series. The original method fits quasi-Poisson models on reference windows
	from prior years. The modified method extends baseline coverage, uses
	seasonal grouping, and computes thresholds via count-distribution quantiles.

	:param df: Input dataframe or grouped dataframe.
	:param t: Name of the date column.
	:param y: Name of the count column.
	:param B: Number of years to include in the baseline.
	:param g: Guardband weeks separating baseline and test week.
	:param w: Half the number of weeks in each reference window.
	:param p: Number of seasonal periods in the baseline.
	:param method: Farrington method to run, ``"original"`` or ``"modified"``.
	:returns: A pandas data frame with Farrington detection outputs.
	:examples:

		>>> from pynssp import alert_farrington
		>>> import pandas as pd
		>>> import numpy as np
		>>>
		>>> df = pd.DataFrame({
		...     "date": pd.date_range("2014-01-05", "2022-02-05", freq="W-SUN"),
		...     "count": np.random.poisson(
		...         lam=25,
		...         size=len(pd.date_range("2014-01-05", "2022-02-05", freq="W-SUN")),
		...     ),
		... })
		>>>
		>>> df_farr_original = alert_farrington(df, t="date", y="count")
		>>> df_farr_original.head()
	"""
	grouped_df = isinstance(df, pd.core.groupby.DataFrameGroupBy)
	if not isinstance(df, pd.DataFrame) and not grouped_df:
		raise ValueError("Argument `df` must be a dataframe")

	if B < 4:
		raise ValueError(
			"Baseline length argument `B` must be greater than or equal to 4. "
			"Farrington algorithm requires a baseline of four or more years."
		)

	if g < 0:
		raise ValueError("Error in alert_farrington: guardband length argument `g` cannot be negative")

	if w < 0:
		raise ValueError("Half-week baseline argument `w` cannot be negative")

	if p < 2:
		raise ValueError("seasonal periods baseline argument `p` cannot be less than 2")

	base_df = df.obj if grouped_df else df

	if len(base_df) < 52 * B + w + 2:
		raise ValueError("Not enough historical data to form baseline")

	date_check = base_df[t]

	if not grouped_df and date_check.nunique(dropna=False) != len(date_check):
		raise ValueError("Duplicate dates detected. Please group your dataframe!")

	parsed_dates = _to_date_series(date_check)

	unique_dates = parsed_dates.drop_duplicates()
	if len(unique_dates) > 1:
		h_dates = unique_dates.diff().dropna().dt.days.to_numpy()
		if len(h_dates) == 0 or np.any(h_dates != 7):
			raise ValueError("Distance between dates is not 7 days. Counts must be weekly!")

	def _run_single(frame):
		local = frame.copy().reset_index(drop=True)
		local[t] = _to_date_series(local[t])

		if method == "modified":
			anomalies = farrington_modified(local, t=t, y=y, B=B, g=g, w=w, p=p)
		elif method == "original":
			anomalies = farrington_original(local, t=t, y=y, B=B, w=w)
		else:
			raise ValueError("Argument `method` must be `original` or `modified`.")

		out = pd.concat([local.reset_index(drop=True), anomalies.reset_index(drop=True)], axis=1)
		out["alert"] = out["alert"].where(out["alert"].notna(), "grey")
		return out

	if grouped_df:
		group_cols = [df.keys] if isinstance(df.keys, str) else list(df.keys)
		pieces = [_run_single(group) for _, group in df]
		alert_tbl = pd.concat(pieces, ignore_index=True)
		ordered = [col for col in group_cols if col in alert_tbl.columns]
		ordered += [col for col in alert_tbl.columns if col not in ordered]
		alert_tbl = alert_tbl.loc[:, ordered]
	else:
		alert_tbl = _run_single(base_df)

	return alert_tbl
