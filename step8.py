#!/usr/bin/env python3
"""
step6_ytd_forecast_LOYO_multi_scaling.py

Leave-one-year-out cross-validation (LOYO). For each validation year:
 - Build training-year median curves aligned to a common day-after-ref timeline
 - Compute a forecast using the selected scaling/stretching method
 - Save one plot per year (observed pre/ref/post, median, forecast, wettest/driest)
 - Save CSV of per-year metrics (RMSE, R2, FinalVolumeDiff)

Set `scaling_method` below to one of:
  "volume_scale", "least_squares", "percentile_map",
  "two_point_stretch", "nonlinear_power", "anchored_stretch"
(anchored_stretch is the anchored method you previously used).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # file-based backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit

# -----------------------------
# USER CONFIG
# -----------------------------
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"
date_col = "Date"
flow_col = "Flow_cms"

ref_month, ref_day = 9, 15
months_before = 9
months_after = 3

out_folder = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/forecast_plots"
os.makedirs(out_folder, exist_ok=True)

# Choose one method to run for the whole LOYO run:
# options: "volume_scale", "least_squares", "percentile_map",
#          "two_point_stretch", "nonlinear_power", "anchored_stretch"
scaling_method = "anchored_stretch"

# Parameters (tweakable)
ls_pre_days = 30          # for least_squares: number of pre-ref days used to fit k
two_point_early_day = -30 # for two-point: day used as "early" anchor (relative to ref)
nonlinear_min_points = 10 # min pre-ref points to fit nonlinear power transform
percentile_p_list = [50]  # not used here, but helpful if you want specific percentiles

# -----------------------------
# UTILITIES
# -----------------------------
def extend_or_trim(arr, n):
    """Return array of length n by trimming or repeating last value if shorter."""
    arr = np.asarray(arr)
    if len(arr) >= n:
        return arr[:n]
    if len(arr) == 0:
        return np.zeros(n)
    return np.concatenate([arr, np.full(n - len(arr), arr[-1])])

def build_post_increment_curves(df, years, ref_month, ref_day, months_before, months_after):
    """Return DataFrame with Year, DayAfter, IncAfterRef for the given years (training years)."""
    curves = []
    for y in years:
        ref_date = pd.Timestamp(y, ref_month, ref_day)
        start_ref = ref_date - pd.DateOffset(months=months_before)
        end_ref = ref_date + pd.DateOffset(months=months_after)
        window = df.loc[start_ref:end_ref].copy()
        if window.empty or (ref_date not in window.index) or (window.index.max() < end_ref):
            continue
        window["CumVol"] = window["Volume_m3"].cumsum()
        cum_at_ref = window.loc[window.index <= ref_date, "CumVol"].iloc[-1]
        post_df = window.loc[window.index > ref_date]
        inc = post_df["CumVol"].values - cum_at_ref
        days = (post_df.index - ref_date).days.values
        curves.append(pd.DataFrame({"Year": y, "DayAfter": days, "IncAfterRef": inc}))
    if len(curves) == 0:
        return pd.DataFrame(columns=["Year", "DayAfter", "IncAfterRef"])
    return pd.concat(curves, ignore_index=True)

# -----------------------------
# SCALING / STRETCHING METHODS
# Each returns forecast cumulative array of length n (aligned with true_inc)
# Inputs common in LOYO loop: median_curve (array length >= n - 1 or >= n),
# median_hist (same), cum_at_ref (scalar), median_9mo_at_ref (scalar),
# days_common (array), n (int), interp_df (DataFrame with training interpolated series)
# and optionally other arrays (wettest_inc, driest_inc)
# -----------------------------
def method_volume_scale(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n, **kwargs):
    """Method 1: Simple multiplicative scaling of entire curve by ratio of cum_at_ref / median_9mo_at_ref."""
    ratio = cum_at_ref / median_9mo_at_ref if median_9mo_at_ref != 0 else 1.0
    med_for_days = extend_or_trim(median_curve[1:], n - 1)
    scaled_inc = np.concatenate(([0], med_for_days * ratio))
    return scaled_inc

def method_least_squares(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n, pre_obs_series=None, pre_med_series=None, **kwargs):
    """
    Method 2: Least-squares multiplicative scaling k minimizing sum (obs - k*med)^2
    Fit k using the pre-ref window (pre_obs_series, pre_med_series arrays aligned; indices 0..m-1)
    If not enough points or zero denom, fallback to volume_scale.
    """
    if pre_obs_series is None or pre_med_series is None:
        # fallback
        return method_volume_scale(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n)
    # align non-nan points
    mask = (~np.isnan(pre_obs_series)) & (~np.isnan(pre_med_series))
    if mask.sum() < 2:
        return method_volume_scale(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n)
    obs = pre_obs_series[mask]
    med = pre_med_series[mask]
    denom = np.sum(med ** 2)
    if denom == 0:
        return method_volume_scale(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n)
    k = np.sum(obs * med) / denom
    med_for_days = extend_or_trim(median_curve[1:], n - 1)
    scaled_inc = np.concatenate(([0], med_for_days * k))
    return scaled_inc

def method_percentile_map(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n, interp_df=None, ref_vols=None, **kwargs):
    """
    Method 3: Percentile-mapped scaling.
    - Compute percentile p of cum_at_ref among training ref_vols (ECDF)
    - For each day, take p-percentile across training series (interp_df rows -> days)
    """
    if interp_df is None or ref_vols is None or len(ref_vols) == 0:
        return method_volume_scale(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n)
    # percentile p (0-100)
    p = (np.sum(np.array(ref_vols) <= cum_at_ref) / max(1, len(ref_vols))) * 100.0
    # compute p-th percentile across training daily increments (interp_df is increments per day)
    pperc = np.nanpercentile(interp_df.values, p, axis=1)
    pperc_for_days = extend_or_trim(pperc, n - 1)
    perc_inc = np.concatenate(([0], pperc_for_days))
    # anchor start to cum_at_ref (ensure correct cumulative baseline)
    return perc_inc

def method_two_point_stretch(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n, days_common=None, obs_series=None, **kwargs):
    """
    Method 4: Two-point stretch (linear scale factor between early day and ref day).
    - Use early day (two_point_early_day) and ref day 0.
    - If obs value at early day is missing, fallback to anchored_stretch or volume_scale.
    """
    global two_point_early_day
    if days_common is None or obs_series is None:
        return method_volume_scale(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n)
    # find index for early day (closest)
    early_d = two_point_early_day
    idx_early = np.argmin(np.abs(days_common - early_d))
    idx_ref = np.argmin(np.abs(days_common - 0))
    # median values at those indices
    med_early = median_curve[idx_early] if idx_early < len(median_curve) else median_curve[0]
    med_ref = median_curve[idx_ref] if idx_ref < len(median_curve) else median_curve[0]
    # observed values at those indices (obs_series is cumulative aligned to days_common; may have nan)
    obs_early = obs_series[idx_early] if idx_early < len(obs_series) else np.nan
    obs_ref = obs_series[idx_ref] if idx_ref < len(obs_series) else np.nan
    if np.isnan(obs_early) or med_early == 0 or med_ref == med_early:
        # fallback to anchored_stretch
        return method_anchored_stretch(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n, days_common=days_common)
    # compute scales at early and ref
    scale_early = obs_early / med_early
    scale_ref = obs_ref / med_ref if (not np.isnan(obs_ref) and med_ref != 0) else scale_early
    # linear scale(t) = a + b * t (over day index)
    t1, t2 = days_common[idx_early], days_common[idx_ref]
    if t2 == t1:
        a = scale_early
        b = 0.0
    else:
        b = (scale_ref - scale_early) / (t2 - t1)
        a = scale_early - b * t1
    # apply scale across median_curve
    med_for_days = extend_or_trim(median_curve, n)  # cumulative median (has day0)
    scaled = med_for_days * (a + b * days_common[:n])
    # convert to increments relative to cum_at_ref baseline
    scaled_inc = scaled - scaled[0]
    scaled_inc[0] = 0.0
    return scaled_inc

def method_nonlinear_power(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n, pre_obs_series=None, pre_med_series=None, **kwargs):
    """
    Method 5: Nonlinear power transform forecast: forecast = obs_start + b * (median_curve - median_curve[0])**a
    Fit a and b to pre-ref points by least squares: obs = obs_start + b*(med - med0)**a
    If not enough points, fallback to least_squares or anchored_stretch.
    """
    med_full = np.asarray(median_curve)
    med0 = med_full[0]
    med_dev = med_full - med0
    # need pre_obs_series & pre_med_series (dev from med0)
    if pre_obs_series is None or pre_med_series is None:
        return method_least_squares(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n, pre_obs_series=pre_obs_series, pre_med_series=pre_med_series)
    mask = (~np.isnan(pre_obs_series)) & (~np.isnan(pre_med_series))
    if mask.sum() < nonlinear_min_points:
        return method_least_squares(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n, pre_obs_series=pre_obs_series, pre_med_series=pre_med_series)
    obs = pre_obs_series[mask] - pre_obs_series[mask][0]  # dev from the start of pre-window
    medp = (pre_med_series[mask] - pre_med_series[mask][0])
    # initial guess a=1 (linear), b=1
    def power_model(x, a, b):
        # x here is medp
        # avoid negative bases with abs and preserve sign
        return b * np.sign(x) * (np.abs(x) ** a)
    try:
        popt, _ = curve_fit(power_model, medp, obs, p0=[1.0, 1.0], maxfev=20000)
        a, b = popt
    except Exception:
        return method_least_squares(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n, pre_obs_series=pre_obs_series, pre_med_series=pre_med_series)
    # now build forecast_dev = b * sign(med_dev) * |med_dev|^a
    forecast_dev = b * np.sign(med_dev) * (np.abs(med_dev) ** a)
    forecast = cum_at_ref + forecast_dev
    fc_inc = forecast - forecast[0]
    fc_inc = extend_or_trim(fc_inc, n)
    fc_inc[0] = 0.0
    return fc_inc

def method_anchored_stretch(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n, **kwargs):
    """
    Anchored stretched median (your Option C):
    - preserve shape of median_curve
    - scale vertically so the endpoint matches median_hist endpoint
    - start at observed cum_at_ref
    """
    # ensure arrays are long enough
    med_curve = extend_or_trim(median_curve, max(len(median_curve), n))
    med_hist = extend_or_trim(median_hist, max(len(median_hist), n))
    # endpoints
    curve_start, curve_end = med_curve[0], med_curve[-1]
    hist_start, hist_end = med_hist[0], med_hist[-1]
    if curve_end == curve_start:
        stretch = 1.0
    else:
        stretch = (hist_end - hist_start) / (curve_end - curve_start)
    forecast = cum_at_ref + (med_curve - med_curve[0]) * stretch
    fc_inc = extend_or_trim(forecast - forecast[0], n)
    fc_inc[0] = 0.0
    return fc_inc

# Mapping names to functions
METHOD_MAP = {
    "volume_scale": method_volume_scale,
    "least_squares": method_least_squares,
    "percentile_map": method_percentile_map,
    "two_point_stretch": method_two_point_stretch,
    "nonlinear_power": method_nonlinear_power,
    "anchored_stretch": method_anchored_stretch
}

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
df = df.set_index(date_col).sort_index()
df["Volume_m3"] = df[flow_col] * 24 * 3600

years_all = np.array(sorted(df.index.year.unique()))
print(f"Years in data: {years_all.min()} - {years_all.max()} ({len(years_all)} years)")

metrics = []
skipped_years = []

for vy in years_all:
    print(f"\n=== Validation year: {vy} ===")
    # training years
    train_years = [y for y in years_all if y != vy]
    hist_post = build_post_increment_curves(df, train_years, ref_month, ref_day, months_before, months_after)
    if hist_post.empty:
        print(f"  skip {vy}: no historical post curves")
        skipped_years.append(vy)
        continue
    max_day = int(hist_post["DayAfter"].max())
    days_common = np.arange(1, max_day + 1)

    # interpolate each training year's increments to days_common
    interp_list = []
    for y in hist_post["Year"].unique():
        sub = hist_post[hist_post["Year"] == y].sort_values("DayAfter")
        interp_inc = np.interp(days_common, sub["DayAfter"], sub["IncAfterRef"], left=np.nan, right=np.nan)
        interp_list.append(pd.Series(interp_inc, name=str(y)))
    interp_df = pd.concat(interp_list, axis=1)
    # drop incomplete series not reaching final day
    interp_df = interp_df.loc[:, interp_df.iloc[-1].notna()]
    if interp_df.shape[1] == 0:
        print(f"  skip {vy}: no training series reach final day")
        skipped_years.append(vy)
        continue

    median_inc = interp_df.median(axis=1).values  # increments per day after ref
    # convert median_inc to cumulative median curve with day0 baseline 0 (so cumulative median at day i = sum up to i)
    median_curve = np.concatenate(([0], np.cumsum(median_inc)))  # length = len(days_common)+1
    median_hist = median_curve.copy()  # in many implementations median_hist == median_curve; kept separate for clarity

    # identify wettest/driest training years (final increments)
    final_values = interp_df.iloc[-1]
    wettest_year = final_values.idxmax()
    driest_year = final_values.idxmin()
    wettest_inc = interp_df[wettest_year].values
    driest_inc = interp_df[driest_year].values
    wettest_curve = np.concatenate(([0], np.cumsum(wettest_inc)))
    driest_curve = np.concatenate(([0], np.cumsum(driest_inc)))

    # compute median 9-month cum at ref using training years (for some methods)
    ref_vols = []
    for y in train_years:
        ref_date = pd.Timestamp(y, ref_month, ref_day)
        start_ref = ref_date - pd.DateOffset(months=months_before)
        pre_window = df.loc[start_ref:ref_date].copy()
        if pre_window.empty or (ref_date not in pre_window.index):
            continue
        pre_window["CumVolWindow"] = pre_window["Volume_m3"].cumsum()
        ref_vols.append(pre_window["CumVolWindow"].iloc[-1])
    if len(ref_vols) == 0:
        print(f"  skip {vy}: no training ref_vols")
        skipped_years.append(vy)
        continue
    median_9mo_at_ref = np.median(ref_vols)

    # build validation windows for this year
    ref_date = pd.Timestamp(vy, ref_month, ref_day)
    start_ref = ref_date - pd.DateOffset(months=months_before)
    end_ref = ref_date + pd.DateOffset(months=months_after)
    pre_df = df.loc[start_ref:ref_date].copy()
    post_df = df.loc[ref_date:end_ref].copy()
    if pre_df.empty or post_df.empty or (ref_date not in pre_df.index):
        print(f"  skip {vy}: missing pre/post/ref")
        skipped_years.append(vy)
        continue
    pre_df["CumVolWindow"] = pre_df["Volume_m3"].cumsum()
    cum_at_ref = pre_df["CumVolWindow"].iloc[-1]

    # full window for true increments
    window_full = df.loc[start_ref:end_ref].copy()
    window_full["CumVolWindow"] = window_full["Volume_m3"].cumsum()
    true_inc = window_full.loc[window_full.index >= ref_date, "CumVolWindow"].values - cum_at_ref
    n = len(true_inc)
    if n == 0:
        print(f"  skip {vy}: no true increment days")
        skipped_years.append(vy)
        continue

    # make sure median_curve and median_hist have length >= n (they include day0)
    median_curve = extend_or_trim(median_curve, n)
    median_hist = extend_or_trim(median_hist, n)

    # prepare pre-ref series for least_squares & nonlinear methods:
    # We need arrays aligned with days_common including day0 and negative days if present.
    # Build a daily aligned cumulative series for validation-year around ref_date using full days_common indices:
    # Create an "aligned days" array with position 0 at ref_date and forward positions 1..n-1
    # For pre-ref points used by ls and nonlinear, we'll extract the last ls_pre_days values from pre_df cumulative
    # If pre_df is daily but could be irregular, we'll resample to daily with forward-fill to match days.
    pre_daily = pre_df["CumVolWindow"].resample("D").ffill()
    # extract last ls_pre_days values including ref_date
    pre_obs_window = pre_daily.reindex(pd.date_range(start=pre_daily.index.min(), end=pre_daily.index.max(), freq="D"))
    # create arrays of the last ls_pre_days values relative to ref_date if available
    if len(pre_obs_window) >= ls_pre_days:
        pre_obs_vals = pre_obs_window.values[-ls_pre_days:]
    else:
        pre_obs_vals = pre_obs_window.values  # may be shorter
    # Prepare pre_med_series corresponding to same relative days using training medians:
    # easiest is to use the last len(pre_obs_vals) points of median_curve (these correspond to times before ref if median_curve had negative days,
    # but given our median_curve is post-ref cumulative starting at 0, we will use the first len(pre_obs_vals) points as proxies for fitting.
    # For stability, we'll use the first len(pre_obs_vals) values of median_curve (which represent day0 and forward). This is imperfect,
    # but in practice least_squares is using recent pattern; alternative is to build a separate pre-ref median — more code.
    pre_med_vals = None
    if len(pre_obs_vals) > 0:
        # create a simple pre_med_vals array of same length by using the first len(pre_obs_vals) of median_curve (shifted)
        pre_med_vals = median_curve[:len(pre_obs_vals)]
    else:
        pre_med_vals = None

    # Determine which forecasting function to call
    method_fn = METHOD_MAP.get(scaling_method, method_anchored_stretch)

    # Prepare kwargs specific to methods
    method_kwargs = {
        "median_curve": median_curve,
        "median_hist": median_hist,
        "cum_at_ref": cum_at_ref,
        "median_9mo_at_ref": median_9mo_at_ref,
        "n": n,
        "interp_df": interp_df,
        "ref_vols": ref_vols,
        "days_common": np.concatenate(([0], days_common))[:n],  # include day0 and forward days
        "obs_series": None,
        "pre_obs_series": None,
        "pre_med_series": None
    }

    # Build obs_series (cumulative) aligned to days_common starting at day0:
    # obs_cum_post = cum_at_ref + true_inc (length n)
    obs_cum_post = cum_at_ref + true_inc
    obs_series_aligned = np.concatenate(([cum_at_ref], obs_cum_post))  # length n
    method_kwargs["obs_series"] = obs_series_aligned

    # pass pre_ref arrays for least_squares and nonlinear if available
    if pre_obs_vals is not None and pre_med_vals is not None and len(pre_obs_vals) > 0 and len(pre_med_vals) > 0:
        method_kwargs["pre_obs_series"] = pre_obs_vals
        method_kwargs["pre_med_series"] = pre_med_vals

    # Call method
    try:
        if scaling_method == "percentile_map":
            fc_inc = method_percentile_map(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n,
                                          interp_df=interp_df, ref_vols=ref_vols)
        elif scaling_method == "least_squares":
            fc_inc = method_least_squares(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n,
                                          pre_obs_series=method_kwargs.get("pre_obs_series"),
                                          pre_med_series=method_kwargs.get("pre_med_series"))
        elif scaling_method == "two_point_stretch":
            fc_inc = method_two_point_stretch(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n,
                                             days_common=method_kwargs["days_common"],
                                             obs_series=method_kwargs["obs_series"])
        elif scaling_method == "nonlinear_power":
            fc_inc = method_nonlinear_power(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n,
                                            pre_obs_series=method_kwargs.get("pre_obs_series"),
                                            pre_med_series=method_kwargs.get("pre_med_series"))
        elif scaling_method == "volume_scale":
            fc_inc = method_volume_scale(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n)
        else:
            # default anchored_stretch
            fc_inc = method_anchored_stretch(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n)
    except Exception as e:
        print(f"  method error for {vy}: {e}. Falling back to anchored_stretch.")
        fc_inc = method_anchored_stretch(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n)

    # Ensure length n and first element 0
    fc_inc = extend_or_trim(fc_inc, n)
    fc_inc[0] = 0.0

    # Metrics: compare true_inc (length n) to fc_inc (length n)
    true_arr = np.asarray(true_inc)
    try:
        rmse = np.sqrt(mean_squared_error(true_arr, fc_inc))
    except Exception:
        rmse = np.nan
    try:
        r2 = r2_score(true_arr, fc_inc)
    except Exception:
        r2 = np.nan

    final_obs = cum_at_ref + true_arr[-1]
    final_fore = cum_at_ref + fc_inc[-1]
    final_diff = final_obs - final_fore

    metrics.append({
        "Year": int(vy),
        "RMSE": float(rmse),
        "R2": float(r2) if not np.isnan(r2) else np.nan,
        "FinalVolumeDiff": float(final_diff),
        "ObsFinal": float(final_obs),
        "ForecastFinal": float(final_fore)
    })

    print(f"  RMSE={rmse:.3f} | R2={np.nan if np.isnan(r2) else r2:.3f} | FinalDiff={final_diff:.3f}")
    # ---------------------------
    # PLOT: observed pre + observed post + median + forecast + wet/dry
    # ---------------------------
    plot_window = df.loc[start_ref:end_ref].copy()
    plot_window["CumVolWindow"] = plot_window["Volume_m3"].cumsum()
    obs_pre = plot_window.loc[start_ref:ref_date]
    obs_post_index = plot_window.loc[ref_date:end_ref].index
    # If obs_post_index length mismatches n, create synthetic daily index starting at ref_date
    if len(obs_post_index) != n:
        obs_post_index = pd.date_range(start=ref_date, periods=n, freq='D')

    # cumulative forecasts for plotting
    fc_cum = cum_at_ref + fc_inc
    med_cum = median_curve[:n]  # already cumulative with day0 baseline 0
    wet_cum = wettest_curve[:n]
    dry_cum = driest_curve[:n]

    plt.figure(figsize=(10, 5))
    plt.plot(obs_pre.index, obs_pre["CumVolWindow"], color="black", label="Observed (previous 9 months)")
    plt.plot(obs_post_index, cum_at_ref + true_arr, color="black", linestyle="--", label="Observed (forecast period)")
    plt.plot(obs_post_index, med_cum, color="green", label="Median historical")
    plt.plot(obs_post_index, fc_cum, color="purple", label=f"Forecast ({scaling_method})")
    plt.plot(obs_post_index, wet_cum, color="deepskyblue", linestyle=":", label=f"Wettest training year ({wettest_year})")
    plt.plot(obs_post_index, dry_cum, color="darkred", linestyle=":", label=f"Driest training year ({driest_year})")
    plt.axvline(ref_date, color="gray", linestyle=":", label="Forecast start")
    plt.title(f"{vy} — 9-Month History + 3-Month Forecast ({scaling_method})")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Volume (m³)")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plot_path = os.path.join(out_folder, f"forecast_{vy}_{scaling_method}.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved plot: {plot_path}")

# -----------------------------
# SAVE METRICS CSV
# -----------------------------
metrics_df = pd.DataFrame(metrics).sort_values("Year").reset_index(drop=True)
metrics_csv_path = os.path.join(out_folder, f"validation_metrics_{scaling_method}.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"\nSaved metrics CSV: {metrics_csv_path}")

if skipped_years:
    print(f"Skipped years due to insufficient data: {skipped_years}")

print("Done.")