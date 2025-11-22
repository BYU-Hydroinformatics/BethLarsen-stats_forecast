import os
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for file-saving (no GUI required)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================================
# SETTINGS (update csv_path if needed)
# ==========================================================
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"
date_col = "Date"
flow_col = "Flow_cms"
ref_month, ref_day = 9, 15
months_before = 9
months_after = 3

# Output folder for plots & csv
out_folder = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/forecast_plots_stretch"
os.makedirs(out_folder, exist_ok=True)
metrics_csv_path = os.path.join(out_folder, "validation_metrics_stretch.csv")

# ==========================================================
# USER-TUNABLE
# ==========================================================
min_pre_windows_for_hist = 3  # minimum number of training-year pre-windows to build median_hist (tuneable)

# ==========================================================
# UTILITIES
# ==========================================================
def build_post_increment_curves(df, years, ref_month, ref_day, months_before, months_after):
    """
    Build DataFrame of post-reference cumulative increments for given years.
    Returns columns: Year, DayAfter, IncAfterRef
    """
    curves = []
    for y in years:
        ref_date = pd.Timestamp(y, ref_month, ref_day)
        start_ref = ref_date - pd.DateOffset(months=months_before)
        end_ref = ref_date + pd.DateOffset(months=months_after)

        # continuous slice across year boundaries
        window = df.loc[start_ref:end_ref].copy()
        # require ref_date present and we have full window to end_ref
        if window.empty or (ref_date not in window.index) or (window.index.max() < end_ref):
            continue

        window["CumVol"] = window["Volume_m3"].cumsum()
        # cumulative value at ref_date (last value <= ref_date)
        cum_at_ref = window.loc[window.index <= ref_date, "CumVol"].iloc[-1]
        post_df = window.loc[window.index > ref_date]
        inc = post_df["CumVol"].values - cum_at_ref
        days = (post_df.index - ref_date).days.values
        curves.append(pd.DataFrame({"Year": y, "DayAfter": days, "IncAfterRef": inc}))
    if len(curves) == 0:
        return pd.DataFrame(columns=["Year", "DayAfter", "IncAfterRef"])
    return pd.concat(curves, ignore_index=True)

def extend_or_trim(arr, n):
    """Return array of length n by trimming or repeating last value if shorter."""
    arr = np.asarray(arr)
    if len(arr) >= n:
        return arr[:n]
    if len(arr) == 0:
        return np.zeros(n)
    # repeat last value to match length
    return np.concatenate([arr, np.full(n - len(arr), arr[-1])])

def method_anchored_stretch(median_curve, median_hist, cum_at_ref, median_9mo_at_ref, n, **kwargs):
    """
    Anchored stretched median:
    - preserve shape of median_curve (post-ref)
    - scale vertically so the endpoint matches median_hist endpoint (pre-window endpoint)
    - start at observed cum_at_ref
    Returns increments (length n, with first element 0).
    """
    # ensure arrays are long enough
    med_curve = extend_or_trim(np.asarray(median_curve, dtype=float), max(len(median_curve), n))
    med_hist = extend_or_trim(np.asarray(median_hist, dtype=float), max(len(median_hist), n))
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

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
df = df.set_index(date_col).sort_index()
df["Volume_m3"] = df[flow_col] * 24 * 3600

# all unique years present in the index (sorted)
years_all = np.array(sorted(df.index.year.unique()))
print(f"Found years: {years_all.min()} - {years_all.max()} (n={len(years_all)})")

# ==========================================================
# MAIN LOYO LOOP
# ==========================================================
metrics = []  # will hold [Year, RMSE_Anchored, R2_Anchored, FinalVolumeDiff_Anchored]
skipped_years = []

for vy in years_all:
    print(f"\n--- Validation year: {vy} ---")
    # training years: all except validation year
    train_years = [y for y in years_all if y != vy]

    # Build historical post-ref increment curves from training years
    hist_post = build_post_increment_curves(df, train_years, ref_month, ref_day, months_before, months_after)
    if hist_post.empty:
        print(f"Skipping {vy}: no historical post-ref curves built from training years.")
        skipped_years.append(vy)
        continue

    max_day = int(hist_post["DayAfter"].max())
    days_common = np.arange(1, max_day + 1)

    # Interpolate each year's curve to common days and build matrix
    interp_list = []
    for y in hist_post["Year"].unique():
        sub = hist_post[hist_post["Year"] == y].sort_values("DayAfter")
        interp_inc = np.interp(days_common, sub["DayAfter"], sub["IncAfterRef"], left=np.nan, right=np.nan)
        interp_list.append(pd.Series(interp_inc, name=str(y)))
    interp_df = pd.concat(interp_list, axis=1)

    # drop any series that don't reach the final day (incomplete windows)
    interp_df = interp_df.loc[:, interp_df.iloc[-1].notna()]

    if interp_df.shape[1] == 0:
        print(f"Skipping {vy}: no training series reach the full post-ref window.")
        skipped_years.append(vy)
        continue

    # median increment (per day after ref) across training years
    median_inc = interp_df.median(axis=1).values  # length = len(days_common)

    # identify wettest/driest training years for plotting
    final_values = interp_df.iloc[-1]
    wettest_year = final_values.idxmax()
    driest_year = final_values.idxmin()
    wettest_inc = interp_df[wettest_year].values
    driest_inc = interp_df[driest_year].values

    # compute median 9-month cumulative volume at ref date using training years (for reference)
    ref_vols = []
    for y in train_years:
        ref_date = pd.Timestamp(y, ref_month, ref_day)
        start_ref = ref_date - pd.DateOffset(months=months_before)
        pre_window = df.loc[start_ref:ref_date].copy()
        if pre_window.empty or (ref_date not in pre_window.index):
            continue
        pre_window["CumVolWindow"] = pre_window["Volume_m3"].cumsum()
        ref_vol = pre_window["CumVolWindow"].iloc[-1]
        ref_vols.append(ref_vol)
    if len(ref_vols) == 0:
        print(f"Skipping {vy}: no reference volumes available from training years to compute median 9-mo at ref.")
        skipped_years.append(vy)
        continue
    median_9mo_at_ref = np.median(ref_vols)

    # Build validation (true) windows for this validation year (allow cross-year)
    ref_date = pd.Timestamp(vy, ref_month, ref_day)
    start_ref = ref_date - pd.DateOffset(months=months_before)
    end_ref = ref_date + pd.DateOffset(months=months_after)

    pre_df = df.loc[start_ref:ref_date].copy()
    post_df = df.loc[ref_date:end_ref].copy()
    # require pre and post presence and ref_date present in pre_df
    if pre_df.empty or post_df.empty or (ref_date not in pre_df.index):
        print(f"Skipping {vy}: missing pre or post data for validation year.")
        skipped_years.append(vy)
        continue

    # compute cum_at_ref for validation year
    pre_df["CumVolWindow"] = pre_df["Volume_m3"].cumsum()
    cum_at_ref = pre_df["CumVolWindow"].iloc[-1]

    # build full window to compute true increments consistently
    window_full = df.loc[start_ref:end_ref].copy()
    window_full["CumVolWindow"] = window_full["Volume_m3"].cumsum()
    true_inc = window_full.loc[window_full.index >= ref_date, "CumVolWindow"].values - cum_at_ref
    n = len(true_inc)
    if n == 0:
        print(f"Skipping {vy}: no true increment days found.")
        skipped_years.append(vy)
        continue

    # Prepare median_inc (extend/trim to at least n-1 days for post forecasts)
    med_for_days = extend_or_trim(median_inc, n - 1)  # length n-1
    wet_for_days = extend_or_trim(wettest_inc, n - 1)
    dry_for_days = extend_or_trim(driest_inc, n - 1)

    # compute ratio = current 9-mo total / median 9-mo total (relative wetness)
    ratio = cum_at_ref / median_9mo_at_ref if median_9mo_at_ref != 0 else 1.0
    # scaled median forecast increments (day0=0, then increments)
    scaled_fore_inc = np.concatenate(([0], med_for_days * ratio))  # length n

    # -------------------------
    # Build median_hist from training years' pre-windows (H1)
    # -------------------------
    pre_median_list = []
    for y in train_years:
        y_ref = pd.Timestamp(y, ref_month, ref_day)
        y_start = y_ref - pd.DateOffset(months=months_before)
        y_pre = df.loc[y_start:y_ref].copy()
        if y_pre.empty or (y_ref not in y_pre.index):
            continue
        y_pre = y_pre.sort_index()
        y_pre["CumVolWindow"] = y_pre["Volume_m3"].cumsum()
        pre_vals = y_pre["CumVolWindow"].values
        pre_median_list.append(pre_vals)

    if len(pre_median_list) < min_pre_windows_for_hist:
        # Insufficient training pre-windows to build a robust median histogram; fallback to scaled median
        print(f"Year {vy}: only {len(pre_median_list)} training pre-windows available (<{min_pre_windows_for_hist}). Using scaled median fallback.")
        use_anchored = False
    else:
        use_anchored = True

    median_hist = None
    if use_anchored:
        # align by trimming to shortest (so ends line up)
        min_len_pre = min(len(arr) for arr in pre_median_list)
        trimmed_pre = np.array([arr[-min_len_pre:] for arr in pre_median_list])
        # median_hist is cumulative pre-window aligned to the last min_len_pre days (start -> ref)
        median_hist = np.median(trimmed_pre, axis=0)

        # Note: median_hist[0] corresponds to an earlier time (start_ref + offset),
        # median_hist[-1] is the typical cumulative at ref_date

    # -------------------------
    # Build median_curve_full (post-ref cumulative starting at 0)
    # -------------------------
    median_curve_full = np.concatenate(([0], med_for_days)).astype(float)  # length n

    # -------------------------
    # Compute anchored-stretch forecast (primary) or fallback
    # -------------------------
    if use_anchored and (median_hist is not None):
        try:
            anchored_inc = method_anchored_stretch(
                median_curve=median_curve_full,
                median_hist=median_hist,
                cum_at_ref=cum_at_ref,
                median_9mo_at_ref=median_9mo_at_ref,
                n=n
            )
            print(f"Year {vy}: anchored-stretch forecast built (median_hist length={len(median_hist)}).")
        except Exception as e:
            print(f"Year {vy}: anchored-stretch failed ({e}). Using scaled median fallback.")
            anchored_inc = scaled_fore_inc.copy()
    else:
        anchored_inc = scaled_fore_inc.copy()
        print(f"Year {vy}: using scaled median forecast (fallback).")

    # Ensure anchored_inc shape
    anchored_inc = extend_or_trim(anchored_inc, n)
    true_inc_arr = np.asarray(true_inc)

    # RMSE and R2 for anchored forecast
    rmse_anchor = np.sqrt(mean_squared_error(true_inc_arr, anchored_inc))
    try:
        r2_anchor = r2_score(true_inc_arr, anchored_inc)
    except Exception:
        r2_anchor = np.nan

    final_obs = cum_at_ref + true_inc_arr[-1]
    final_fore_anchor = cum_at_ref + anchored_inc[-1]
    final_diff_anchor = final_obs - final_fore_anchor

    metrics.append([vy, rmse_anchor, r2_anchor, final_diff_anchor])
    print(f"Year {vy} | RMSE_Anchored={rmse_anchor:.3f} | R2_Anchored={np.nan if pd.isna(r2_anchor) else r2_anchor:.3f} | FinalDiff_Anchored={final_diff_anchor:.3f}")

    # ==========================================================
    # PLOTTING (Observed history + observed future + forecasts)
    # ==========================================================
    # Build plotting window and series
    plot_window = df.loc[start_ref:end_ref].copy()
    plot_window["CumVolWindow"] = plot_window["Volume_m3"].cumsum()
    obs_pre = plot_window.loc[start_ref:ref_date]
    obs_post_index = plot_window.loc[ref_date:end_ref].index

    # cumulative series for plotting
    anchor_cum = cum_at_ref + anchored_inc
    scaled_cum = cum_at_ref + scaled_fore_inc

    # median and wet/dry cumulative series for plotting (prepend cum_at_ref as day0)
    med_inc_full = np.concatenate(([0], med_for_days))
    wet_inc_full = np.concatenate(([0], wet_for_days))
    dry_inc_full = np.concatenate(([0], dry_for_days))
    med_cum = cum_at_ref + med_inc_full
    wet_cum = cum_at_ref + wet_inc_full
    dry_cum = cum_at_ref + dry_inc_full

    # create date index for forecast days: obs_post_index length should equal n
    if len(obs_post_index) != n:
        obs_post_index = pd.date_range(start=ref_date, periods=n, freq='D')

    plt.figure(figsize=(10, 5))
    plt.plot(obs_pre.index, obs_pre["CumVolWindow"], color="black", label="Observed (previous 9 months)")
    plt.plot(obs_post_index, cum_at_ref + true_inc_arr, color="black", linestyle="--", label="Observed (forecast period)")

    # Plot median (unscaled) for reference
    plt.plot(obs_post_index, med_cum, color="green", label="Median forecast (unscaled)")

    # Plot anchored-stretch forecast (primary)
    plt.plot(obs_post_index, anchor_cum, color="purple", label="Anchored-stretch forecast")

    # Plot scaled median as faded comparison
    plt.plot(obs_post_index, scaled_cum, color="purple", alpha=0.4, linestyle="--", label="Scaled median (comparison)")

    plt.plot(obs_post_index, wet_cum, color="deepskyblue", linestyle=":", label=f"Wettest training year ({wettest_year})")
    plt.plot(obs_post_index, dry_cum, color="darkred", linestyle=":", label=f"Driest training year ({driest_year})")
    plt.axvline(ref_date, color="gray", linestyle=":", label="Forecast start")
    plt.title(f"{vy} — 9-Month History + 3-Month Forecast (Anchored Stretch)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Volume (m³)")
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(out_folder, f"forecast_{vy}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")

# ==========================================================
# SAVE METRICS CSV
# ==========================================================
metrics_df = pd.DataFrame(metrics, columns=["Year", "RMSE_Anchored", "R2_Anchored", "FinalVolumeDiff_Anchored"])
metrics_df = metrics_df.sort_values("Year").reset_index(drop=True)
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"\nSaved metrics CSV: {metrics_csv_path}")

if skipped_years:
    print(f"\nSkipped years (insufficient data or missing windows): {skipped_years}")

print("\nDone.")