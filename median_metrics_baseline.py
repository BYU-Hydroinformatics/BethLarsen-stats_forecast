import os
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for file-saving (no GUI required)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit  # kept import though not used; safe to remove

# ==========================================================
# SETTINGS (update csv_path if needed)
# ==========================================================
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/Retrospective_Data/retrospective_ohio.csv"
date_col = "Date"
flow_col = "Discharge"
ref_month, ref_day = 1, 30
months_before = 9
months_after = 3

# Output folder for plots & csv
out_folder = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/median_plots_metrics/forecast_plots_median_ohio"
os.makedirs(out_folder, exist_ok=True)
metrics_csv_path = os.path.join(out_folder, "validation_metrics_ohio_median.csv")


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
        base_ref = pd.Timestamp(year=2000, month=ref_month, day=ref_day)
        ref_date = base_ref.replace(year=y)
        start_ref = ref_date - pd.DateOffset(months=months_before)
        end_ref = ref_date + pd.DateOffset(months=months_after)

        # continuous slice across year boundaries
        window = df.loc[start_ref:end_ref].copy()
        # require ref_date present and we have full window to end_ref
        if window.empty or (ref_date not in window.index):
            #or (window.index.max() < end_ref):
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
# MAIN LOYO LOOP (MEDIAN METHOD)
# ==========================================================
metrics = []  # will hold [Year, RMSE, NSE, Willmott_d1, R2, FinalVolumeDiff]
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
    #interp_df = interp_df.loc[:, interp_df.iloc[-1].notna()]

    #if interp_df.shape[1] == 0:
        #print(f"Skipping {vy}: no training series reach the full post-ref window.")
        #skipped_years.append(vy)
        #continue

    # median increment (per day after ref) across training years
    median_inc = interp_df.median(axis=1).values  # length = len(days_common)

    # identify wettest/driest training years for plotting
    final_values = interp_df.apply(lambda c: c.dropna().iloc[-1])
    wettest_year = final_values.idxmax()
    driest_year = final_values.idxmin()
    wettest_inc = interp_df[wettest_year].values
    driest_inc = interp_df[driest_year].values


    # Build validation (true) windows for this validation year (allow cross-year)
    ref_date = pd.Timestamp(vy, ref_month, ref_day)
    start_ref = ref_date - pd.DateOffset(months=months_before)
    end_ref = ref_date + pd.DateOffset(months=months_after)


    pre_df = df.loc[start_ref:ref_date].copy()
    post_df = df.loc[ref_date:end_ref].copy()
    ref_day_ts = ref_date.floor("D")
    pre_days = pre_df.index.floor("D")
    # require pre and post presence and ref_date present in pre_df
    #if pre_df.empty or post_df.empty or (ref_day_ts not in pre_days):
        #print(f"Skipping {vy}: missing pre or post data for validation year.")
        #skipped_years.append(vy)
        #continue
    if pre_df.empty:
        continue

    if len(post_df) < int(0.3 * months_after * 30):
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


    # Use median as the forecast (primary)
    forecast_inc = extend_or_trim(median_inc, n)
    forecast_inc[0] = 0.0

    # True increments array
    true_inc_arr = np.asarray(true_inc)

    # -----------------------------------------
    # METRICS FOR MEDIAN METHOD
    # -----------------------------------------
    y = true_inc_arr
    yhat = forecast_inc
    y_mean = np.mean(y)

    # RMSE
    rmse = np.sqrt(np.mean((y - yhat) ** 2))

    # --- NSE ---
    SSE = np.sum((y - yhat) ** 2)
    SST = np.sum((y - y_mean) ** 2)
    SSR = np.sum((yhat - y_mean) ** 2)
    nse = 1 - SSE / SST if SST != 0 else np.nan

    # --- Willmott d1 (modified index of agreement) ---
    numerator = np.sum(np.abs(y - yhat))
    denominator = np.sum(np.abs(yhat - y_mean) + np.abs(y - y_mean))
    d1 = 1 - numerator / denominator if denominator != 0 else np.nan

    # --- R² (traditional) ---
    r2 = SSR / SST if SST != 0 else np.nan

    final_obs = cum_at_ref + true_inc_arr[-1]
    three_month_obs = true_inc_arr[-1]
    final_fore = cum_at_ref + forecast_inc[-1]
    final_diff = final_obs - final_fore
    final_diff_pe = final_diff/ final_obs
    rmse_pe = rmse/final_obs

    metrics.append([vy, rmse, nse, d1, r2, final_diff, final_obs, final_diff_pe, rmse_pe, three_month_obs])

    print(f"Year {vy} | RMSE={rmse:.3f} | R2={np.nan if pd.isna(r2) else r2:.3f} | FinalDiff={final_diff:.3f} | FinalObs={final_obs}")

    # ==========================================================
    # PLOTTING (Observed history + observed future + forecasts)
    # ==========================================================
    # Build plotting window and series
    plot_window = df.loc[start_ref:end_ref].copy()
    plot_window["CumVolWindow"] = plot_window["Volume_m3"].cumsum()
    obs_pre = plot_window.loc[start_ref:ref_date]
    obs_post_index = plot_window.loc[ref_date:end_ref].index

    # cumulative series for plotting
    scaled_cum = cum_at_ref + forecast_inc

    # median and wet/dry cumulative series for plotting (prepend cum_at_ref as day0)
    med_inc_full = np.concatenate(([0], med_for_days))
    wet_inc_full = np.concatenate(([0], wet_for_days))
    dry_inc_full = np.concatenate(([0], dry_for_days))
    med_cum = cum_at_ref + med_inc_full
    wet_cum = cum_at_ref + wet_inc_full
    dry_cum = cum_at_ref + dry_inc_full

    # create date index for forecast days: obs_post_index length should equal n
    # If lengths mismatch, align by truncation/extension of index (rare)
    if len(obs_post_index) != n:
        # create artificial index by using ref_date + days (0..n-1)
        obs_post_index = pd.date_range(start=ref_date, periods=n, freq='D')

    # Combine pre and post for plotting observed lines
    plt.figure(figsize=(10, 5))
    plt.plot(obs_pre.index, obs_pre["CumVolWindow"], color="black", label="Observed (previous 9 months)")
    # observed post (actual cumulative during forecast period)
    plt.plot(obs_post_index, cum_at_ref + true_inc_arr, color="black", linestyle="--", label="Observed (forecast period)")

    # Plot median (unscaled) for reference
    plt.plot(obs_post_index, med_cum, label="Historical Median")


    # Plot wet/dry training years for context
    plt.plot(obs_post_index, wet_cum, linestyle=":", label=f"Wettest training year ({wettest_year})")
    plt.plot(obs_post_index, dry_cum, linestyle=":", label=f"Driest training year ({driest_year})")
    plt.axvline(ref_date, color="gray", linestyle=":", label="Forecast start")
    plt.title(f"{vy} — 9-Month History + 3-Month Median/Wettest/Driest)")
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
metrics_df = pd.DataFrame(
    metrics,
    columns=[
        "Year",
        "RMSE",
        "NSE",
        "Willmott_d1",
        "R2",
        "FinalVolumeDiff",
        "FinalVol",
        "FinalDiffPE",
        "RMSE_PE",
        "3 Month Obs"
    ]
)
metrics_df = metrics_df.sort_values("Year").reset_index(drop=True)
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"\nSaved metrics CSV: {metrics_csv_path}")

if skipped_years:
    print(f"\nSkipped years (insufficient data or missing windows): {skipped_years}")

print("\nDone.")
