import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------- USER SETTINGS ----------------
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"   # change me
date_col = "Date"
flow_col = "Flow_cms"

ref_month = 9
ref_day = 15
months_before = 9       # used to compute observed cum@ref but not required here
months_after = 3        # forecast horizon (3 months)
validation_fraction = 0.1
random_seed = 42

# ---------------- LOAD DATA ----------------
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
df = df.set_index(date_col).sort_index()
df["Volume_m3"] = df[flow_col] * 24 * 3600
years_all = np.array(sorted(df.index.year.unique()))

# ---------------- SELECT TRAIN / VALIDATION YEARS ----------------
np.random.seed(random_seed)
n_val = max(1, int(len(years_all) * validation_fraction))
val_years = np.random.choice(years_all, size=n_val, replace=False)
train_years = np.array([y for y in years_all if y not in val_years])
print("Train years:", len(train_years), "Validation years:", len(val_years))
print("Validation years sample:", sorted(val_years))

# ---------------- BUILD HISTORICAL POST-DATE INCREMENTAL CURVES ----------------
def build_post_increment_curves(df, years, ref_month, ref_day, months_after):
    rows = []
    for y in years:
        ref_date = pd.Timestamp(y, ref_month, ref_day)
        start_ref = ref_date
        end_ref = ref_date + pd.DateOffset(months=months_after)
        if start_ref < df.index.min() or end_ref > df.index.max():
            continue
        # cumulative up to ref (inclusive)
        year_df = df.loc[ref_date - pd.DateOffset(months=months_before): end_ref]  # ensure cum exists
        # compute per-year cumulative from Jan1 (or start of window) to index
        # But easier: compute cumulative in absolute m3 across entire slice then compute increments relative to cum@ref
        slice_df = df.loc[ref_date - pd.DateOffset(months=months_before): end_ref].copy()
        if slice_df.empty:
            continue
        slice_df["CumVol"] = slice_df["Volume_m3"].cumsum()
        # get cumulative at ref
        # if exact ref date missing, pick nearest earlier or equal
        try:
            cum_at_ref = slice_df.loc[slice_df.index <= ref_date, "CumVol"].iloc[-1]
        except IndexError:
            continue
        # post period rows (ref_date < t <= end_ref), we want day 0..N inclusive starting at ref_date
        post_df = slice_df.loc[slice_df.index > ref_date].copy()
        if post_df.empty:
            continue
        days_after = (post_df.index - ref_date).days.values
        inc_after = post_df["CumVol"].values - cum_at_ref  # incremental cumulative after ref
        # store in rows with year, days_after, inc_after
        rows.append(pd.DataFrame({"Year": y, "DayAfter": days_after, "IncAfterRef": inc_after}))
    if not rows:
        return pd.DataFrame(columns=["Year","DayAfter","IncAfterRef"])
    return pd.concat(rows, ignore_index=True)

# Build historical post-date increment curves using training years only
hist_post = build_post_increment_curves(df, train_years, ref_month, ref_day, months_after)
if hist_post.empty:
    raise RuntimeError("No historical post-date curves (train) extracted. Check date coverage and settings.")

# ---------------- INTERPOLATE TO COMMON GRID ----------------
# Determine days_remaining by looking at validation reference (assume same across years)
# Use target days length = ceil(max days in hist_post)
max_day = int(hist_post["DayAfter"].max())
days_common = np.arange(1, max_day+1)  # day 1..N (post-date days)

# pivot: for each year, interpolate its increments onto days_common
interp_list = []
for y in hist_post["Year"].unique():
    sub = hist_post[hist_post["Year"]==y].sort_values("DayAfter")
    # ensure unique increasing day indices
    da = sub["DayAfter"].values
    inc = sub["IncAfterRef"].values
    if len(da) < 2:
        continue
    # interpolation; days_common might extend beyond some years -> np.interp will use edge values
    interp_inc = np.interp(days_common, da, inc, left=np.nan, right=np.nan)
    interp_list.append(pd.Series(interp_inc, name=str(y)))
if len(interp_list) == 0:
    raise RuntimeError("No usable historical curves for interpolation (need at least two points per historical post-year).")

interp_df = pd.concat(interp_list, axis=1)  # columns = years, rows = days_common

# Optionally drop columns with NaNs at the final day (incomplete post period)
interp_df = interp_df.loc[:, interp_df.iloc[-1].notna()]

if interp_df.shape[1] == 0:
    raise RuntimeError("No historical post curves cover the full forecast horizon. Try reducing months_after or checking data coverage.")

# ---------------- COMPUTE MEDIAN + PERCENTILES ----------------
median_inc = interp_df.median(axis=1).values        # median incremental cumulative after ref per day
p10_inc = interp_df.quantile(0.10, axis=1).values
p90_inc = interp_df.quantile(0.90, axis=1).values
# wettest = max final increment; dry = min final increment
final_increments = interp_df.iloc[-1, :]
wet_year_col = final_increments.idxmax()
dry_year_col = final_increments.idxmin()
wet_inc = interp_df[wet_year_col].values
dry_inc = interp_df[dry_year_col].values

# ---------------- FORECAST FOR EACH VALIDATION YEAR ----------------
results = []
for vy in sorted(val_years):
    # get cum@ref for validation year
    ref_date_val = pd.Timestamp(vy, ref_month, ref_day)
    # ensure we have data earlier to get cumulative
    # take slice from start of file up to ref_date_val
    if ref_date_val not in df.index:
        # find nearest <= ref_date_val
        prior_idx = df.index[df.index <= ref_date_val]
        if len(prior_idx) == 0:
            print(f"Skipping {vy}: no pre-ref data")
            continue
        nearest_ref = prior_idx[-1]
    else:
        nearest_ref = ref_date_val
    # build cumulative across whole file up to nearest_ref
    # compute CumVol from beginning of file (this was used in hist_post too)
    # easier: make cum series once (cache)
    # compute once outside loop (below) — but for clarity do here:
    cum_series = df["Volume_m3"].cumsum()
    cum_at_ref = cum_series.loc[nearest_ref]
    # actual post data for vy
    end_ref_val = ref_date_val + pd.DateOffset(months=months_after)
    if end_ref_val > df.index.max():
        print(f"Skipping {vy}: incomplete post period")
        continue
    actual_post_df = df.loc[ref_date_val + pd.Timedelta(days=1): end_ref_val].copy()
    if actual_post_df.empty:
        print(f"Skipping {vy}: no actual post rows")
        continue
    days_post_true = (actual_post_df.index - ref_date_val).days.values
    cum_true_post = actual_post_df["Volume_m3"].cumsum().values - cum_at_ref  # increments after ref
    # ensure we have same length as days_common; we'll align to days_common[0:len(cum_true_post)]
    n_post = len(cum_true_post)
    if n_post > len(days_common):
        # truncate to days_common length if dataset has more days (rare)
        cum_true_post = cum_true_post[:len(days_common)]
        n_post = len(cum_true_post)
    # Build forecasts (median, p10/p90, wet/dry) for the appropriate length
    med_forecast_inc = median_inc[:n_post]
    p10_forecast_inc = p10_inc[:n_post]
    p90_forecast_inc = p90_inc[:n_post]
    wet_forecast_inc = wet_inc[:n_post]
    dry_forecast_inc = dry_inc[:n_post]
    # scale to absolute cumulative by adding cum_at_ref
    med_forecast_cum = cum_at_ref + med_forecast_inc
    p10_forecast_cum = cum_at_ref + p10_forecast_inc
    p90_forecast_cum = cum_at_ref + p90_forecast_inc
    wet_forecast_cum = cum_at_ref + wet_forecast_inc
    dry_forecast_cum = cum_at_ref + dry_forecast_inc
    true_post_cum = cum_at_ref + cum_true_post
    # Evaluate median forecast vs true (on cumulative or increments)
    rmse_cum = np.sqrt(mean_squared_error(true_post_cum, med_forecast_cum))
    mae_cum = mean_absolute_error(true_post_cum, med_forecast_cum)
    r2_cum = r2_score(true_post_cum, med_forecast_cum) if len(true_post_cum)>1 else np.nan
    # also on increments
    rmse_inc = np.sqrt(mean_squared_error(cum_true_post, med_forecast_inc))
    mae_inc = mean_absolute_error(cum_true_post, med_forecast_inc)
    r2_inc = r2_score(cum_true_post, med_forecast_inc) if len(cum_true_post)>1 else np.nan
    results.append({
        "Year": vy,
        "n_post_days": n_post,
        "RMSE_cum_med": rmse_cum,
        "MAE_cum_med": mae_cum,
        "R2_cum_med": r2_cum,
        "RMSE_inc_med": rmse_inc,
        "MAE_inc_med": mae_inc,
        "R2_inc_med": r2_inc
    })
    # PLOT: observed YTD cumulative up to ref, then median/p10/p90/wet/dry continuations and actual
    dates_future = actual_post_df.index
    # observed YTD cumulative series
    cum_series_full = df["Volume_m3"].cumsum()
    ytd_dates = df.loc[:nearest_ref].index
    ytd_cum = cum_series_full.loc[ytd_dates]
    plt.figure(figsize=(8,4))
    plt.plot(ytd_dates, ytd_cum, label=f"{vy} observed YTD", color="tab:blue")
    plt.plot(dates_future, med_forecast_cum, label="Median continuation", color="green", linestyle="--")
    plt.fill_between(dates_future, p10_forecast_cum, p90_forecast_cum, color="green", alpha=0.2, label="10-90% band")
    plt.plot(dates_future, wet_forecast_cum, label="Wettest historical continuation", color="purple", linestyle=":")
    plt.plot(dates_future, dry_forecast_cum, label="Driest historical continuation", color="orange", linestyle=":")
    plt.plot(dates_future, true_post_cum, label="Actual continuation", color="tab:gray", linewidth=2)
    plt.axvline(nearest_ref, color="black", linestyle=":")
    plt.xlabel("Date")
    plt.ylabel("Cumulative volume (m³)")
    plt.title(f"Year {vy}: YTD + Median / Wet / Dry continuations")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------- SUMMARY ----------------
res_df = pd.DataFrame(results)
if res_df.empty:
    print("No validation results computed.")
else:
    print("\nValidation summary (median continuation):")
    print(res_df.round(3))
    print("\nAverage metrics (median continuation):")
    print(res_df[["RMSE_cum_med","MAE_cum_med","R2_cum_med","RMSE_inc_med","MAE_inc_med","R2_inc_med"]].mean())
    res_df.to_csv("validation_median_continuation_results.csv", index=False)
    print("Saved validation results to 'validation_median_continuation_results.csv'")