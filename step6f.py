# step6_ytd_forecast_with_extremes_noblend.py
# Predict 3-month cumulative volume forecast using 9 months of observed data,
# comparing median, logistic, scaled, stretched, and wettest/driest forecasts.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================================
# SETTINGS
# ==========================================================
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"   # update path
date_col = "Date"
flow_col = "Flow_cms"
ref_month, ref_day = 9, 15
months_before = 9
months_after = 3
validation_fraction = 0.1
random_seed = 42

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
df = df.set_index(date_col).sort_index()
df["Volume_m3"] = df[flow_col] * 24 * 3600

years_all = np.array(sorted(df.index.year.unique()))
np.random.seed(random_seed)
n_val = max(1, int(len(years_all) * validation_fraction))
val_years = np.random.choice(years_all, size=n_val, replace=False)
train_years = np.array([y for y in years_all if y not in val_years])

print(f"Training years: {len(train_years)} | Validation years: {len(val_years)}")

# ==========================================================
# BUILD HISTORICAL INCREMENT CURVES AFTER REF DATE
# ==========================================================
def build_post_increment_curves(df, years, ref_month, ref_day, months_after):
    curves = []
    for y in years:
        ref_date = pd.Timestamp(y, ref_month, ref_day)
        end_ref = ref_date + pd.DateOffset(months=months_after)
        sub = df.loc[str(y)].copy()
        if ref_date not in sub.index or end_ref > sub.index.max():
            continue
        sub["CumVol"] = sub["Volume_m3"].cumsum()
        cum_at_ref = sub.loc[sub.index <= ref_date, "CumVol"].iloc[-1]
        post_df = sub.loc[sub.index > ref_date]
        inc = post_df["CumVol"].values - cum_at_ref
        days = (post_df.index - ref_date).days.values
        curves.append(pd.DataFrame({"Year": y, "DayAfter": days, "IncAfterRef": inc}))
    return pd.concat(curves, ignore_index=True)

hist_post = build_post_increment_curves(df, train_years, ref_month, ref_day, months_after)
max_day = int(hist_post["DayAfter"].max())
days_common = np.arange(1, max_day + 1)

# ==========================================================
# INTERPOLATE HISTORICAL CURVES AND COMPUTE MEDIAN
# ==========================================================
interp_list = []
for y in hist_post["Year"].unique():
    sub = hist_post[hist_post["Year"] == y].sort_values("DayAfter")
    interp_inc = np.interp(days_common, sub["DayAfter"], sub["IncAfterRef"], left=np.nan, right=np.nan)
    interp_list.append(pd.Series(interp_inc, name=str(y)))
interp_df = pd.concat(interp_list, axis=1)
interp_df = interp_df.loc[:, interp_df.iloc[-1].notna()]
median_inc = interp_df.median(axis=1).values

# Identify wettest and driest years
final_values = interp_df.iloc[-1]
wettest_year = final_values.idxmax()
driest_year = final_values.idxmin()
wettest_inc = interp_df[wettest_year].values
driest_inc = interp_df[driest_year].values
print(f"Wettest year: {wettest_year} | Driest year: {driest_year}")

# ==========================================================
# FIT LOGISTIC CURVE TO MEDIAN INCREMENT
# ==========================================================
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

try:
    popt, _ = curve_fit(logistic, days_common, median_inc, p0=[median_inc[-1], 0.05, 0], maxfev=10000)
    logistic_inc = logistic(days_common, *popt)
    print("Logistic fit parameters:", popt)
except RuntimeError:
    print("⚠️ Logistic fit failed — reverting to median only.")
    logistic_inc = median_inc.copy()
    popt = [np.nan, np.nan, np.nan]

# ==========================================================
# VALIDATION AND VISUALIZATION
# ==========================================================
results = []

# Compute median 9-month cumulative volume at ref date (for scaling)
ref_vols = []
for y in train_years:
    ref_date = pd.Timestamp(y, ref_month, ref_day)
    start_ref = ref_date - pd.DateOffset(months=months_before)
    sub = df.loc[str(y)].copy()
    if ref_date not in sub.index or start_ref < sub.index.min():
        continue
    sub["CumVol"] = sub["Volume_m3"].cumsum()
    ref_vol = sub.loc[sub.index <= ref_date, "CumVol"].iloc[-1]
    ref_vols.append(ref_vol)
median_9mo_at_ref = np.median(ref_vols)

for vy in sorted(val_years):
    ref_date = pd.Timestamp(vy, ref_month, ref_day)
    start_ref = ref_date - pd.DateOffset(months=months_before)
    end_ref = ref_date + pd.DateOffset(months=months_after)
    sub = df.loc[str(vy)].copy()
    if len(sub) == 0:
        continue
    sub["CumVol"] = sub["Volume_m3"].cumsum()
    pre_df = sub.loc[start_ref:ref_date].copy()
    post_df = sub.loc[ref_date:end_ref].copy()
    if pre_df.empty or post_df.empty:
        continue

    cum_at_ref = pre_df["CumVol"].iloc[-1]
    true_inc = post_df["CumVol"].values - cum_at_ref
    n = len(true_inc)

    # Forecasts
    med_fore = median_inc[:n]
    log_fore = logistic_inc[:n]
    wet_fore = wettest_inc[:n]
    dry_fore = driest_inc[:n]

    # Shape-adjusted forecasts
    ratio = cum_at_ref / median_9mo_at_ref  # relative wetness/dryness
    scaled_fore = median_inc[:n] * ratio
    stretched_days = days_common / ratio
    stretched_fore = np.interp(days_common[:n], stretched_days, median_inc, left=np.nan, right=np.nan)

    # Metrics for main forecasts
    rmse_med = np.sqrt(mean_squared_error(true_inc, med_fore))
    rmse_log = np.sqrt(mean_squared_error(true_inc, log_fore))
    r2_med = r2_score(true_inc, med_fore)
    r2_log = r2_score(true_inc, log_fore)
    results.append([vy, rmse_med, rmse_log, r2_med, r2_log])

    # === Visualization (9 months observed + 3 months forecast) ===
    plt.figure(figsize=(10, 5))
    plt.plot(pre_df.index, pre_df["CumVol"], color="black", label="Observed (previous 9 months)")
    plt.plot(post_df.index, cum_at_ref + true_inc, color="black", linestyle="--", label="Observed (forecast period)")
    plt.plot(post_df.index, cum_at_ref + med_fore, color="green", label="Median forecast")
    plt.plot(post_df.index, cum_at_ref + log_fore, color="orange", label="Logistic forecast")
    plt.plot(post_df.index, cum_at_ref + scaled_fore, color="purple", label="Scaled median")
    plt.plot(post_df.index, cum_at_ref + stretched_fore, color="brown", label="Stretched median")
    plt.plot(post_df.index, cum_at_ref + wet_fore, color="deepskyblue", linestyle=":", label=f"Wettest year ({wettest_year})")
    plt.plot(post_df.index, cum_at_ref + dry_fore, color="darkred", linestyle=":", label=f"Driest year ({driest_year})")
    plt.axvline(ref_date, color="gray", linestyle=":", label="Forecast start")
    plt.title(f"{vy} — 9-Month History + 3-Month Forecast")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Volume (m³)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==========================================================
# SUMMARY
# ==========================================================
res_df = pd.DataFrame(results, columns=["Year", "RMSE_Median", "RMSE_Logistic", "R2_Median", "R2_Logistic"])
print("\nValidation summary:")
print(res_df.round(3))
print("\nAverage metrics:")
print(res_df[["RMSE_Median", "RMSE_Logistic", "R2_Median", "R2_Logistic"]].mean())