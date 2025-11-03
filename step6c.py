import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- SETTINGS ---
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"
date_col = "Date"
flow_col = "Flow_cms"

ref_month, ref_day = 9, 15
months_before = 9
months_after = 3
validation_fraction = 0.1
random_seed = 42
blend_alpha = 0.3   # weight of logistic fit vs median (0 = median only, 1 = logistic only)

# --- LOAD DATA ---
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

# --- Helper: build historical post-date incremental curves ---
def build_post_increment_curves(df, years, ref_month, ref_day, months_after):
    curves = []
    for y in years:
        ref_date = pd.Timestamp(y, ref_month, ref_day)
        end_ref = ref_date + pd.DateOffset(months=months_after)
        sub = df.loc[str(y)]
        if ref_date not in sub.index or end_ref not in sub.index:
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

# interpolate all historical curves
interp_list = []
for y in hist_post["Year"].unique():
    sub = hist_post[hist_post["Year"] == y].sort_values("DayAfter")
    interp_inc = np.interp(days_common, sub["DayAfter"], sub["IncAfterRef"], left=np.nan, right=np.nan)
    interp_list.append(pd.Series(interp_inc, name=str(y)))
interp_df = pd.concat(interp_list, axis=1)
interp_df = interp_df.loc[:, interp_df.iloc[-1].notna()]

# --- Compute median curve ---
median_inc = interp_df.median(axis=1).values

# --- Fit logistic curve to median ---
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

p0 = [median_inc[-1], 0.05, 0]  # initial guess
try:
    popt, _ = curve_fit(logistic, days_common, median_inc, p0=p0, maxfev=10000)
    logistic_inc = logistic(days_common, *popt)
    print("Logistic fit parameters:", popt)
except RuntimeError:
    print("⚠️ Logistic fit failed — reverting to median only.")
    logistic_inc = median_inc.copy()
    popt = [np.nan, np.nan, np.nan]

# --- Blend curves ---
blend_inc = blend_alpha * logistic_inc + (1 - blend_alpha) * median_inc

# --- Validation ---
results = []
for vy in sorted(val_years):
    ref_date = pd.Timestamp(vy, ref_month, ref_day)
    end_ref = ref_date + pd.DateOffset(months=months_after)
    sub = df.loc[str(vy)]
    if ref_date not in sub.index or end_ref not in sub.index:
        continue

    sub["CumVol"] = sub["Volume_m3"].cumsum()
    cum_at_ref = sub.loc[sub.index <= ref_date, "CumVol"].iloc[-1]
    post_df = sub.loc[sub.index > ref_date].copy()
    true_inc = post_df["CumVol"].values - cum_at_ref
    n = len(true_inc)
    days = (post_df.index - ref_date).days.values

    # forecasts
    med_fore = median_inc[:n]
    log_fore = logistic_inc[:n]
    blend_fore = blend_inc[:n]

    # scores
    rmse_med = np.sqrt(mean_squared_error(true_inc, med_fore))
    rmse_log = np.sqrt(mean_squared_error(true_inc, log_fore))
    rmse_blend = np.sqrt(mean_squared_error(true_inc, blend_fore))
    r2_blend = r2_score(true_inc, blend_fore)
    results.append([vy, rmse_med, rmse_log, rmse_blend, r2_blend])

    # plot
    plt.figure(figsize=(8,4))
    plt.plot(days, true_inc, label="Actual", color="black")
    plt.plot(days, med_fore, label="Median", color="green")
    plt.plot(days, log_fore, label="Logistic", color="orange")
    plt.plot(days, blend_fore, label=f"Blend (α={blend_alpha})", color="blue", linestyle="--")
    plt.title(f"{vy} Validation — Median vs Logistic vs Blend")
    plt.xlabel("Days after reference date")
    plt.ylabel("Incremental cumulative volume (m³)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Summary ---
res_df = pd.DataFrame(results, columns=["Year", "RMSE_Median", "RMSE_Logistic", "RMSE_Blend", "R2_Blend"])
print("\nValidation summary:")
print(res_df.round(3))
print("\nAverage RMSE / R²:")
print(res_df[["RMSE_Median", "RMSE_Logistic", "RMSE_Blend", "R2_Blend"]].mean())