# forecast_from_9m_to_3m.py
# Predict next 3 months from previous 9 months using training mean curve
# Author: ChatGPT (adapt for your project)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------- USER SETTINGS ----------------
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"  # <<--- change this
date_col = "Date"           # date column name
flow_col = "Flow_cms"       # daily flow (m3/s) column name

ref_month = 9               # reference date month (e.g., 9 for Sept)
ref_day = 15                # reference date day
months_before = 9           # use previous 9 months as "observed"
months_after = 3            # predict next 3 months
validation_fraction = 0.10  # fraction of years held out for validation
random_seed = 42

# ---------------- MODEL DEFINITIONS ----------------
def logistic(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def power_law(x, a, b):
    # ensure positive x for power-law; we will fit over the full days range, may need to shift
    return a * np.power(x, b)

def polynomial2(x, a, b, c):
    return a * x**2 + b * x + c

def polynomial3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

models = {
    "Logistic": (logistic, [0.03, 10]),         # starting guesses
    "Exponential": (exponential, [1e-3, 0.01, 0]),
    "Quadratic": (polynomial2, [1e-6, 1e-3, 1e-3]),
    "Cubic": (polynomial3, [1e-9, 1e-6, 1e-3, 1e-3])
}
# (Power law omitted because domain around zero and negative days cause issues; can be added with shifting.)

# ---------------- UTILITIES ----------------
def extract_windows(df, years, ref_month, ref_day, months_before, months_after, flow_col):
    """Extract windows for given years: returns DataFrame with Year, Days_Relative, CumVol, CumVol_ref, CumVol_rel"""
    rows = []
    for y in years:
        ref_date = pd.Timestamp(y, ref_month, ref_day)
        start = ref_date - pd.DateOffset(months=months_before)
        end = ref_date + pd.DateOffset(months=months_after)
        if start < df.index.min() or end > df.index.max():
            # skip years where full window isn't available
            continue
        w = df.loc[start:end].copy()
        if w.empty:
            continue
        w["Year"] = y
        w["Days_Relative"] = (w.index - ref_date).days
        w["Volume_m3"] = w[flow_col] * 24 * 3600
        w["CumVol"] = w["Volume_m3"].cumsum()
        # cumulative at reference date (day 0)
        try:
            cum_at_ref = w.loc[w["Days_Relative"] == 0, "CumVol"].iloc[0]
        except Exception:
            # if day 0 missing (rare), get nearest day (shouldn't happen if daily)
            cum_at_ref = w["CumVol"].iloc[(w["Days_Relative"]-0).abs().argmin()]
        if cum_at_ref == 0:
            # avoid division by zero; skip this year
            continue
        w["CumVol_ref"] = cum_at_ref
        # relative cumulative: 1.0 at day 0
        w["CumVol_rel"] = w["CumVol"] / cum_at_ref
        rows.append(w[["Year", "Days_Relative", "CumVol", "CumVol_ref", "CumVol_rel"]])
    if not rows:
        return pd.DataFrame(columns=["Year", "Days_Relative", "CumVol", "CumVol_ref", "CumVol_rel"])
    return pd.concat(rows, ignore_index=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col])
df = df.set_index(date_col).sort_index()
if df.index.tzinfo is not None:
    df.index = df.index.tz_localize(None)

# Ensure flow column exists
if flow_col not in df.columns:
    raise ValueError(f"flow column '{flow_col}' not found in CSV")

# ---------------- CREATE YEAR LIST & SPLIT ----------------
years_all = np.array(sorted(df.index.year.unique()))
np.random.seed(random_seed)
n_val = max(1, int(len(years_all) * validation_fraction))
val_years = np.random.choice(years_all, size=n_val, replace=False)
train_years = np.array([y for y in years_all if y not in val_years])

print(f"Total years: {len(years_all)} | Train: {len(train_years)} | Val: {len(val_years)}")
print("Validation years:", sorted(val_years))

# ---------------- EXTRACT WINDOWS ----------------
train_windows = extract_windows(df, train_years, ref_month, ref_day, months_before, months_after, flow_col)
val_windows = extract_windows(df, val_years, ref_month, ref_day, months_before, months_after, flow_col)

if train_windows.empty:
    raise RuntimeError("No training windows extracted — check date coverage / settings.")
if val_windows.empty:
    raise RuntimeError("No validation windows extracted — check date coverage / settings.")

# ---------------- BUILD TRAINING MEAN RELATIVE CURVE ----------------
train_mean = train_windows.groupby("Days_Relative")["CumVol_rel"].mean().reset_index().dropna()
x_train = train_mean["Days_Relative"].values
y_train = train_mean["CumVol_rel"].values

# ---------------- FIT MODELS ON TRAINING MEAN ----------------
fit_results = []
for name, (func, p0) in models.items():
    try:
        # curve_fit can be sensitive; allow many iterations
        popt, _ = curve_fit(func, x_train, y_train, p0=p0, maxfev=20000)
        y_fit_train = func(x_train, *popt)
        r2_tr = r2_score(y_train, y_fit_train)
        rmse_tr = np.sqrt(mean_squared_error(y_train, y_fit_train))
        fit_results.append((name, func, popt, r2_tr, rmse_tr))
        print(f"{name:10s} | Train R²={r2_tr:.4f} | RMSE={rmse_tr:.4f}")
    except Exception as e:
        print(f"{name:10s} | fit failed: {e}")

if not fit_results:
    raise RuntimeError("No models converged on the training data.")

# Choose best model by highest training R²
best_name, best_func, best_params, best_r2, best_rmse = max(fit_results, key=lambda t: t[3])
print(f"\nSelected best model: {best_name} (train R²={best_r2:.4f})")
print("Parameters:", best_params)

# ---------------- VALIDATION: predict next N days for each validation year ----------------
val_years_used = sorted(val_windows["Year"].unique())
validation_records = []

for y in val_years_used:
    w = val_windows[val_windows["Year"] == y].copy().sort_values("Days_Relative")
    # split observed pre (days <= 0) and actual post (days > 0)
    observed_pre = w[w["Days_Relative"] <= 0].copy()
    actual_post = w[w["Days_Relative"] > 0].copy()
    if observed_pre.empty or actual_post.empty:
        # skip if missing
        continue

    # observed cumulative at ref (should be CumVol_ref)
    cum_ref = observed_pre["CumVol_ref"].iloc[0]

    # days to predict (post-date days relative)
    x_post = actual_post["Days_Relative"].values

    # Predict relative cumulative for post days using trained model
    y_rel_pred_post = best_func(x_post, *best_params)

    # Scale to absolute predicted cumulative: Cum_pred = y_rel_pred * Cum_at_ref
    cum_pred_post = y_rel_pred_post * cum_ref

    # Build arrays aligned for comparison
    cum_true_post = actual_post["CumVol"].values

    # Compute metrics comparing predicted cumulative vs actual cumulative over post period
    rmse = np.sqrt(mean_squared_error(cum_true_post, cum_pred_post))
    mae = mean_absolute_error(cum_true_post, cum_pred_post)
    # r2 on cumulative values
    r2 = r2_score(cum_true_post, cum_pred_post)

    # Also compute metrics on incremental flows (post increments): predicted increment = cum_pred - cum_ref
    inc_pred = cum_pred_post - cum_ref
    inc_true = cum_true_post - cum_ref
    rmse_inc = np.sqrt(mean_squared_error(inc_true, inc_pred))
    mae_inc = mean_absolute_error(inc_true, inc_pred)
    r2_inc = r2_score(inc_true, inc_pred) if len(inc_true) > 1 else np.nan

    validation_records.append({
        "Year": y,
        "RMSE_cum": rmse,
        "MAE_cum": mae,
        "R2_cum": r2,
        "RMSE_inc": rmse_inc,
        "MAE_inc": mae_inc,
        "R2_inc": r2_inc,
        "n_post_days": len(x_post)
    })

    # PLOT per-year: observed pre, predicted post, actual post
    plt.figure(figsize=(8,4))
    # observed pre (plot cumulative)
    plt.plot(observed_pre["Days_Relative"], observed_pre["CumVol"], label=f"Observed pre ({y})", color="tab:blue")
    # predicted post cumulative (line)
    plt.plot(x_post, cum_pred_post, label=f"Predicted post ({best_name})", color="tab:green", linestyle="--")
    # actual post cumulative
    plt.plot(x_post, cum_true_post, label=f"Actual post ({y})", color="tab:orange", linestyle="-")
    plt.axvline(0, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Days relative to reference date")
    plt.ylabel("Cumulative volume (m³)")
    plt.title(f"{y} — Predicted vs Actual post-date cumulative (scaled from observed pre-date)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------- SUMMARY ----------------
val_df = pd.DataFrame(validation_records)
if val_df.empty:
    print("No validation years were processed (check date coverage).")
else:
    print("\nValidation summary (per-year):")
    print(val_df.round(3))
    print("\nMean validation metrics (cumulative):")
    print("Mean RMSE_cum:", val_df["RMSE_cum"].mean())
    print("Mean MAE_cum:", val_df["MAE_cum"].mean())
    print("Mean R2_cum:", val_df["R2_cum"].mean())
    print("\nMean validation metrics (increments):")
    print("Mean RMSE_inc:", val_df["RMSE_inc"].mean())
    print("Mean MAE_inc:", val_df["MAE_inc"].mean())
    print("Mean R2_inc:", val_df["R2_inc"].mean())

    # save results
    val_df.to_csv("validation_results_9m_to_3m.csv", index=False)
    print("\nSaved validation results to 'validation_results_9m_to_3m.csv'")

# ---------------- OPTIONAL: plot training mean and fitted relative curve ----------------
plt.figure(figsize=(9,4))
plt.plot(train_mean["Days_Relative"], train_mean["CumVol_rel"], label="Training mean (relative)", color="C0")
x_fit = np.linspace(train_mean["Days_Relative"].min(), train_mean["Days_Relative"].max(), 400)
plt.plot(x_fit, best_func(x_fit, *best_params), label=f"{best_name} fit", color="C2", linestyle="--")
plt.axvline(0, color="gray", linestyle=":")
plt.xlabel("Days relative to reference date")
plt.ylabel("Cumulative (relative to cum@ref = 1.0)")
plt.title("Training mean relative curve and fitted model")
plt.legend()
plt.tight_layout()
plt.show()