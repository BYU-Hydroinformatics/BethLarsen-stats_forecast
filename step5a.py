import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

# ===============================================================
# SETTINGS
# ===============================================================
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"
date_col = "Date"
flow_col = "Flow_cms"
ref_month, ref_day = 3, 15     # target date (e.g., Sep 15)
window_days = 360              # ±6 months
val_fraction = 0.1             # 10% of years held out
np.random.seed(42)             # reproducible split

# ===============================================================
# LOAD AND PREP DATA
# ===============================================================
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
df = df.set_index(date_col).sort_index()
df["Volume_m3"] = df[flow_col] * 24 * 3600  # daily volume (m³)

years = df.index.year.unique()
val_years = np.random.choice(years, size=int(len(years)*val_fraction), replace=False)
train_years = [y for y in years if y not in val_years]
print(f"Training years: {len(train_years)}, Validation years: {len(val_years)}")

# ===============================================================
# FUNCTION TO EXTRACT 6-MONTH WINDOW AROUND TARGET DATE
# ===============================================================
def extract_window(df, years):
    curves = []
    for year in years:
        ref_date = pd.Timestamp(f"{year}-{ref_month:02d}-{ref_day:02d}")
        start = ref_date - pd.Timedelta(days=window_days)
        end = ref_date + pd.Timedelta(days=window_days)
        if start < df.index.min() or end > df.index.max():
            continue

        window = df.loc[start:end].copy()
        window["Days_Relative"] = (window.index - ref_date).days
        window["CumVol"] = window["Volume_m3"].cumsum()
        window["CumVol_norm"] = window["CumVol"] / window["CumVol"].iloc[-1]
        window["Year"] = year
        curves.append(window[["Days_Relative", "CumVol_norm", "Year"]])
    return pd.concat(curves, ignore_index=True)

train_curves = extract_window(df, train_years)
val_curves = extract_window(df, val_years)

# ===============================================================
# MEAN CURVES
# ===============================================================
train_mean = train_curves.groupby("Days_Relative")["CumVol_norm"].mean().reset_index()
val_mean = val_curves.groupby("Days_Relative")["CumVol_norm"].mean().reset_index()

x_train = train_mean["Days_Relative"].values
y_train = train_mean["CumVol_norm"].values
x_val = val_mean["Days_Relative"].values
y_val = val_mean["CumVol_norm"].values

# ===============================================================
# DEFINE CANDIDATE MODELS
# ===============================================================
def logistic(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def polynomial2(x, a, b, c):
    return a*x**2 + b*x + c

models = {
    "Logistic": (logistic, [0.05, 0]),
    "Exponential": (exponential, [1, -0.01, 0]),
    "Quadratic": (polynomial2, [1e-5, 1e-3, 0]),
}

# ===============================================================
# FIT AND EVALUATE MODELS
# ===============================================================
fit_results = []

for name, (func, p0) in models.items():
    try:
        popt, _ = curve_fit(func, x_train, y_train, p0=p0, maxfev=20000)
        y_pred_train = func(x_train, *popt)
        y_pred_val = func(x_val, *popt)

        r2_train = r2_score(y_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        r2_val = r2_score(y_val, y_pred_val)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

        fit_results.append((name, popt, r2_train, rmse_train, r2_val, rmse_val))
        print(f"{name:10s}  Train R²={r2_train:.4f}, RMSE={rmse_train:.4f} | Val R²={r2_val:.4f}, RMSE={rmse_val:.4f}")
    except RuntimeError:
        print(f"{name} fit did not converge")

# ===============================================================
# SELECT BEST MODEL
# ===============================================================
best_model = max(fit_results, key=lambda x: x[4])  # best validation R²
best_name, best_params = best_model[0], best_model[1]
print(f"\nBest model: {best_name} with params {best_params}")

# ===============================================================
# PLOT RESULTS
# ===============================================================
plt.figure(figsize=(10,6))
plt.scatter(x_train, y_train, color='blue', s=10, label='Train mean')
plt.scatter(x_val, y_val, color='orange', s=10, label='Validation mean')

x_fit = np.linspace(-window_days, window_days, 300)
y_fit = models[best_name][0](x_fit, *best_params)
plt.plot(x_fit, y_fit, color='green', lw=2, label=f'{best_name} fit')

plt.axvline(0, color='gray', linestyle='--', alpha=0.6)
plt.xlabel("Days relative to target date")
plt.ylabel("Normalized cumulative volume")
plt.title(f"Mean cumulative curves ±{window_days} days around {ref_month}/{ref_day}")
plt.legend()
plt.tight_layout()
plt.show()
