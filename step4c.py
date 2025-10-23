# step_4.py
# Cumulative volume curves before/after reference date, fit mean equation

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

# === SETTINGS ===
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"
date_col = "Date"
flow_col = "Flow_cms"
ref_month, ref_day = 12, 15
window_days = 180  # 3 months before/after

# === LOAD DATA ===
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
df = df.set_index(date_col).sort_index()
df["Volume_m3"] = df[flow_col] * 24 * 3600

# === BUILD CUMULATIVE CURVES ===
curves = []
for year in df.index.year.unique():
    ref_date = pd.Timestamp(f"{year}-{ref_month:02d}-{ref_day:02d}")
    start = ref_date - pd.Timedelta(days=window_days)
    end = ref_date + pd.Timedelta(days=window_days)

    if start < df.index.min() or end > df.index.max():
        continue

    window = df.loc[start:end].copy()
    window["Days_Relative"] = (window.index - ref_date).days
    window["CumVol"] = window["Volume_m3"].cumsum()

    # normalize by total 6-month volume (optional)
    window["CumVol_norm"] = window["CumVol"] / window["CumVol"].iloc[-1]
    window["Year"] = year
    curves.append(window[["Days_Relative", "CumVol_norm", "Year"]])

all_curves = pd.concat(curves, ignore_index=True)

# === MEAN CURVE ===
mean_curve = all_curves.groupby("Days_Relative")["CumVol_norm"].mean().reset_index()


# ===============================================================
# STEP 4: Define candidate curve models
# ===============================================================

def logistic(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def power_law(x, a, b):
    return a * np.power(x, b)

def polynomial2(x, a, b, c):
    return a*x**2 + b*x + c

def polynomial3(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

models = {
    "Logistic": (logistic, [0.05, 0]),
    "Exponential": (exponential, [1, -0.01, 0]),
    "Power Law": (power_law, [1, 1]),
    "Quadratic": (polynomial2, [1, 0, 0]),
    "Cubic": (polynomial3, [1, 0, 0, 0]),
}

# ===============================================================
# STEP 5: Fit all models and evaluate
# ===============================================================

x = mean_curve["Days_Relative"].values
y = mean_curve["CumVol_norm"].values

fit_results = []

for name, (func, p0) in models.items():
    try:
        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
        y_pred = func(x, *popt)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        fit_results.append((name, popt, r2, rmse))
        print(f"{name} fit: R²={r2:.4f}, RMSE={rmse:.4f}")
    except RuntimeError:
        print(f"{name} fit failed to converge")

# ===============================================================
# STEP 6: Select and visualize the best-fitting model
# ===============================================================

best_model = max(fit_results, key=lambda i: i[2])  # highest R²
best_name, best_params, best_r2, best_rmse = best_model

print(f"\nBest fit: {best_name} (R²={best_r2:.4f}, RMSE={best_rmse:.4f})")

plt.figure(figsize=(9, 5))
plt.scatter(x, y, color='gray', alpha=0.6, label="Observed mean")
x_fit = np.linspace(x.min(), x.max(), 300)
y_fit = models[best_name][0](x_fit, *best_params)
plt.plot(x_fit, y_fit, color='blue', lw=2, label=f'{best_name} fit')
plt.xlabel("Days Relative to Target Date")
plt.ylabel("Normalized Cumulative Volume")
plt.title(f"Best Curve Fit: {best_name}")
plt.legend()
plt.show()

# ===============================================================
# STEP 7: Save model fit results
# ===============================================================

results_df = pd.DataFrame(
    [
        {"Model": name, "R2": r2, "RMSE": rmse, "Parameters": popt}
        for name, popt, r2, rmse in fit_results
    ]
)
results_df.to_csv("model_fit_results.csv", index=False)
print("\nSaved model fit comparison to 'model_fit_results.csv'")