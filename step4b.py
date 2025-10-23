# step_4_self_sufficient.py
# Beth Larsen | Flow Forecast Project | Step 4 (Self-contained curve fitting)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

# ===============================================================
# STEP 1: Load daily flow data
# ===============================================================

# Example CSV format:  Date, Flow
# Date = YYYY-MM-DD, Flow = mean daily discharge (m3/s)
df = pd.read_csv("/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv")

# Convert date column to datetime and set as index
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

# ===============================================================
# STEP 2: Choose target day and extract +/- 3 months each year
# ===============================================================

target_month = 3   # March
target_day = 15
window_days = 90   # roughly 3 months before and after

records = []

for year in range(df.index.year.min(), df.index.year.max() + 1):
    try:
        target_date = pd.Timestamp(year, target_month, target_day)
        start_date = target_date - timedelta(days=window_days)
        end_date = target_date + timedelta(days=window_days)

        subset = df.loc[start_date:end_date].copy()
        if subset.empty:
            continue

        subset["Days_Relative"] = (subset.index - target_date).days
        subset["Year"] = year

        # Compute cumulative volume (m3/s * day)
        subset["CumVol"] = subset["Flow_cms"].cumsum()

        # Normalize cumulative volume (0–1)
        subset["CumVol_norm"] = (subset["CumVol"] - subset["CumVol"].min()) / (
            subset["CumVol"].max() - subset["CumVol"].min()
        )

        records.append(subset[["Year", "Days_Relative", "CumVol_norm"]])
    except Exception:
        continue

all_data = pd.concat(records, ignore_index=True)

# ===============================================================
# STEP 3: Compute mean normalized cumulative curve across years
# ===============================================================

mean_curve = all_data.groupby("Days_Relative")["CumVol_norm"].mean().reset_index()

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