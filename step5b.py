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
ref_month, ref_day = 9, 15     # reference date (e.g., Sept 15)
window_days = 180              # ±6 months
val_fraction = 0.1             # hold out 10% of years for validation
np.random.seed(42)             # reproducible split

# ===============================================================
# LOAD DATA
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
# FUNCTION TO EXTRACT 6-MONTH WINDOW AROUND REFERENCE DATE
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
# TRAINING: MEAN CURVE FITTING
# ===============================================================
train_mean = train_curves.groupby("Days_Relative")["CumVol_norm"].mean().reset_index()
x_train = train_mean["Days_Relative"].values
y_train = train_mean["CumVol_norm"].values

# ===============================================================
# DEFINE MODEL FUNCTIONS
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
# FIT MODELS TO TRAINING MEAN CURVE
# ===============================================================
fit_results = []
for name, (func, p0) in models.items():
    try:
        popt, _ = curve_fit(func, x_train, y_train, p0=p0, maxfev=20000)
        y_pred_train = func(x_train, *popt)
        r2_train = r2_score(y_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        fit_results.append((name, popt, r2_train, rmse_train))
        print(f"{name:10s}  Train R²={r2_train:.4f}, RMSE={rmse_train:.4f}")
    except RuntimeError:
        print(f"{name} fit did not converge")

# Pick the best model (highest training R²)
best_model = max(fit_results, key=lambda x: x[2])
best_name, best_params = best_model[0], best_model[1]
print(f"\nBest training model: {best_name} with parameters {best_params}")

# ===============================================================
# VALIDATION: TEST PREDICTIONS ON EACH VALIDATION YEAR
# ===============================================================
val_year_scores = []

for year in val_curves["Year"].unique():
    subset = val_curves[val_curves["Year"] == year].copy()
    x_val = subset["Days_Relative"].values
    y_true = subset["CumVol_norm"].values
    y_pred = models[best_name][0](x_val, *best_params)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    val_year_scores.append((year, r2, rmse))

val_df = pd.DataFrame(val_year_scores, columns=["Year", "R2", "RMSE"])
avg_r2 = val_df["R2"].mean()
avg_rmse = val_df["RMSE"].mean()

print(f"\nValidation performance across {len(val_df)} years:")
print(val_df)
print(f"\nMean validation R² = {avg_r2:.4f}, RMSE = {avg_rmse:.4f}")

# ===============================================================
# PLOT RESULTS
# ===============================================================
plt.figure(figsize=(10,6))
plt.scatter(x_train, y_train, color='blue', s=10, label='Train mean')
plt.axvline(0, color='gray', linestyle='--', alpha=0.6)

x_fit = np.linspace(-window_days, window_days, 300)
y_fit = models[best_name][0](x_fit, *best_params)
plt.plot(x_fit, y_fit, color='green', lw=2, label=f'{best_name} fit')

# Add each validation year's actual curve
for year in val_df["Year"]:
    subset = val_curves[val_curves["Year"] == year]
    plt.plot(subset["Days_Relative"], subset["CumVol_norm"], color='orange', alpha=0.5, lw=1)

plt.xlabel("Days relative to target date")
plt.ylabel("Normalized cumulative volume")
plt.title(f"{best_name} fit (training) vs. validation year curves\nMean R²={avg_r2:.3f}, RMSE={avg_rmse:.3f}")
plt.legend()
plt.tight_layout()
plt.show()