import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === USER INPUTS ===
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"  # your full 80-year daily flow dataset
date_col = "Date"
flow_col = "Flow_cms"
target_day = "4-15"  # reference date (MM-DD)
months_window = 6     # number of months before/after
validation_fraction = 0.1  # fraction of years to leave out for testing

# === MODEL FUNCTIONS ===
def logistic(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def power_law(x, a, b):
    return a * np.power(x, b)

def polynomial2(x, a, b, c):
    return a * x**2 + b * x + c

def polynomial3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

models = {
    "Logistic": (logistic, [0.05, 0]),
    "Exponential": (exponential, [1, -0.01, 0]),
    "Power Law": (power_law, [1, 1]),
    "Quadratic": (polynomial2, [1, 0, 0]),
    "Cubic": (polynomial3, [1, 0, 0, 0]),
}

# === FUNCTIONS ===
def get_window_data(df, target_day, months_window):
    """Return data within +/- months_window around target_day for all years"""
    df["Date"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df = df.set_index("Date").sort_index()
    years = df.index.year.unique()

    all_years_data = []

    for y in years:
        center_date = pd.Timestamp(f"{y}-{target_day}")
        start_date = center_date - pd.DateOffset(months=months_window)
        end_date = center_date + pd.DateOffset(months=months_window)

        if start_date < df.index.min() or end_date > df.index.max():
            continue

        subset = df.loc[start_date:end_date].copy()
        subset["Year"] = y
        subset["Days_Relative"] = (subset.index - center_date).days
        subset["CumVol"] = subset[flow_col].cumsum()
        subset["CumVol_norm"] = subset["CumVol"] / subset["CumVol"].max()
        all_years_data.append(subset)

    return pd.concat(all_years_data)

# === LOAD & PREP DATA ===
df = pd.read_csv(csv_path)
window_data = get_window_data(df, target_day, months_window)
years = sorted(window_data["Year"].unique())
n_valid = max(1, int(len(years) * validation_fraction))

train_years = years[:-n_valid]
valid_years = years[-n_valid:]

train_data = window_data[window_data["Year"].isin(train_years)]
valid_data = window_data[window_data["Year"].isin(valid_years)]

# === FIT EACH MODEL ON TRAINING DATA ===
mean_curve = (
    train_data.groupby("Days_Relative")["CumVol_norm"]
    .mean()
    .reset_index()
    .dropna()
)
x_train = mean_curve["Days_Relative"].values
y_train = mean_curve["CumVol_norm"].values

fit_results = []
for name, (func, p0) in models.items():
    try:
        popt, _ = curve_fit(func, x_train, y_train, p0=p0, maxfev=10000)
        y_pred = func(x_train, *popt)
        r2 = r2_score(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        fit_results.append((name, func, popt, r2, rmse))
        print(f"{name:10s} | R²={r2:.4f} | RMSE={rmse:.4f}")
    except RuntimeError:
        print(f"{name} fit failed to converge")

# === SELECT BEST MODEL ===
best_model = max(fit_results, key=lambda x: x[3])
best_name, best_func, best_params, best_r2, best_rmse = best_model
print(f"\nBest model: {best_name} (R²={best_r2:.4f}, RMSE={best_rmse:.4f})")

# === VALIDATION EVALUATION ===
validation_results = []
for year in valid_years:
    df_y = valid_data[valid_data["Year"] == year].copy()
    X = df_y["Days_Relative"].values
    y_true = df_y["CumVol_norm"].values
    y_pred = best_func(X, *best_params)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    validation_results.append({"Year": year, "RMSE": rmse, "MAE": mae, "R2": r2})

    # Plot observed vs predicted
    plt.figure(figsize=(7, 4))
    plt.plot(X, y_true, label=f"Observed {year}", color="blue")
    plt.plot(X, y_pred, label=f"Predicted ({best_name})", color="orange", linestyle="--")
    plt.title(f"Validation Year {year} — {best_name} Fit")
    plt.xlabel("Days Relative to Target Date")
    plt.ylabel("Normalized Cumulative Volume")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === SUMMARY TABLE ===
results_df = pd.DataFrame(validation_results)
print("\n=== Validation Performance ===")
print(results_df.round(3))
print("\nAverage RMSE:", results_df["RMSE"].mean())
print("Average R²:", results_df["R2"].mean())