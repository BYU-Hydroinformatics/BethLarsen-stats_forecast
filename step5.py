import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. Load retrospective CSV ---
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"  # update to your file path
df = pd.read_csv(csv_path)

# Ensure datetime and sorting
df["date"] = pd.to_datetime(df["Date"])
df = df.sort_values("date")

# Compute year and day-of-year cumulative volume
df["year"] = df["date"].dt.year
df["doy"] = df["date"].dt.dayofyear

# If not already cumulative, create cumulative volume per year
df["CumVol"] = df.groupby("year")["Flow_cms"].cumsum()
df["CumVol_norm"] = df.groupby("year")["CumVol"].transform(lambda x: x / x.max())

# --- 2. Randomly split into train (90%) and validation (10%) sets by year ---
np.random.seed(42)  # for reproducibility
years = df["year"].unique()
val_years = np.random.choice(years, size=int(0.1 * len(years)), replace=False)
train_years = [y for y in years if y not in val_years]

train = df[df["year"].isin(train_years)]
val = df[df["year"].isin(val_years)]

print(f"Training years: {len(train_years)}, Validation years: {len(val_years)}")

# --- 3. Create mean normalized curve for training set ---
mean_curve = (
    train.groupby("doy")["CumVol_norm"]
    .mean()
    .reset_index()
    .rename(columns={"doy": "Days_Relative"})
)

# --- 4. Define candidate equations ---
def logistic(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def polynomial2(x, a, b, c):
    return a * x**2 + b * x + c

# --- 5. Fit models to training data ---
x_data = mean_curve["Days_Relative"].values
y_data = mean_curve["CumVol_norm"].values

fits = {}

for name, func, p0 in [
    ("logistic", logistic, [0.05, 180]),
    ("exponential", exponential, [0.001, 0.01, 0]),
    ("polynomial2", polynomial2, [1e-5, 1e-3, 0]),
]:
    try:
        popt, _ = curve_fit(func, x_data, y_data, p0=p0, maxfev=20000)
        y_pred_train = func(x_data, *popt)
        r2_train = r2_score(y_data, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_data, y_pred_train))
        fits[name] = {"func": func, "popt": popt, "r2_train": r2_train, "rmse_train": rmse_train}
    except RuntimeError:
        print(f"⚠️ {name} fit did not converge.")
        continue

# --- 6. Evaluate on validation years ---
val_mean = (
    val.groupby("doy")["CumVol_norm"]
    .mean()
    .reset_index()
    .rename(columns={"doy": "Days_Relative"})
)

x_val = val_mean["Days_Relative"].values
y_val = val_mean["CumVol_norm"].values

for name, info in fits.items():
    func, popt = info["func"], info["popt"]
    y_pred_val = func(x_val, *popt)
    r2_val = r2_score(y_val, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    fits[name]["r2_val"] = r2_val
    fits[name]["rmse_val"] = rmse_val

# --- 7. Display results ---
for name, info in fits.items():
    print(f"\n{name.upper()}")
    print(f"  Training R²: {info['r2_train']:.4f}, RMSE: {info['rmse_train']:.4f}")
    print(f"  Validation R²: {info['r2_val']:.4f}, RMSE: {info['rmse_val']:.4f}")

# --- 8. Plot training and validation curves ---
plt.figure(figsize=(10,6))
plt.scatter(x_data, y_data, s=8, color='blue', label='Training Mean Curve')
plt.scatter(x_val, y_val, s=8, color='orange', label='Validation Mean Curve')

for name, info in fits.items():
    func, popt = info["func"], info["popt"]
    plt.plot(x_data, func(x_data, *popt), label=f"{name} fit (R²={info['r2_val']:.3f})")

plt.xlabel("Day of Year")
plt.ylabel("Normalized Cumulative Volume")
plt.legend()
plt.title("Training vs Validation Model Fits (Random 10% Validation)")
plt.show()