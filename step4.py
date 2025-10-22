# step_4.py
# Cumulative volume curves before/after reference date, fit mean equation

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# === SETTINGS ===
csv_path = "/Users/bethlarsen/Documents/Hydro Research/geoglows_stat_forecast/flow_data.csv"
date_col = "Date"
flow_col = "Flow_m3s"
ref_month, ref_day = 3, 15
window_days = 90  # 3 months before/after

# === LOAD DATA ===
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col])
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

# === FIT A FUNCTION ===
# Try logistic function: y = 1 / (1 + exp(−k*(x−x0)))
def logistic(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

popt, _ = curve_fit(logistic, mean_curve["Days_Relative"], mean_curve["CumVol_norm"], p0=[0.05, 0])

# === PLOTS ===
plt.figure(figsize=(8, 6))
for yr, g in all_curves.groupby("Year"):
    plt.plot(g["Days_Relative"], g["CumVol_norm"], color="lightgray", alpha=0.5)
plt.plot(mean_curve["Days_Relative"], mean_curve["CumVol_norm"], color="blue", lw=2, label="Mean")
plt.plot(mean_curve["Days_Relative"],
         logistic(mean_curve["Days_Relative"], *popt),
         color="red", lw=2, label=f"Logistic fit (k={popt[0]:.3f}, x₀={popt[1]:.1f})")
plt.xlabel("Days Relative to Reference Date")
plt.ylabel("Normalized Cumulative Volume")
plt.title(f"Cumulative Volume Curves ±{window_days} days around {ref_month:02d}/{ref_day:02d}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()