import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# === USER SETTINGS ===
csv_path = "/Users/beth/Downloads/retrospective_760706416_ohio.csv"  # your file path
date_col = "Date"  # column name for dates
flow_col = "Flow_m3s"  # column name for flow
ref_month = 3  # March
ref_day = 15  # 15th
window_months = 12  # before and after (so total 24 months)
normalize = False  # set True if you want normalized volumes

# === 1. Load and prepare data ===
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
df = df.set_index(date_col).sort_index()

# Ensure continuous daily data (optional)
df = df.asfreq("D")

# === 2. Loop through each year and extract 24-month window ===
years = df.index.year.unique()
results = {}

for year in years:
    ref_date = datetime(year, ref_month, ref_day)
    start_date = ref_date - pd.DateOffset(months=window_months)
    end_date = ref_date + pd.DateOffset(months=window_months)

    # Skip if window is incomplete (e.g., first/last years)
    if start_date < df.index.min() or end_date > df.index.max():
        continue

    window = df.loc[start_date:end_date].copy()

    # === 3. Compute cumulative volume (m³) ===
    window["Volume_m3"] = window[flow_col] * 86400  # 86400 sec/day
    window["Cumulative_m3"] = window["Volume_m3"].cumsum()

    # Optional normalization by final cumulative volume
    if normalize:
        window["Cumulative_m3"] /= window["Cumulative_m3"].iloc[-1]

    # Store with relative day axis
    window["Days_from_ref"] = (window.index - ref_date).days
    results[year] = window

# === 4. Plot spaghetti of all 24-month cumulative volumes ===
plt.figure(figsize=(10, 6))
for year, w in results.items():
    plt.plot(w["Days_from_ref"], w["Cumulative_m3"], alpha=0.5, label=str(year))

plt.axvline(0, color="black", linestyle="--", label="Reference date")
plt.title(f"Cumulative Volume ({window_months} mo before/after {ref_month:02d}-{ref_day:02d})")
plt.xlabel("Days from reference date")
plt.ylabel("Cumulative Volume (m³)" if not normalize else "Normalized cumulative volume")
plt.grid(True, alpha=0.3)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()
