import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you prefer
import matplotlib.pyplot as plt
from datetime import datetime

# === USER SETTINGS ===
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"      # your CSV
date_col = "Date"
flow_col = "Flow_cms"
ref_month, ref_day = 3, 15        # March 15 reference
window_months = 6
normalize = True                  # True for shape comparison
highlight_year = 2024             # optional: highlight a specific year

# === 1. Load & prepare ===
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
df = df.set_index(date_col).sort_index().asfreq("D")

# === 2. Extract 24-month windows per year ===
years = df.index.year.unique()
records = []  # to store all curves together

for year in years:
    ref_date = datetime(year, ref_month, ref_day)
    start_date = ref_date - pd.DateOffset(months=window_months)
    end_date = ref_date + pd.DateOffset(months=window_months)
    if start_date < df.index.min() or end_date > df.index.max():
        continue

    w = df.loc[start_date:end_date].copy()
    w["Volume_m3"] = w[flow_col] * 86400
    w["Cumulative_m3"] = w["Volume_m3"].cumsum()
    if normalize:
        w["Cumulative_m3"] /= w["Cumulative_m3"].iloc[-1]
    w["Days_from_ref"] = (w.index - ref_date).days
    w["Year"] = year
    records.append(w[["Days_from_ref", "Cumulative_m3", "Year"]])

data = pd.concat(records)

# === 3. Compute percentiles across years ===
grouped = data.groupby("Days_from_ref")["Cumulative_m3"]
stats = pd.DataFrame({
    "median": grouped.median(),
    "p10": grouped.quantile(0.1),
    "p90": grouped.quantile(0.9)
}).reset_index()

# === 4. Plot ===
plt.figure(figsize=(10,6))

# Historical envelope
plt.fill_between(stats["Days_from_ref"], stats["p10"], stats["p90"],
                 color="lightblue", alpha=0.4, label="10–90% range")
plt.plot(stats["Days_from_ref"], stats["median"],
         color="blue", linewidth=2, label="Median")

# Highlight chosen year (if available)
if highlight_year in data["Year"].unique():
    current = data[data["Year"] == highlight_year]
    plt.plot(current["Days_from_ref"], current["Cumulative_m3"],
             color="red", linewidth=2, label=f"{highlight_year}")

plt.axvline(0, color="black", linestyle="--", label="Reference date")
plt.title(f"Cumulative Volume Envelopes ({window_months} mo before/after {ref_month:02d}-{ref_day:02d})")
plt.xlabel("Days from reference date")
plt.ylabel("Normalized cumulative volume" if normalize else "Cumulative volume (m³)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
