import pandas as pd
import matplotlib
import numpy as np
matplotlib.use("TkAgg")  # or "Qt5Agg" if you prefer
import matplotlib.pyplot as plt
from datetime import datetime

# --- Load data ---
df = pd.read_csv("/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv")

# --- Parse date and basic time columns ---
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["DOY"] = df["Date"].dt.dayofyear

# --- Compute daily and cumulative volume ---
df["Volume_m3"] = df["Flow_cms"] * 86400  # m³/s → m³/day
df["Cumulative_Volume_m3"] = df.groupby("Year")["Volume_m3"].cumsum()

# --- Compute total annual volume for each year ---
current_year = datetime.now().year
yearly_total = df.groupby("Year")["Volume_m3"].sum()
yearly_total_clean = yearly_total.drop(current_year, errors="ignore")
yearly_sorted = yearly_total_clean.sort_values()

print(yearly_total)

# Identify special years

wettest_year = yearly_sorted.idxmax()
driest_year = yearly_sorted.idxmin()

median_year = yearly_sorted.index[len(yearly_sorted)//2]
year_25 = yearly_sorted.index[int(0.25 * (len(yearly_sorted) - 1))]
year_75 = yearly_sorted.index[int(0.75 * (len(yearly_sorted) - 1))]

print(f"Wettest year: {wettest_year}")
print(f"Driest year: {driest_year}")
print(f"Median year: {median_year}")

# --- Compute average cumulative volume (mean across all years) ---
avg_cum = df.groupby("DOY")["Cumulative_Volume_m3"].mean().reset_index()

# --- Plot ---
plt.figure(figsize=(12, 7))

# All years in light gray (spaghetti background)
for year, group in df.groupby("Year"):
    plt.plot(group["DOY"], group["Cumulative_Volume_m3"]/1e6,
             color="lightgray", alpha=0.4, linewidth=0.8)

# Highlight key years
def plot_highlight(year, color, label):
    g = df[df["Year"] == year]
    plt.plot(g["DOY"], g["Cumulative_Volume_m3"]/1e6,
             color=color, lw=2, label=label)

plot_highlight(wettest_year, "blue", f"Wettest ({wettest_year})")
plot_highlight(driest_year, "red", f"Driest ({driest_year})")
plot_highlight(median_year, "green", f"Median ({median_year})")
plot_highlight(year_25, "yellow", f"25th Percentile ({year_25})")
plot_highlight(year_75, "purple", f"75th Percentile ({year_75})")

# Average line
plt.plot(avg_cum["DOY"], avg_cum["Cumulative_Volume_m3"]/1e6,
         color="black", lw=2.5, label="Average")

# --- Formatting ---
plt.xlabel("Day of Year")
plt.ylabel("Cumulative Volume (million m³)")
plt.title("Cumulative Volume by Year (1940–Present)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()