import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you prefer
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_csv("/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv", parse_dates=["Date"])
seconds_per_day = 24 * 60 * 60
df["Volume_m3"] = df["Flow_cms"] * seconds_per_day

# --- Remove Feb 29 for consistency ---
df = df[~((df["Date"].dt.month == 2) & (df["Date"].dt.day == 29))]

# --- Add time columns ---
df["Year"] = df["Date"].dt.year
df["DOY"] = df["Date"].dt.dayofyear
df["Month"] = df["Date"].dt.month

# ----------------------------
# 1. Average cumulative volume (daily)
# ----------------------------
avg_daily = df.groupby("DOY")["Volume_m3"].mean()
avg_cum_daily = avg_daily.cumsum()

# ----------------------------
# 2. Identify wettest and driest years
# ----------------------------
annual_totals = df.groupby("Year")["Volume_m3"].sum()
wettest_year = annual_totals.idxmax()
driest_year = annual_totals.idxmin()

print(f"Wettest year: {wettest_year} ({annual_totals[wettest_year]/1e9:.2f} billion m³)")
print(f"Driest year: {driest_year} ({annual_totals[driest_year]/1e9:.2f} billion m³)")

# ----------------------------
# 3. Compute cumulative volumes per year
# ----------------------------
df["CumVolume"] = df.groupby("Year")["Volume_m3"].cumsum()

# Subset for wettest & driest
subset = df[df["Year"].isin([wettest_year, driest_year])]

# ----------------------------
# 4. Average cumulative volume (monthly)
# ----------------------------
import calendar

# --- Correct monthly aggregation ---
# Average flow (m³/s) per month across all years
avg_monthly_flow = df.groupby("Month")["Flow_cms"].mean()

# Convert to monthly volume: flow * seconds/day * days_in_month
avg_monthly_volume = []
for month, flow in avg_monthly_flow.items():
    days_in_month = calendar.monthrange(2021, month)[1]  # use a non-leap year
    volume = flow * seconds_per_day * days_in_month
    avg_monthly_volume.append(volume)

avg_monthly_volume = pd.Series(avg_monthly_volume, index=range(1,13))

# Now cumulative
avg_cum_monthly = avg_monthly_volume.cumsum()

# ----------------------------
# 5. Plotting
# ----------------------------
fig, axes = plt.subplots(2, 1, figsize=(12,12), sharex=False)

# --- Left: Daily (with wettest & driest years) ---
axes[0].plot(avg_cum_daily.index, avg_cum_daily/1e6, color="black", lw=2, label="Average (1940–Present)")
for y, color in zip([wettest_year, driest_year], ["blue", "red"]):
    year_data = subset[subset["Year"] == y]
    axes[0].plot(year_data["DOY"], year_data["CumVolume"]/1e6, lw=1.5, label=f"{y} ({'Wettest' if y==wettest_year else 'Driest'})", color=color)
axes[0].set_xlabel("Day of Year")
axes[0].set_ylabel("Cumulative Volume (Million m³)")
axes[0].set_title("Cumulative Volume by Day")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Right: Monthly average ---
axes[1].plot(range(1,13), avg_cum_monthly/1e6, color="blue", lw=2, marker="o")
axes[1].set_xticks(range(1,13))
axes[1].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
axes[1].set_ylabel("Cumulative Volume (Million m³)")
axes[1].set_title("Average Cumulative Volume by Month")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
