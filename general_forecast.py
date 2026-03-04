import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for file-saving (no GUI required)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------
# USER SETTINGS
# -------------------------------

input_csv = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/Retrospective_Data/retrospective_ohio.csv"

date_col = "Date"
flow_col = "Discharge"

past_months = 9
future_months = 3


# -------------------------------
# PLOT SETTINGS
# -------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10
})

# -------------------------------
# LOAD DATA
# -------------------------------

df = pd.read_csv(input_csv, parse_dates=[date_col])
df[date_col] = df[date_col].dt.tz_localize(None)
df["Year"] = df[date_col].dt.year

# Convert discharge (m3/s) to daily volume (m3)
df["volume_cms"] = df["Flow_cms"] * 86400.0

today = df[date_col].max().floor("D")
current_year = today.year

# -------------------------------
# OBSERVED PAST (LAST 9 MONTHS)
# -------------------------------

past_start_obs = today - pd.DateOffset(months=past_months)

obs_past = df[
    (df[date_col] >= past_start_obs) &
    (df[date_col] <= today)
].copy()

# Collapse to daily volume
obs_past = (
    obs_past
    .assign(DateDay=obs_past[date_col].dt.floor("D"))
    .groupby("DateDay", as_index=False)["volume_cms"]
    .sum()
)

obs_past[date_col] = obs_past["DateDay"]

# Cumulative observed volume
obs_past["CumVol"] = obs_past["volume_cms"].cumsum()
obs_past["Day"] = (obs_past[date_col] - today).dt.days

obs_curve = obs_past.set_index("Day")["CumVol"]

# -------------------------------
# STORAGE
# -------------------------------


future_curves = {}   # year -> Series indexed by DayAfterRef
future_totals = {}   # year -> total 3-mo volume

# -------------------------------
# LOOP OVER YEARS
# -------------------------------

for y in sorted(df["Year"].unique()):
    ref_date = today.replace(year=y)

    past_start = ref_date - pd.DateOffset(months=past_months)
    future_end = ref_date + pd.DateOffset(months=future_months)

    sub = df.copy()

    # ---- Past window ----
    past = sub[(sub[date_col] >= past_start) &
               (sub[date_col] < ref_date)].sort_values(date_col)

    # ---- Future window ----
    future = sub[(sub[date_col] > ref_date) &
                 (sub[date_col] <= future_end)]

    # --- Collapse to daily volume (handles sub-daily / duplicates, year-crossing safe) ---
    future = (
        future
        .assign(DateDay=future[date_col].dt.floor("D"))
        .groupby("DateDay", as_index=False)["volume_cms"]
        .sum()
    )

    future[date_col] = future["DateDay"]

    # Require full coverage
    if past.empty or future.empty:
        continue

    # --- Collapse to daily volume (handles sub-daily or duplicate timestamps) ---
    past = (
        past
        .assign(DateDay=past[date_col].dt.floor("D"))
        .groupby("DateDay", as_index=False)["volume_cms"]
        .sum()
    )


    # ---- Future cumulative (reset at ref) ----
    future["CumVol"] = future["volume_cms"].cumsum()
    future["Day"] = (future[date_col] - ref_date).dt.days
    future_curve = future.set_index("Day")["CumVol"]

    # >>>>>>> OPTION B FIX STARTS HERE <<<<<<<
    full_future_days = np.arange(
        0,
        (future_end - ref_date).days + 1
    )

    future_daily = (
        future
        .set_index("Day")["volume_cms"]
        .reindex(full_future_days)
        .fillna(0)
    )

    future_curve = future_daily.cumsum()
    # >>>>>>> OPTION B FIX ENDS HERE <<<<<<<
    was_missing = ~future.set_index("Day")["volume_cms"].reindex(full_future_days).notna()
    n_filled_zeros = was_missing.sum()

    print(f"Year {y}: {n_filled_zeros} zero days from gap-filling")

    future_curves[y] = future_curve
    future_totals[y] = future_curve.iloc[-1]




print("Number of future curves:", len(future_curves))

# -------------------------------
# INTERPOLATE TO COMMON DAY GRIDS
# -------------------------------



future_min = max(s.index.min() for s in future_curves.values())
future_max = min(s.index.max() for s in future_curves.values())
future_days = np.arange(future_min, future_max + 1)



future_interp = {}
for y, s in future_curves.items():
    future_interp[y] = np.interp(
        future_days, s.index, s.values, left=np.nan, right=np.nan
    )

future_arr = np.vstack(list(future_interp.values()))


# -------------------------------
# FORECAST STATISTICS
# -------------------------------

future_median = np.nanmedian(future_arr, axis=0)
future_p25 = np.nanpercentile(future_arr, 25, axis=0)
future_p75 = np.nanpercentile(future_arr, 75, axis=0)

# Wettest / driest years by total volume
wettest_year = max(future_totals, key=future_totals.get)
driest_year = min(future_totals, key=future_totals.get)

wettest_curve = future_interp[wettest_year]
driest_curve = future_interp[driest_year]

print("OBSERVED:")
print("  min day:", obs_past.min())
print("  max day:", obs_past.max())

print("FORECAST:")
print("  min day:", future_days.min())
print("  max day:", future_days.max())

# -------------------------------
# PLOTTING
# -------------------------------

print(future_days[:5], future_days[-5:])

hist_anchor = obs_curve.iloc[-1]  # value at day 0-


plt.figure(figsize=(12, 6))

# Past median
plt.plot(
    obs_curve.index,
    obs_curve.values,
    label="Observed (last 9 months)",
    linewidth=2
)


# Offset forecast curves so they connect
plt.plot(future_days,
         hist_anchor + future_median,
         label="Forecast Median", linewidth=2)

plt.fill_between(
    future_days,
    hist_anchor + future_p25,
    hist_anchor + future_p75,
    alpha=0.3,
    label="25–75th Percentile"
)

plt.plot(
    future_days,
    hist_anchor + wettest_curve,
    linestyle="--",
    label=f"Wettest Year ({wettest_year})"
)

plt.plot(
    future_days,
    hist_anchor + driest_curve,
    linestyle="--",
    label=f"Driest Year ({driest_year})"
)

# Reference date
plt.axvline(0, linestyle=":", linewidth=2)

plt.xlabel("Days Relative to Reference Date")
plt.ylabel("Cumulative Volume")
plt.title("Cumulative Volume: Observed Flow + 3-Month Statistical Envelope")
plt.legend()
plt.grid(True)



outfile = f"/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/savedplots_general/cumvol_forecast.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()


