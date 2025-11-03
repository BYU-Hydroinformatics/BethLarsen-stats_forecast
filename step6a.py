# ytd_scenarios_plot.py
# Show YTD observed cumulative for a year and possible end-of-year scenarios (wettest, driest, median)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ---------------- USER SETTINGS ----------------
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"   # <<-- change this
date_col = "Date"
flow_col = "Flow_cms"

validation_year = 2020   # the year you want to visualize
ref_month, ref_day = 9, 15   # reference date for “today”

# ---------------- LOAD AND PREP ----------------
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
df = df.set_index(date_col).sort_index()
df["Volume_m3"] = df[flow_col] * 24 * 3600

# ensure full-year daily data
df["Year"] = df.index.year
df["DOY"] = df.index.dayofyear

# compute cumulative per year
cum_df = (
    df.groupby("Year")["Volume_m3"]
    .cumsum()
    .rename("CumVol")
    .to_frame()
    .join(df["Year"])
)

# ---------------- GET REFERENCE INFO ----------------
ref_date = pd.Timestamp(validation_year, ref_month, ref_day)
start_of_year = pd.Timestamp(validation_year, 1, 1)
end_of_year = pd.Timestamp(validation_year, 12, 31)

if ref_date not in df.index:
    ref_date = df.index[(df.index - pd.Timestamp(validation_year, ref_month, ref_day)).abs().argmin()]
    print(f"Exact ref date missing; using nearest available date: {ref_date.date()}")

# validation year observed up to ref date
this_year = cum_df[cum_df["Year"] == validation_year].copy()
this_year_cum = this_year.loc[start_of_year:ref_date]
cum_ref = this_year_cum["CumVol"].iloc[-1]

# ---------------- BUILD HISTORICAL POST-DATE SEGMENTS ----------------
post_curves = []
years = sorted(df["Year"].unique())
for y in years:
    if y == validation_year:
        continue
    year_df = cum_df[cum_df["Year"] == y].copy()
    if ref_date.replace(year=y) not in year_df.index:
        continue
    # find cumulative at ref-date equivalent
    ref_equiv = ref_date.replace(year=y)
    cum_start = year_df.loc[:ref_equiv, "CumVol"].iloc[-1]
    year_post = year_df.loc[ref_equiv:end_of_year]
    if len(year_post) < 10:
        continue
    # relative increase after ref date
    rel_days = (year_post.index - ref_equiv).days
    rel_increase = year_post["CumVol"].values - cum_start
    post_curves.append(pd.DataFrame({
        "Year": y,
        "DaysAfter": rel_days,
        "IncAfterRef": rel_increase
    }))

if not post_curves:
    raise RuntimeError("No full post-date curves found in history")

post_df = pd.concat(post_curves)

# ---------------- FIND WETTEST / DRIEST / MEDIAN CURVES ----------------
# Interpolate each curve to common daily grid (0..days_remaining)
days_remaining = (end_of_year - ref_date).days
x_common = np.arange(0, days_remaining + 1)
interp_curves = []

for y in post_df["Year"].unique():
    w = post_df[post_df["Year"] == y]
    interp = np.interp(x_common, w["DaysAfter"], w["IncAfterRef"])
    interp_curves.append(pd.Series(interp, name=y))

interp_df = pd.concat(interp_curves, axis=1)

if interp_df.empty or interp_df.shape[0] == 0:
    raise ValueError("No data available for interpolation — check your date window or input CSV.")

# Find the last valid row (end-of-year)
last_row = interp_df.iloc[-1]

# Identify columns with max/min cumulative volume
wettest_col = last_row.idxmax()
driest_col = last_row.idxmin()

wettest = interp_df[wettest_col]
driest = interp_df[driest_col]
median  = interp_df.median(axis=1)

# ---------------- BUILD SCENARIO LINES ----------------
dates_future = pd.date_range(ref_date, end_of_year, freq="D")
cum_ytd_series = this_year_cum["CumVol"]

# continuation scenarios = cum_ref + relative increase
wet_line = cum_ref + wettest.values
dry_line = cum_ref + driest.values
med_line = cum_ref + median.values

# ---------------- PLOT ----------------
plt.figure(figsize=(9,5))
plt.plot(this_year_cum.index, this_year_cum["CumVol"], label=f"{validation_year} YTD observed", color="tab:blue")
plt.plot(dates_future, wet_line, label="Wettest historical continuation", color="purple", linestyle="--")
plt.plot(dates_future, dry_line, label="Driest historical continuation", color="orange", linestyle="--")
plt.plot(dates_future, med_line, label="Median continuation", color="green", linestyle="--")

plt.axvline(ref_date, color="gray", linestyle=":", lw=1)
plt.xlabel("Date")
plt.ylabel("Cumulative Volume (m³)")
plt.title(f"{validation_year}: YTD + Historical End-of-Year Scenarios")
plt.legend()
plt.tight_layout()
plt.show()