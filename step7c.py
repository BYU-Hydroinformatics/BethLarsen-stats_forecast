import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os

# ==========================================================
# CONFIGURATION
# ==========================================================
input_csv = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"  # <-- replace with your flow file
date_col = "Date"
flow_col = "Flow_cms"

months_before = 9   # observed period length before forecast reference
months_after = 3    # forecast period length
ref_month = 9       # reference month (e.g., June)
ref_day = 15        # reference day of month

save_dir = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/savedplots2"
os.makedirs(save_dir, exist_ok=True)

# ==========================================================
# LOAD AND PREP DATA
# ==========================================================
df = pd.read_csv(input_csv, parse_dates=[date_col])
df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)  # <--- ADD THIS
df = df.sort_values(date_col).set_index(date_col)
df["Volume_m3"] = df[flow_col] * 24 * 3600  # daily volume (if daily data)
df["Year"] = df.index.year
df.index.to_series().diff().value_counts()

# ==========================================================
# SETUP COMMON GRID
# ==========================================================
days_common = np.arange(0, 93)  # 3-month forecast horizon (~90 days)
summary_list = []

# ==========================================================
# LOOP THROUGH EACH YEAR
# ==========================================================
for test_year in sorted(df["Year"].unique()):
    print(f"Processing {test_year}...")

    # Define reference and window dates
    ref_date = pd.Timestamp(test_year, ref_month, ref_day)
    start_ref = ref_date - pd.DateOffset(months=months_before)
    end_ref = ref_date + pd.DateOffset(months=months_after)

    # Pull observed data across year boundaries
    sub_all = df.loc[str(test_year - 1):str(test_year + 1)].copy()
    sub_all = sub_all.loc[start_ref:end_ref].copy()

    # Reset cumulative within this window
    sub_all["CumVol"] = sub_all["Volume_m3"].cumsum()
    sub_all["CumVol"] -= sub_all["CumVol"].iloc[0]

    # Compute cumulative at reference date (start of forecast)
    cum_at_ref = sub_all.loc[sub_all.index <= ref_date, "CumVol"].iloc[-1]

    # Split data into pre- and post-forecast
    obs_pre = sub_all.loc[:ref_date]
    obs_post = sub_all.loc[ref_date:]

    # Create historical post-ref dataset for all years (training pool)
    hist_post_list = []
    for y in sorted(df["Year"].unique()):
        ref_y = pd.Timestamp(y, ref_month, ref_day)
        start_y = ref_y - pd.DateOffset(months=months_before)
        end_y = ref_y + pd.DateOffset(months=months_after)

        sub_y = df.loc[str(y - 1):str(y + 1)].copy()
        sub_y = sub_y.loc[start_y:end_y].copy()
        if sub_y.empty:
            continue

        # Reset cumulative within window
        sub_y["CumVol"] = sub_y["Volume_m3"].cumsum()
        sub_y["CumVol"] -= sub_y["CumVol"].iloc[0]

        # Get increment after reference date
        sub_post = sub_y.loc[ref_y:].copy()  # <-- copy ensures no SettingWithCopyWarning
        sub_post.loc[:, "DayAfter"] = (sub_post.index - ref_y).days
        sub_post.loc[:, "IncAfterRef"] = sub_post["CumVol"] - sub_post["CumVol"].iloc[0]
        sub_post.loc[:, "Year"] = y
        hist_post_list.append(sub_post[["Year", "DayAfter", "IncAfterRef"]])

    hist_post = pd.concat(hist_post_list, ignore_index=True)

    # Remove current test year from training
    train = hist_post[hist_post["Year"] != test_year]
    test = hist_post[hist_post["Year"] == test_year].sort_values("DayAfter")

    # Interpolate training years to common grid
    interp_list = []
    for y in train["Year"].unique():
        sub = train[train["Year"] == y].sort_values("DayAfter")
        interp_inc = np.interp(days_common, sub["DayAfter"], sub["IncAfterRef"],
                               left=np.nan, right=np.nan)
        interp_list.append(pd.Series(interp_inc, name=y))
    interp_df = pd.concat(interp_list, axis=1)

    # Compute reference curves
    median_inc = interp_df.median(axis=1).values
    wettest_inc = interp_df.sum().idxmax()
    driest_inc = interp_df.sum().idxmin()
    wettest_curve = interp_df[wettest_inc].values
    driest_curve = interp_df[driest_inc].values

    # Observed (true) forecast-period increments
    true_inc = np.interp(days_common, test["DayAfter"], test["IncAfterRef"],
                         left=np.nan, right=np.nan)

    # Cumulative curves
    true_cum = np.nancumsum(true_inc)
    med_cum = np.nancumsum(median_inc)

    # Scale median to match observed cumulative at end
    scale_factor = true_cum[-1] / med_cum[-1] if med_cum[-1] != 0 else np.nan
    med_scaled_cum = med_cum * scale_factor

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(true_cum, med_scaled_cum))
    r2 = r2_score(true_cum, med_scaled_cum)
    vol_diff = med_scaled_cum[-1] - true_cum[-1]

    summary_list.append({
        "Year": test_year,
        "RMSE": rmse,
        "R2": r2,
        "VolumeDiff": vol_diff,
        "ScaleFactor": scale_factor
    })

    # ==========================================================
    # PLOT EACH YEAR
    # ==========================================================
    plt.figure(figsize=(9, 5))

    # Observed data (past 9 months and forecast period)
    plt.plot(obs_pre.index, obs_pre["CumVol"], color="gray", lw=2, label="Observed (pre-forecast)")
    plt.plot(obs_post.index, obs_post["CumVol"], color="black", linestyle="--", lw=2,
             label="Observed (forecast period)")

    # Forecast curves
    plt.plot(ref_date + pd.to_timedelta(days_common, "D"),
             cum_at_ref + med_cum, "b", alpha=0.4, label="Median")
    plt.plot(ref_date + pd.to_timedelta(days_common, "D"),
             cum_at_ref + med_scaled_cum, "b", lw=2, label="Scaled Median")
    plt.plot(ref_date + pd.to_timedelta(days_common, "D"),
             cum_at_ref + np.nancumsum(wettest_curve), "g", alpha=0.4,
             label=f"Wettest ({wettest_inc})")
    plt.plot(ref_date + pd.to_timedelta(days_common, "D"),
             cum_at_ref + np.nancumsum(driest_curve), "r", alpha=0.4,
             label=f"Driest ({driest_inc})")

    plt.xlabel("Date")
    plt.ylabel("Cumulative Volume (m³)")
    plt.title(f"Forecast Validation – {test_year}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"validation_{test_year}.png"), dpi=200)
    plt.close()

# ==========================================================
# SAVE SUMMARY + METRICS PLOTS
# ==========================================================
summary_df = pd.DataFrame(summary_list).sort_values("Year")
summary_df.to_csv("/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/savedplots2/summary_leave_one_out.csv", index=False)

plt.figure(figsize=(9, 5))
plt.plot(summary_df["Year"], summary_df["RMSE"], "o-", label="RMSE")
plt.xlabel("Validation Year")
plt.ylabel("RMSE")
plt.title("RMSE of Scaled Median Forecast by Year")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("RMSE_over_time.png", dpi=300)
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
ax[0].plot(summary_df["Year"], summary_df["R2"], "o-", color="teal")
ax[0].set_ylabel("R²")
ax[0].grid(alpha=0.3)

ax[1].plot(summary_df["Year"], summary_df["VolumeDiff"], "o-", color="darkorange")
ax[1].set_xlabel("Validation Year")
ax[1].set_ylabel("Final Volume Difference (m³)")
ax[1].grid(alpha=0.3)

fig.suptitle("Model Validation Metrics Over Time", fontsize=13)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig("Validation_Metrics_over_time.png", dpi=300)
plt.show()

print("✅ Done! All yearly plots and metrics saved.")

obs_start = df.loc[df['Date'] == forecast_start, 'Cumulative'].iloc[0]
obs_end = df.loc[df['Date'] == forecast_end, 'Cumulative'].iloc[0]
print(obs_end - obs_start)

print("Median daily increments:", median_daily[:5])
print("Logistic daily increments:", logistic_daily[:5])
print("Historical wettest daily increments:", wet_daily[:5])
