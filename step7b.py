# ==========================================================
# step7_leave_one_out_forecast_with_history.py
# ==========================================================
# Performs leave-one-year-out validation for 3-month flow forecasts
# using historical increment curves from 9-month reference periods.
# Includes observed data from the previous 9 months in each plot.
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os

# ==========================================================
# SETTINGS
# ==========================================================
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"
date_col = "Date"
flow_col = "Flow_cms"
ref_month, ref_day = 6, 15      # reference forecast date
months_before = 9
months_after = 3
save_dir = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/saved_plots"
os.makedirs(save_dir, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
df = df.set_index(date_col).sort_index()
df["Volume_m3"] = df[flow_col] * 24 * 3600  # convert to m³/day

years_all = sorted(df.index.year.unique())
print(f"Total available years: {len(years_all)}")

# ==========================================================
# FUNCTION: Build historical post-reference increment curves
# ==========================================================
def build_post_increment_curves(df, years, ref_month, ref_day, months_after):
    curves = []
    for y in years:
        ref_date = pd.Timestamp(y, ref_month, ref_day)
        end_ref = ref_date + pd.DateOffset(months=months_after)
        sub = df.loc[str(y - 1):str(y + 1)].copy()  # allow cross-year
        if ref_date not in sub.index or end_ref > sub.index.max():
            continue
        sub["CumVol"] = sub["Volume_m3"].cumsum()
        cum_at_ref = sub.loc[sub.index <= ref_date, "CumVol"].iloc[-1]
        post_df = sub.loc[sub.index > ref_date]
        inc = post_df["CumVol"].values - cum_at_ref
        days = (post_df.index - ref_date).days.values
        curves.append(pd.DataFrame({"Year": y, "DayAfter": days, "IncAfterRef": inc}))
    return pd.concat(curves, ignore_index=True)

# ==========================================================
# BUILD HISTORICAL CURVES
# ==========================================================
hist_post = build_post_increment_curves(df, years_all, ref_month, ref_day, months_after)
max_day = int(hist_post["DayAfter"].max())
days_common = np.arange(1, max_day + 1)

# ==========================================================
# LEAVE-ONE-YEAR-OUT VALIDATION
# ==========================================================
summary_list = []

for test_year in years_all:
    print(f"Processing {test_year}...")

    # Split training and test data
    train = hist_post[hist_post["Year"] != test_year]
    test = hist_post[hist_post["Year"] == test_year]
    if test.empty:
        continue

    # Interpolate training curves
    interp_list = []
    for y in train["Year"].unique():
        sub = train[train["Year"] == y].sort_values("DayAfter")
        interp_inc = np.interp(days_common, sub["DayAfter"], sub["IncAfterRef"], left=np.nan, right=np.nan)
        interp_list.append(pd.Series(interp_inc, name=y))
    interp_df = pd.concat(interp_list, axis=1)
    interp_df = interp_df.loc[:, interp_df.iloc[-1].notna()]

    # Compute median, wettest, driest curves
    median_inc = interp_df.median(axis=1).values
    final_vals = interp_df.iloc[-1]
    wettest_year = final_vals.idxmax()
    driest_year = final_vals.idxmin()
    wettest_inc = interp_df[wettest_year].values
    driest_inc = interp_df[driest_year].values

    # True observed increment and cumulative
    test = test.sort_values("DayAfter")
    true_inc = np.interp(days_common, test["DayAfter"], test["IncAfterRef"], left=np.nan, right=np.nan)
    true_cum = np.nancumsum(true_inc)
    med_cum = np.nancumsum(median_inc)

    # Scale median to observed total
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
        "ScaleFactor": scale_factor,
        "WettestTrainYear": wettest_year,
        "DriestTrainYear": driest_year
    })

    # ==========================================================
    # PLOT: Observed 9 months + 3-month forecast
    # ==========================================================
    ref_date = pd.Timestamp(test_year, ref_month, ref_day)
    start_ref = ref_date - pd.DateOffset(months=months_before)
    end_ref = ref_date + pd.DateOffset(months=months_after)
    sub_all = df.loc[start_ref:end_ref].copy()
    sub_all["CumVol"] = sub_all["Volume_m3"].cumsum()
    cum_at_ref = sub_all.loc[sub_all.index <= ref_date, "CumVol"].iloc[-1]

    # Pre and post observed
    obs_pre = sub_all.loc[:ref_date]
    obs_post = sub_all.loc[ref_date:]

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(obs_pre.index, obs_pre["CumVol"], color="black", label="Observed (past 9 months)")
    plt.plot(obs_post.index, obs_post["CumVol"], color="black", linestyle="--", label="Observed (forecast period)")
    plt.plot(obs_post.index, cum_at_ref + med_cum[:len(obs_post)], color="gray", alpha=0.5, label="Median")
    plt.plot(obs_post.index, cum_at_ref + med_scaled_cum[:len(obs_post)], "b", lw=2, label="Scaled Median")
    plt.plot(obs_post.index, cum_at_ref + np.nancumsum(wettest_inc)[:len(obs_post)], "g", alpha=0.4, label=f"Wettest ({wettest_year})")
    plt.plot(obs_post.index, cum_at_ref + np.nancumsum(driest_inc)[:len(obs_post)], "r", alpha=0.4, label=f"Driest ({driest_year})")

    plt.axvline(ref_date, color="gray", linestyle=":", label="Forecast Start")
    plt.title(f"{test_year} Forecast — 9-Month History + 3-Month Forecast")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Volume (m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"forecast_{test_year}.png"), dpi=200)
    plt.close()

# ==========================================================
# SUMMARY + PERFORMANCE PLOTS
# ==========================================================
summary_df = pd.DataFrame(summary_list).sort_values("Year")
summary_df.to_csv("/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/saved_plots/leave_one_out_summary.csv", index=False)
print("\n✅ Saved summary to leave_one_out_summary.csv")

# --- RMSE over time ---
plt.figure(figsize=(9, 5))
plt.plot(summary_df["Year"], summary_df["RMSE"], "o-", label="RMSE")
plt.xlabel("Validation Year")
plt.ylabel("RMSE")
plt.title("RMSE of Scaled Median Forecast by Year")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("RMSE_over_time.png", dpi=300)
plt.show()

# --- R² and VolumeDiff over time ---
fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
ax[0].plot(summary_df["Year"], summary_df["R2"], "o-", color="teal")
ax[0].set_ylabel("R²")
ax[0].grid(True, alpha=0.3)

ax[1].plot(summary_df["Year"], summary_df["VolumeDiff"], "o-", color="darkorange")
ax[1].set_xlabel("Validation Year")
ax[1].set_ylabel("Final Volume Difference (m³)")
ax[1].grid(True, alpha=0.3)

fig.suptitle("Model Validation Metrics Over Time", fontsize=13)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig("Validation_Metrics_over_time.png", dpi=300)
plt.show()

print("✅ All done! Plots and summary saved.")