import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os

# --- INPUTS ---
hist_post = pd.read_csv("hist_post.csv")  # <-- replace with your file path
days_common = np.arange(0, 93)            # standardized day grid
save_dir = "plots_leave_one_out"
os.makedirs(save_dir, exist_ok=True)

# --- STORAGE ---
summary_list = []

# --- LOOP THROUGH EACH YEAR ---
for test_year in sorted(hist_post["Year"].unique()):
    # Split training and validation data
    train = hist_post[hist_post["Year"] != test_year]
    test = hist_post[hist_post["Year"] == test_year].sort_values("DayAfter")

    # Interpolate all training years to common grid
    interp_list = []
    for y in train["Year"].unique():
        sub = train[train["Year"] == y].sort_values("DayAfter")
        interp_inc = np.interp(days_common, sub["DayAfter"], sub["IncAfterRef"], left=np.nan, right=np.nan)
        interp_list.append(pd.Series(interp_inc, name=y))
    interp_df = pd.concat(interp_list, axis=1)

    # --- Compute stats from training ---
    median_inc = interp_df.median(axis=1).values
    wettest_inc = interp_df.sum().idxmax()
    driest_inc = interp_df.sum().idxmin()
    wettest_curve = interp_df[wettest_inc].values
    driest_curve = interp_df[driest_inc].values

    # --- Validation data ---
    true_inc = np.interp(days_common, test["DayAfter"], test["IncAfterRef"], left=np.nan, right=np.nan)
    true_cum = np.nancumsum(true_inc)
    med_cum = np.nancumsum(median_inc)

    # --- Scale median to match final cumulative observed ---
    scale_factor = true_cum[-1] / med_cum[-1] if med_cum[-1] != 0 else np.nan
    med_scaled_cum = med_cum * scale_factor

    # --- Compute metrics ---
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

    # --- Plot individual year ---
    plt.figure(figsize=(8, 5))
    plt.plot(days_common, true_cum, "k--", label=f"Observed {test_year}")
    plt.plot(days_common, med_cum, "b", alpha=0.5, label="Median")
    plt.plot(days_common, med_scaled_cum, "b", lw=2, label="Scaled Median")
    plt.plot(days_common, np.nancumsum(wettest_curve), "g", alpha=0.4, label=f"Wettest {wettest_inc}")
    plt.plot(days_common, np.nancumsum(driest_curve), "r", alpha=0.4, label=f"Driest {driest_inc}")
    plt.xlabel("Days After Reference")
    plt.ylabel("Cumulative Increment (units)")
    plt.title(f"Forecast validation for {test_year}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"validation_{test_year}.png"))
    plt.close()

# --- Compile results ---
summary_df = pd.DataFrame(summary_list)
summary_df = summary_df.sort_values("Year")
summary_df.to_csv("summary_leave_one_out.csv", index=False)

# --- Plot metrics over time ---
plt.figure(figsize=(9, 5))
plt.plot(summary_df["Year"], summary_df["RMSE"], "o-", label="RMSE")
plt.xlabel("Validation Year")
plt.ylabel("RMSE")
plt.title("RMSE of Scaled Median Forecast by Year")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("RMSE_over_time.png", dpi=300)
plt.show()

# --- Optional: also plot R² and VolumeDiff ---
fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
ax[0].plot(summary_df["Year"], summary_df["R2"], "o-", color="teal")
ax[0].set_ylabel("R²")
ax[0].grid(True, alpha=0.3)

ax[1].plot(summary_df["Year"], summary_df["VolumeDiff"], "o-", color="darkorange")
ax[1].set_xlabel("Validation Year")
ax[1].set_ylabel("Final Volume Difference")
ax[1].grid(True, alpha=0.3)

fig.suptitle("Model Validation Metrics Over Time", fontsize=13)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig("Validation_Metrics_over_time.png", dpi=300)
plt.show()

print("✅ Done! Summary saved as 'summary_leave_one_out.csv' and plots exported.")