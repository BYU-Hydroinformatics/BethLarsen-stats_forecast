# step_3b.py
# Compare predictive power of different rolling windows (past vs future flow volumes)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# === USER SETTINGS ===
csv_path = "/Users/bethlarsen/Documents/Hydro Research/geoglows_stat_forecast/flow_data.csv"
ref_month = 6  # reference month (e.g., June)
ref_day = 15   # reference day
window_months_list = [3, 6, 9, 12]  # months of past/future to test
date_col = "Date"
flow_col = "Flow_m3s"

# === LOAD AND PREP DATA ===
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col])
df = df.set_index(date_col).sort_index()

# Convert flow to daily volume (m³/day)
df["Volume_m3"] = df[flow_col] * 24 * 60 * 60

results = []

# === MAIN LOOP ===
for window_months in window_months_list:
    records = []
    for year in df.index.year.unique():
        ref_date = pd.Timestamp(f"{year}-{ref_month:02d}-{ref_day:02d}")
        start_past = ref_date - pd.DateOffset(months=window_months)
        end_future = ref_date + pd.DateOffset(months=window_months)

        # Skip if out of bounds
        if start_past < df.index.min() or end_future > df.index.max():
            continue

        past_sum = df.loc[start_past:ref_date, "Volume_m3"].sum()
        future_sum = df.loc[ref_date:end_future, "Volume_m3"].sum()
        records.append({"Year": year, "Past_m3": past_sum, "Future_m3": future_sum})

    # Convert to DataFrame for this window
    rec_df = pd.DataFrame(records)
    if rec_df.empty:
        continue

    X = rec_df[["Past_m3"]]
    y = rec_df["Future_m3"]

    # Linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)

    results.append({
        "Window_months": window_months,
        "Intercept": model.intercept_,
        "Slope": model.coef_[0],
        "R2": r2
    })

    # Plot regression for this window
    plt.figure(figsize=(6, 5))
    plt.scatter(X, y, color="blue", alpha=0.7)
    plt.plot(X, y_pred, color="red", label=f"y = {model.coef_[0]:.3f}x + {model.intercept_:.2e}")
    plt.xlabel(f"Past {window_months} Months Volume (m³)")
    plt.ylabel(f"Future {window_months} Months Volume (m³)")
    plt.title(f"{window_months}-Month Window Regression (R² = {r2:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === SUMMARIZE RESULTS ===
summary_df = pd.DataFrame(results)
best = summary_df.loc[summary_df["R2"].idxmax()]

print("\n=== Regression Summary ===")
print(summary_df.round(3))
print("\nBest window length:")
print(best.round(3))

# === OPTIONAL: Plot R² vs window length ===
plt.figure(figsize=(6, 4))
plt.plot(summary_df["Window_months"], summary_df["R2"], marker="o", linewidth=2)
plt.title("Predictive Skill vs. Window Length")
plt.xlabel("Window Length (months)")
plt.ylabel("R²")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()