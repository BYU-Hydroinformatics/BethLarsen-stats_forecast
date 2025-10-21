import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === USER SETTINGS ===
csv_path = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/retrospective_760706416.csv"
date_col = "Date"
flow_col = "Flow_cms"
ref_month, ref_day = 3, 15  # March 15
target_year = 2024  # year to forecast
normalize = False  # normalization not needed for regression

# === 1. Load and prepare ===
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
df = df.set_index(date_col).sort_index().asfreq("D")

# === 2. Compute cumulative volumes per day ===
df["Volume_m3"] = df[flow_col] * 86400
df["Cumulative_m3"] = df["Volume_m3"].cumsum()

# === 3. Loop through years to get YTD and full-year totals ===
records = []

for year in df.index.year.unique():
    start_year = pd.Timestamp(f"{year}-01-01")
    end_year = pd.Timestamp(f"{year}-12-31")
    ref_date = pd.Timestamp(f"{year}-{ref_month:02d}-{ref_day:02d}")

    if end_year not in df.index or ref_date not in df.index:
        continue  # skip incomplete years

    # Year-to-date cumulative volume
    ytd = df.loc[start_year:ref_date, "Volume_m3"].sum()
    # Full-year cumulative volume
    total = df.loc[start_year:end_year, "Volume_m3"].sum()
    records.append({"Year": year, "YTD_m3": ytd, "Total_m3": total})

data = pd.DataFrame(records)

# === 4. Fit regression ===
X = data[["YTD_m3"]]
y = data["Total_m3"]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

a, b = model.intercept_, model.coef_[0]
print(f"Regression equation: Total = {a:.2e} + {b:.3f} * YTD")
print(f"R² = {r2:.3f}")

# === 5. Plot ===
plt.figure(figsize=(8, 6))
plt.scatter(data["YTD_m3"], data["Total_m3"], alpha=0.6, label="Historical years")
plt.plot(X, y_pred, color="red", label=f"Fit: Total = {b:.2f}×YTD + {a / 1e9:.1f}B\nR²={r2:.2f}")

# Highlight target year if available
if target_year in data["Year"].values:
    ytd_target = data.loc[data["Year"] == target_year, "YTD_m3"].values[0]
    total_pred = model.predict([[ytd_target]])[0]
    plt.scatter(ytd_target, total_pred, color="green", s=100, zorder=5,
                label=f"Forecast {target_year}\nPredicted total = {total_pred / 1e9:.2f}×10⁹ m³")

plt.xlabel("YTD cumulative volume (m³)")
plt.ylabel("Full-year cumulative volume (m³)")
plt.title(f"Forecast Regression: Total Volume vs YTD ({ref_month:02d}-{ref_day:02d})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()