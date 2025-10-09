import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
# Replace with your actual CSV path
df = pd.read_csv("/Users/beth/Downloads/retrospective_760686408.csv")

# --- Parse date and compute daily cumulative volumes per year ---
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["DOY"] = df["Date"].dt.dayofyear

# Convert flow (m³/s) to daily volume (m³/day)
df["Volume_m3"] = df["Flow_cms"] * 86400

# Compute cumulative volume for each year
df["Cumulative_Volume_m3"] = df.groupby("Year")["Volume_m3"].cumsum()

# --- Plot spaghetti plot ---
plt.figure(figsize=(12, 7))

for year, group in df.groupby("Year"):
    plt.plot(group["DOY"], group["Cumulative_Volume_m3"] / 1e6, alpha=0.4)  # convert to millions m³ for readability

# --- Plot formatting ---
plt.xlabel("Day of Year")
plt.ylabel("Cumulative Volume (million m³)")
plt.title("Cumulative Volume by Year (1940–Present)")
plt.grid(True, alpha=0.3)

# Optional: add average or percentile reference lines
mean_cum = (
    df.groupby("DOY")["Cumulative_Volume_m3"]
    .mean()
    .reset_index()
)
plt.plot(mean_cum["DOY"], mean_cum["Cumulative_Volume_m3"] / 1e6, color="black", lw=2, label="Average")
plt.legend()

plt.tight_layout()
plt.show()
