import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import sys


# ── Configuration ────────────────────────────────────────────────────────────
CSV_PATH = '/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/water_year_forecast/ohio/loyo_results_ohio.csv'          # <-- change to your CSV file path

COL_YEAR = "WaterYear"             # <-- adjust if your column names differ
COL_DAY  = "Evaluation_WY_Day"
COL_NSE  = "NSE"

# Discrete day-of-water-year values and their display order
DAY_OPTIONS = [30, 60, 90, 120, 150, 180, 270]
# ─────────────────────────────────────────────────────────────────────────────


def load_data(path):
    df = pd.read_csv(path)

    # Normalize column names (strip whitespace, lowercase) for robustness
    df.columns = df.columns.str.strip().str.lower()
    col_map = {
        COL_YEAR.lower(): "year",
        COL_DAY.lower():  "day",
        COL_NSE.lower():  "NSE",
    }
    df = df.rename(columns=col_map)

    # Keep only rows whose 'day' value is one of the expected discrete options
    df = df[df["day"].isin(DAY_OPTIONS)].copy()
    df["day"] = pd.Categorical(df["day"], categories=DAY_OPTIONS, ordered=True)
    df = df.sort_values(["day", "year"])
    return df


def make_plot(df):
    # One distinct color per day-of-water-year group
    cmap = matplotlib.colormaps["tab10"].resampled(len(DAY_OPTIONS))
    color_map = {day: cmap(i) for i, day in enumerate(DAY_OPTIONS)}

    fig, ax = plt.subplots(figsize=(11, 6))

    for day in DAY_OPTIONS:
        subset = df[df["day"] == day].sort_values("year")
        if subset.empty:
            continue
        color = color_map[day]
        # Connected line
        ax.plot(subset["year"], subset["NSE"],
                color=color, linewidth=1.4, alpha=0.7)
        # Scatter dots on top
        ax.scatter(subset["year"], subset["NSE"],
                   color=color, s=55, zorder=5,
                   label=f"Day {day}")

    ax.set_xlabel("Water Year", fontsize=12)
    ax.set_ylabel("NSE", fontsize=12)
    ax.set_title("NSE by Water Year\n(color = day of water year)", fontsize=14)
    ax.legend(title="Day of\nWater Year", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(-1, 1)
    years = sorted(df["year"].unique())
    ax.set_xticks([y for y in years if y % 5 == 0])
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("nse_plot.png", dpi=150, bbox_inches="tight")
    print("Plot saved to nse_plot.png")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    df = load_data(path)
    if df.empty:
        print("No data found after filtering. Check column names and DAY_OPTIONS.")
    else:
        make_plot(df)
