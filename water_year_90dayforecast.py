import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def prepare_water_year_dataframe(df, date_col, volume_col):
    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col])
    df[date_col] = df[date_col].dt.tz_localize(None)
    df = df.sort_values(date_col)

    df["Water_Year"] = df[date_col].apply(
        lambda d: d.year + 1 if d.month >= 10 else d.year
    )

    wy_start_dates = pd.to_datetime(
        df["Water_Year"] - 1, format="%Y"
    ) + pd.offsets.DateOffset(months=9)

    df["WY_Day"] = (df[date_col] - wy_start_dates).dt.days + 1

    df["Cumulative_Volume"] = (
        df.groupby("Water_Year")[volume_col].cumsum()
    )

    return df


def water_year_forecast_90day(
    df,
    forecast_date,
    date_col="Date",
    volume_col="Volume_m3",
    slope_window=7,
    n_analogs=5,
    min_wy_day=14,
    forecast_horizon=90,
):
    """
    90-day ahead water year cumulative forecast (LOYO-safe).

    Rather than forecasting to the end of the water year, this function
    forecasts cumulative volume for the next `forecast_horizon` days only.
    The observed cumulative curve still starts from Oct 1 (WY day 1).

    Parameters
    ----------
    df : DataFrame
    forecast_date : datetime
    date_col : str
    volume_col : str
    slope_window : int
    n_analogs : int
    min_wy_day : int
    forecast_horizon : int
        Number of days ahead to forecast (default 90)

    Returns
    -------
    dict with:
        observed_up_to_date   — full observed cumulative from WY start to forecast date
        forecast_curve        — projected cumulative for next `forecast_horizon` days
        horizon_total_projection — projected cumulative volume at end of horizon
        true_curve            — actual observed cumulative over the forecast window
        metadata
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    forecast_date = pd.to_datetime(forecast_date)

    # -----------------------------
    # 1. Assign Water Year + WY_Day
    # -----------------------------
    df["WaterYear"] = df[date_col].dt.year
    df.loc[df[date_col].dt.month >= 10, "WaterYear"] += 1

    wy_start = pd.to_datetime(df["WaterYear"] - 1, format="%Y") + pd.DateOffset(months=9)
    df["WY_Start"] = wy_start
    df["WY_Day"] = (df[date_col] - df["WY_Start"]).dt.days + 1

    df["CumVol_WY"] = df.groupby("WaterYear")[volume_col].cumsum()

    # -----------------------------
    # 2. Define Test WY + State
    # -----------------------------
    test_row = df[df[date_col] == forecast_date]

    if len(test_row) == 0:
        raise ValueError("forecast_date not found in dataset.")

    test_row = test_row.iloc[0]

    test_wy = test_row["WaterYear"]
    current_wy_day = test_row["WY_Day"]
    current_cum = test_row["CumVol_WY"]
    horizon_day = current_wy_day + forecast_horizon

    if current_wy_day < min_wy_day:
        raise ValueError("Too early in WY for stable forecast.")

    # -----------------------------
    # 3. Historical Pool (Exclude Test WY)
    # -----------------------------
    hist = df[df["WaterYear"] != test_wy].copy()

    # Use volume accumulated over the next `forecast_horizon` days as target
    # For each historical WY, compute cumulative at current_wy_day and at horizon_day
    hist_at_start = hist[hist["WY_Day"] == current_wy_day].set_index("WaterYear")["CumVol_WY"]
    hist_at_horizon = hist[hist["WY_Day"] == horizon_day].set_index("WaterYear")["CumVol_WY"]

    valid_wys = hist_at_start.index.intersection(hist_at_horizon.index)

    # Volume added over the horizon window in each historical year
    hist_horizon_volume = hist_at_horizon.loc[valid_wys] - hist_at_start.loc[valid_wys]
    hist_horizon_volume = hist_horizon_volume[hist_horizon_volume > 0]

    if len(hist_horizon_volume) == 0:
        raise ValueError("No valid historical horizon volumes found.")

    # Ratio of current cumulative to historical cumulative at same day
    fraction_of_hist = current_cum / hist_at_start.loc[hist_horizon_volume.index]
    fraction_of_hist = fraction_of_hist.replace([np.inf, -np.inf], np.nan).dropna()

    # Scale historical horizon volumes by how the current year compares so far
    projected_horizon_volumes = hist_horizon_volume.loc[fraction_of_hist.index] * fraction_of_hist
    horizon_volume_projection = projected_horizon_volumes.median()
    horizon_total_projection = current_cum + horizon_volume_projection

    # -----------------------------
    # 4. Slope-Based Analog Selection
    # -----------------------------
    recent_current = df[
        (df["WaterYear"] == test_wy) &
        (df["WY_Day"] > current_wy_day - slope_window) &
        (df["WY_Day"] <= current_wy_day)
    ]

    current_slope = recent_current[volume_col].mean()

    hist_slopes = {}

    for wy in hist["WaterYear"].unique():
        sub = hist[
            (hist["WaterYear"] == wy) &
            (hist["WY_Day"] > current_wy_day - slope_window) &
            (hist["WY_Day"] <= current_wy_day)
        ]
        if len(sub) == slope_window:
            hist_slopes[wy] = sub[volume_col].mean()

    hist_slopes = pd.Series(hist_slopes)

    slope_diff = (hist_slopes - current_slope).abs()
    top_analogs = slope_diff.nsmallest(n_analogs).index

    # -----------------------------
    # 5. Build Normalized Remainder Shapes (90-day window only)
    # -----------------------------
    remainder_shapes = []

    for wy in top_analogs:
        sub = hist[hist["WaterYear"] == wy].copy()

        # Only look at the forecast horizon window
        sub = sub[
            (sub["WY_Day"] >= current_wy_day) &
            (sub["WY_Day"] <= horizon_day)
        ]

        sub = sub.drop_duplicates(subset="WY_Day")

        if current_wy_day not in sub["WY_Day"].values:
            continue

        base_cum = sub.loc[sub["WY_Day"] == current_wy_day, "CumVol_WY"].values[0]

        sub["Remainder"] = sub["CumVol_WY"] - base_cum
        total_remaining = sub["Remainder"].max()

        if total_remaining <= 0:
            continue

        sub["NormShape"] = sub["Remainder"] / total_remaining
        remainder_shapes.append(
            sub[["WY_Day", "NormShape"]].drop_duplicates("WY_Day").set_index("WY_Day")
        )

    if len(remainder_shapes) == 0:
        raise ValueError("No valid analog shapes found.")

    shape_df = pd.concat(remainder_shapes, axis=1)
    mean_shape = shape_df.mean(axis=1)

    # Scale the shape to match the projected horizon volume
    forecast_cum = current_cum + horizon_volume_projection * mean_shape

    forecast_curve = pd.DataFrame({
        "WY_Day": mean_shape.index,
        "ForecastCum": forecast_cum.values
    })

    # -----------------------------
    # 6. True Curve Over Horizon (For Metrics)
    # -----------------------------
    true_wy = df[df["WaterYear"] == test_wy].copy()

    true_future = true_wy[
        (true_wy["WY_Day"] >= current_wy_day) &
        (true_wy["WY_Day"] <= horizon_day)
    ].copy()

    true_curve = true_future[["WY_Day", "CumVol_WY"]].copy()
    true_curve.rename(columns={"CumVol_WY": "TrueCum"}, inplace=True)

    observed_up_to_date = true_wy[
        true_wy["WY_Day"] <= current_wy_day
    ][[date_col, "WY_Day", "CumVol_WY"]]

    return {
        "observed_up_to_date": observed_up_to_date,
        "forecast_curve": forecast_curve,
        "true_curve": true_curve,
        "horizon_total_projection": horizon_total_projection,
        "metadata": {
            "test_wy": test_wy,
            "forecast_date": forecast_date,
            "current_wy_day": current_wy_day,
            "horizon_day": horizon_day,
        }
    }


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def bias(a, b):
    return np.mean(a - b)


def nse(sim, obs):
    return 1 - np.sum((sim - obs)**2) / np.sum((obs - np.mean(obs))**2)


def run_loyo_evaluation_90day(
    df,
    date_col="Date",
    volume_col="Volume_m3",
    slope_window=7,
    n_analogs=5,
    forecast_horizon=90,
    evaluation_wy_days=(45, 90, 135, 180, 225, 270)
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    df["WaterYear"] = df[date_col].dt.year
    df.loc[df[date_col].dt.month >= 10, "WaterYear"] += 1

    wy_start = pd.to_datetime(df["WaterYear"] - 1, format="%Y") + pd.DateOffset(months=9)
    df["WY_Day"] = (df[date_col] - wy_start).dt.days + 1

    all_wys = sorted(df["WaterYear"].unique())

    results = []
    forecast_store = {}

    for test_wy in all_wys:

        wy_data = df[df["WaterYear"] == test_wy]

        if wy_data["WY_Day"].max() < 300:
            continue

        for wy_day in evaluation_wy_days:

            eval_row = wy_data[wy_data["WY_Day"] == wy_day]

            if len(eval_row) == 0:
                continue

            forecast_date = eval_row[date_col].values[0]

            try:
                fc = water_year_forecast_90day(
                    df,
                    forecast_date=forecast_date,
                    date_col=date_col,
                    volume_col=volume_col,
                    slope_window=slope_window,
                    n_analogs=n_analogs,
                    forecast_horizon=forecast_horizon,
                )

                forecast_curve = fc["forecast_curve"]
                true_curve = fc["true_curve"]

                merged = forecast_curve.merge(true_curve, on="WY_Day", how="inner")

                if len(merged) >= 5:
                    sim = merged["ForecastCum"].values
                    obs = merged["TrueCum"].values

                    true_horizon_vol = true_curve["TrueCum"].max() - true_curve["TrueCum"].min()
                    pred_horizon_vol = fc["horizon_total_projection"] - true_curve["TrueCum"].min()

                    results.append({
                        "WaterYear": test_wy,
                        "Evaluation_WY_Day": wy_day,
                        "Horizon_Day": wy_day + forecast_horizon,
                        "ForecastDate": forecast_date,
                        "TrueHorizonVol": true_curve["TrueCum"].max(),
                        "PredHorizonVol": fc["horizon_total_projection"],
                        "HorizonError": fc["horizon_total_projection"] - true_curve["TrueCum"].max(),
                        "HorizonPctError": (
                            (fc["horizon_total_projection"] - true_curve["TrueCum"].max())
                            / true_curve["TrueCum"].max() * 100
                        ),
                        "RMSE": rmse(sim, obs),
                        "Bias": bias(sim, obs),
                        "NSE": nse(sim, obs)
                    })

                    forecast_store[(test_wy, wy_day)] = fc

            except Exception as e:
                import traceback
                traceback.print_exc()

    return pd.DataFrame(results), forecast_store


# -----------------------------
# Main
# -----------------------------
df = pd.read_csv('/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/Retrospective_Data/retrospective_rhine.csv')

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

SECONDS_PER_DAY = 86400
df["Volume_m3"] = df["Discharge"] * SECONDS_PER_DAY

df = prepare_water_year_dataframe(
    df,
    date_col="Date",
    volume_col="Volume_m3"
)

results_df, forecast_store = run_loyo_evaluation_90day(
    df,
    date_col="Date",
    volume_col="Volume_m3",
    slope_window=7,
    n_analogs=5,
    forecast_horizon=90,
    evaluation_wy_days= (45, 90, 135, 180, 225, 270)
)

print(results_df.head())
print(results_df.describe())

# -----------------------------
# Save results CSV
# -----------------------------
output_dir = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/Forecast_Results_90day/rhine"
os.makedirs(output_dir, exist_ok=True)

results_df.to_csv(os.path.join(output_dir, "loyo_results_90day.csv"), index=False)
print(f"Results saved to {output_dir}/loyo_results_90day.csv")

# -----------------------------
# Plot and save every water year
# -----------------------------
plots_dir = os.path.join(output_dir, "WY_Plots")
os.makedirs(plots_dir, exist_ok=True)

available_wys = sorted(set(wy for (wy, _) in forecast_store.keys()))

for wy in available_wys:
    wy_obs = df[df["Water_Year"] == wy]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Full observed cumulative curve for context
    ax.plot(
        wy_obs["WY_Day"],
        wy_obs["Cumulative_Volume"],
        label="Observed (full year)",
        linewidth=2,
        color="black"
    )

    wy_keys_sorted = sorted(day for (y, day) in forecast_store.keys() if y == wy)

    for eval_day in wy_keys_sorted:
        fc = forecast_store[(wy, eval_day)]
        horizon_day = fc["metadata"]["horizon_day"]

        observed = fc["observed_up_to_date"]
        forecast_curve = fc["forecast_curve"]

        # Observed segment up to forecast date
        obs_segment = observed[["WY_Day", "CumVol_WY"]].rename(
            columns={"CumVol_WY": "Cum"}
        )

        # Forecast segment (90 days ahead only)
        fcast_segment = forecast_curve.rename(columns={"ForecastCum": "Cum"})
        fcast_segment = fcast_segment[
            fcast_segment["WY_Day"] > obs_segment["WY_Day"].max()
        ]

        combined = pd.concat([obs_segment, fcast_segment], ignore_index=True)

        line, = ax.plot(
            combined["WY_Day"],
            combined["Cum"],
            linestyle="--",
            label=f"Forecast @ WY Day {eval_day} (→ Day {horizon_day})"
        )

        # Mark start and end of forecast window
        ax.axvline(x=eval_day, color=line.get_color(), linestyle=":", alpha=0.4)
        ax.axvline(x=horizon_day, color=line.get_color(), linestyle=":", alpha=0.4)

    ax.set_xlabel("Water Year Day")
    ax.set_ylabel("Cumulative Volume (m³)")
    ax.set_title(f"90-Day Ahead Cumulative Volume Forecast — WY {wy}")
    ax.legend(fontsize=8)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, f"WY_{wy}_90day_forecast.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot for WY {wy}")

print(f"All plots saved to {plots_dir}")