import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def prepare_water_year_dataframe(df, date_col, volume_col):
    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col])

    # Remove timezone if present
    df[date_col] = df[date_col].dt.tz_localize(None)
    df = df.sort_values(date_col)

    # Water Year
    df["Water_Year"] = df[date_col].apply(
        lambda d: d.year + 1 if d.month >= 10 else d.year
    )

    # Water Year Day
    wy_start_dates = pd.to_datetime(
        df["Water_Year"] - 1, format="%Y"
    ) + pd.offsets.DateOffset(months=9)  # Oct 1

    df["WY_Day"] = (df[date_col] - wy_start_dates).dt.days + 1

    # Cumulative Volume
    df["Cumulative_Volume"] = (
        df.groupby("Water_Year")[volume_col]
        .cumsum()
    )

    return df


def water_year_forecast_loyo(
    df,
    forecast_date,
    date_col="Date",
    volume_col="Volume_m3",
    slope_window=7,
    n_analogs=5,
    min_wy_day=14,
):
    """
    Water-year cumulative forecast (LOYO-safe).

    Parameters
    ----------
    df : DataFrame
        Must contain full multi-year daily data.
    forecast_date : datetime
        Evaluation date inside the test WY.
    date_col : str
    volume_col : str
    slope_window : int
    n_analogs : int
    min_wy_day : int

    Returns
    -------
    dict with:
        observed_up_to_date
        forecast_curve
        final_total_projection
        true_remaining_curve
        metadata
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    forecast_date = pd.to_datetime(forecast_date)

    # -----------------------------
    # 1. Assign Water Year
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

    if current_wy_day < min_wy_day:
        raise ValueError("Too early in WY for stable forecast.")

    # -----------------------------
    # 3. Historical Pool (Exclude Test WY)
    # -----------------------------
    hist = df[df["WaterYear"] != test_wy].copy()

    wy_totals = hist.groupby("WaterYear")[volume_col].sum()


    hist_at_day = hist[hist["WY_Day"] == current_wy_day].drop_duplicates(subset="WaterYear")

    hist_cum = hist_at_day.set_index("WaterYear")["CumVol_WY"]

    valid_wys = hist_cum.index.intersection(wy_totals.index)

    fraction_complete = hist_cum.loc[valid_wys] / wy_totals.loc[valid_wys]
    fraction_complete = fraction_complete.replace(0, np.nan).dropna()

    projected_totals = current_cum / fraction_complete

    final_total_projection = projected_totals.median()

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
    # 5. Build Normalized Remainder Shapes
    # -----------------------------
    remainder_shapes = []

    for wy in top_analogs:
        sub = hist[hist["WaterYear"] == wy].copy()
        sub = sub[sub["WY_Day"] >= current_wy_day]

        # FIX: drop duplicate WY_Day rows before indexing
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

        remaining_volume = final_total_projection - current_cum
        forecast_cum = current_cum + remaining_volume * mean_shape

        forecast_curve = pd.DataFrame({
            "WY_Day": mean_shape.index,
            "ForecastCum": forecast_cum.values
        })

    # -----------------------------
    # 6. True Remaining Curve (For Metrics)
    # -----------------------------
    true_wy = df[df["WaterYear"] == test_wy].copy()

    true_future = true_wy[true_wy["WY_Day"] >= current_wy_day].copy()

    true_curve = true_future[["WY_Day", "CumVol_WY"]].copy()
    true_curve.rename(columns={"CumVol_WY": "TrueCum"}, inplace=True)

    observed_up_to_date = true_wy[
        true_wy["WY_Day"] <= current_wy_day
    ][[date_col, "WY_Day", "CumVol_WY"]]

    return {
        "observed_up_to_date": observed_up_to_date,
        "forecast_curve": forecast_curve,
        "true_curve": true_curve,
        "final_total_projection": final_total_projection,
        "metadata": {
            "test_wy": test_wy,
            "forecast_date": forecast_date,
            "current_wy_day": current_wy_day,
        }
    }


def plot_water_year_forecasts(df, forecast_store, target_wy):
    """
    Plot observed cumulative curve and analog-shaped forecast curves
    for a given water year.

    Parameters
    ----------
    df : DataFrame
        Prepared dataframe with Water_Year, WY_Day, Cumulative_Volume columns.
    forecast_store : dict
        Keyed by (WaterYear, evaluation_wy_day), values are the full result
        dicts returned by water_year_forecast_loyo.
    target_wy : int
        The water year to plot.
    """
    wy_obs = df[df["Water_Year"] == target_wy]

    plt.figure(figsize=(12, 6))

    # Plot true observed cumulative curve for full WY
    plt.plot(
        wy_obs["WY_Day"],
        wy_obs["Cumulative_Volume"],
        label="Observed",
        linewidth=2,
        color="black"
    )

    # Plot each stored forecast curve for this WY
    wy_keys = [(wy, day) for (wy, day) in forecast_store if wy == target_wy]
    wy_keys_sorted = sorted(wy_keys, key=lambda x: x[1])

    for (wy, eval_day) in wy_keys_sorted:
        fc = forecast_store[(wy, eval_day)]

        observed = fc["observed_up_to_date"]
        forecast_curve = fc["forecast_curve"]

        # Combine observed portion + forecast remainder into one curve
        obs_segment = observed[["WY_Day", "CumVol_WY"]].rename(
            columns={"CumVol_WY": "Cum"}
        )
        fcast_segment = forecast_curve.rename(
            columns={"ForecastCum": "Cum"}
        )

        # Drop the overlap point from forecast so they join cleanly
        fcast_segment = fcast_segment[fcast_segment["WY_Day"] > obs_segment["WY_Day"].max()]

        combined = pd.concat([obs_segment, fcast_segment], ignore_index=True)

        plt.plot(
            combined["WY_Day"],
            combined["Cum"],
            linestyle="--",
            label=f"Forecast @ WY Day {eval_day}"
        )

        # Mark the evaluation point
        plt.axvline(x=eval_day, color="grey", linestyle=":", alpha=0.4)

    plt.xlabel("Water Year Day")
    plt.ylabel("Cumulative Volume (m³)")
    plt.title(f"Cumulative Volume Forecast — WY {target_wy}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def bias(a, b):
    return np.mean(a - b)


def nse(sim, obs):
    return 1 - np.sum((sim - obs)**2) / np.sum((obs - np.mean(obs))**2)


def run_loyo_evaluation(
    df,
    date_col="Date",
    volume_col="Volume_m3",
    slope_window=7,
    n_analogs=5,
    evaluation_wy_days=(30, 60, 90, 120, 150, 180, 270)
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

            try:                                          # <-- try
                fc = water_year_forecast_loyo(
                    df,
                    forecast_date=forecast_date,
                    date_col=date_col,
                    volume_col=volume_col,
                    slope_window=slope_window,
                    n_analogs=n_analogs
                )

                forecast_curve = fc["forecast_curve"]
                true_curve = fc["true_curve"]

                merged = forecast_curve.merge(
                    true_curve,
                    on="WY_Day",
                    how="inner"
                )

                if len(merged) >= 5:
                    sim = merged["ForecastCum"].values
                    obs = merged["TrueCum"].values

                    true_final = true_curve["TrueCum"].max()
                    pred_final = fc["final_total_projection"]

                    results.append({
                        "WaterYear": test_wy,
                        "Evaluation_WY_Day": wy_day,
                        "ForecastDate": forecast_date,
                        "FinalTrue": true_final,
                        "Forecast_Total": pred_final,
                        "FinalError": pred_final - true_final,
                        "FinalPctError": (pred_final - true_final) / true_final * 100,
                        "RMSE": rmse(sim, obs),
                        "Bias": bias(sim, obs),
                        "NSE": nse(sim, obs)
                    })

                    forecast_store[(test_wy, wy_day)] = fc

            #except Exception as e:                        # <-- except at same level as try
                #print(f"Skipping WY {test_wy} day {wy_day}: {e}")
            except Exception as e:
                import traceback
                traceback.print_exc()

    return pd.DataFrame(results), forecast_store


# -----------------------------
# Main
# -----------------------------
df = pd.read_csv("/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/Retrospective_Data/retrospective_ohio.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

SECONDS_PER_DAY = 86400
df["Volume_m3"] = df["Discharge"] * SECONDS_PER_DAY

df = prepare_water_year_dataframe(
    df,
    date_col="Date",
    volume_col="Volume_m3"
)

# run_loyo_evaluation now returns both the results table and the forecast store
results_df, forecast_store = run_loyo_evaluation(
    df,
    date_col="Date",
    volume_col="Volume_m3",
    slope_window=7,
    n_analogs=5,
    evaluation_wy_days=(30, 60, 90, 120, 150, 180, 270)
)

print(results_df.head())
print(results_df.describe())


import os

# -----------------------------
# Save results_df to CSV
# -----------------------------
output_dir = "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/water_year_forecast/ohio"
os.makedirs(output_dir, exist_ok=True)

results_df.to_csv(os.path.join(output_dir, "loyo_results.csv"), index=False)
print(f"Results saved to {output_dir}/loyo_results.csv")

# -----------------------------
# Plot and save every water year
# -----------------------------
plots_dir = os.path.join(output_dir, "WY_Plots")
os.makedirs(plots_dir, exist_ok=True)

available_wys = sorted(set(wy for (wy, _) in forecast_store.keys()))

for wy in available_wys:
    wy_obs = df[df["Water_Year"] == wy]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Observed full curve
    ax.plot(
        wy_obs["WY_Day"],
        wy_obs["Cumulative_Volume"],
        label="Observed",
        linewidth=2,
        color="black"
    )

    # Forecast curves for each evaluation day
    wy_keys_sorted = sorted((day for (y, day) in forecast_store.keys() if y == wy))

    for eval_day in wy_keys_sorted:
        fc = forecast_store[(wy, eval_day)]

        observed = fc["observed_up_to_date"]
        forecast_curve = fc["forecast_curve"]

        obs_segment = observed[["WY_Day", "CumVol_WY"]].rename(
            columns={"CumVol_WY": "Cum"}
        )
        fcast_segment = forecast_curve.rename(columns={"ForecastCum": "Cum"})
        fcast_segment = fcast_segment[
            fcast_segment["WY_Day"] > obs_segment["WY_Day"].max()
        ]

        combined = pd.concat([obs_segment, fcast_segment], ignore_index=True)

        ax.plot(
            combined["WY_Day"],
            combined["Cum"],
            linestyle="--",
            label=f"Forecast @ WY Day {eval_day}"
        )
        ax.axvline(x=eval_day, color="grey", linestyle=":", alpha=0.4)

    ax.set_xlabel("Water Year Day")
    ax.set_ylabel("Cumulative Volume (m³)")
    ax.set_title(f"Cumulative Volume Forecast — WY {wy}")
    ax.legend(fontsize=8)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, f"WY_{wy}_forecast.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot for WY {wy}")

print(f"All plots saved to {plots_dir}")