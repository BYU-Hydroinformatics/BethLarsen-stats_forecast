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

    hist_at_day = hist[hist["WY_Day"] == current_wy_day]

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

        if current_wy_day not in sub["WY_Day"].values:
            continue

        base_cum = sub.loc[sub["WY_Day"] == current_wy_day, "CumVol_WY"].values[0]

        sub["Remainder"] = sub["CumVol_WY"] - base_cum
        total_remaining = sub["Remainder"].max()

        if total_remaining <= 0:
            continue

        sub["NormShape"] = sub["Remainder"] / total_remaining
        remainder_shapes.append(
            sub[["WY_Day", "NormShape"]].set_index("WY_Day")
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


def plot_water_year_forecasts(df, results_df, target_wy):
    # Extract observed cumulative curve for that WY
    wy_obs = df[df["Water_Year"] == target_wy]

    plt.figure()

    # Plot observed cumulative
    plt.plot(
        wy_obs["WY_Day"],
        wy_obs["Cumulative_Volume"],
        label="Observed",
        linewidth=2
    )

    # Extract forecast results for this WY
    wy_forecasts = results_df[
        results_df["WaterYear"] == target_wy
        ]

    print("Unique WYs in results_df:")
    print(results_df["WaterYear"].unique())

    print(f"\nRows for WY {target_wy}:")
    print(results_df[results_df["WaterYear"] == target_wy])

    # Plot forecast curves for each evaluation day
    for _, row in wy_forecasts.iterrows():
        eval_day = row["Evaluation_WY_Day"]
        forecast_total = row["Forecast_Total"]

        # Get observed cumulative up to forecast day
        observed_to_date = wy_obs[
            wy_obs["WY_Day"] <= eval_day
            ]

        current_cum = observed_to_date["Cumulative_Volume"].iloc[-1]

        # Forecast remainder
        remaining_days = wy_obs[
            wy_obs["WY_Day"] > eval_day
            ]["WY_Day"]

        # Build linear forecast remainder curve
        forecast_remainder = np.linspace(
            current_cum,
            forecast_total,
            len(remaining_days)
        )

        # Combine
        forecast_curve_x = np.concatenate([
            observed_to_date["WY_Day"].values,
            remaining_days.values
        ])

        forecast_curve_y = np.concatenate([
            observed_to_date["Cumulative_Volume"].values,
            forecast_remainder
        ])

        plt.plot(
            forecast_curve_x,
            forecast_curve_y,
            linestyle="--",
            label=f"Forecast @ WY Day {int(eval_day)}"
        )

    plt.xlabel("Water Year Day")
    plt.ylabel("Cumulative Volume (m³)")
    plt.title(f"Cumulative Volume Forecast — WY {target_wy}")
    plt.legend()
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
    evaluation_wy_days=(30, 60, 90, 120, 150, 180)
):

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Assign water year once here (so we know which WYs exist)
    df["WaterYear"] = df[date_col].dt.year
    df.loc[df[date_col].dt.month >= 10, "WaterYear"] += 1

    all_wys = sorted(df["WaterYear"].unique())

    results = []

    for test_wy in all_wys:

        wy_data = df[df["WaterYear"] == test_wy]
        print(test_wy, wy_data["WY_Day"].max())

        # Skip incomplete WY (e.g., current live WY)
        if wy_data["WY_Day"].max() < 300:
            continue

        for wy_day in evaluation_wy_days:

            eval_row = wy_data[wy_data["WY_Day"] == wy_day]

            if len(eval_row) == 0:
                continue

            forecast_date = eval_row[date_col].values[0]

            try:
                fc = water_year_forecast_loyo(
                    df,
                    forecast_date=forecast_date,
                    date_col=date_col,
                    volume_col=volume_col,
                    slope_window=slope_window,
                    n_analogs=n_analogs
                )
            except:
                continue

            forecast_curve = fc["forecast_curve"]
            true_curve = fc["true_curve"]

            merged = forecast_curve.merge(
                true_curve,
                on="WY_Day",
                how="inner"
            )

            if len(merged) < 5:
                continue

            sim = merged["ForecastCum"].values
            obs = merged["TrueCum"].values

            # Final total metrics
            true_final = true_curve["TrueCum"].max()
            pred_final = fc["final_total_projection"]

            results.append({
                "WaterYear": test_wy,
                "WY_Day": wy_day,
                "ForecastDate": forecast_date,
                "FinalTrue": true_final,
                "FinalPred": pred_final,
                "FinalError": pred_final - true_final,
                "FinalPctError": (pred_final - true_final) / true_final * 100,
                "RMSE": rmse(sim, obs),
                "Bias": bias(sim, obs),
                "NSE": nse(sim, obs)
            })

    return pd.DataFrame(results)


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

results_df = run_loyo_evaluation(
    df,
    date_col="Date",
    volume_col="Volume_m3",
    slope_window=7,
    n_analogs=5,
    evaluation_wy_days=(30, 60, 90, 120, 150, 180)
)

print(results_df.head())
print(results_df.describe())


plot_water_year_forecasts(df, results_df, target_wy=1978)