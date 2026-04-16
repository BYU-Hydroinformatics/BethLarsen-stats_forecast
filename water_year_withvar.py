import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. DATA PREPARATION
# ============================================================

def prepare_water_year_dataframe(df, date_col, volume_col):
    """
    Assigns water year, WY_Day, and cumulative volume to each row.
    Water year runs Oct 1 -> Sep 30.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df = df.sort_values(date_col)

    df["Water_Year"] = df[date_col].apply(
        lambda d: d.year + 1 if d.month >= 10 else d.year
    )

    wy_start_dates = (
        pd.to_datetime(df["Water_Year"] - 1, format="%Y")
        + pd.offsets.DateOffset(months=9)
    )
    df["WY_Day"] = (df[date_col] - wy_start_dates).dt.days + 1
    df["Cumulative_Volume"] = df.groupby("Water_Year")[volume_col].cumsum()

    return df


def compute_climate_features(df, date_col="Date",
                              precip_col="precipitation",
                              temp_col="temperature",
                              snow_col="snowfall",
                              melt_temp_threshold=0.0,
                              melt_rate=3.0,
                              rolling_window=30):
    """
    Computes derived climate features from raw ERA-5 daily data:

      - rolling_precip   : cumulative precipitation over last `rolling_window` days
      - rolling_snowfall : cumulative snowfall over last `rolling_window` days
      - rolling_temp     : average temperature over last `rolling_window` days
      - snowpack_index   : accumulated snowfall minus temperature-driven melt.
                           Melt on a given day = max(0, T - melt_temp_threshold)
                           * melt_rate, capped at current snowpack.
                           Resets to 0 each Oct 1 (water year boundary).

    Parameters
    ----------
    df : DataFrame with at least date_col, precip_col, temp_col, snow_col
    melt_temp_threshold : float
        Temperature (same units as your data) above which melt occurs.
        If your temperatures are in Celsius, 0.0 is a sensible default.
    melt_rate : float
        How many mm of snowpack melts per degree above threshold per day.
        Tune this to your basin — 3.0 mm/°C/day is a reasonable starting point.
    rolling_window : int
        Number of days for rolling statistics (default 30).

    Returns
    -------
    df with four new columns added.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df = df.sort_values(date_col)

    # Rolling statistics (min_periods=1 so early rows aren't dropped)
    df["rolling_precip"] = (
        df[precip_col].rolling(rolling_window, min_periods=1).sum()
    )
    df["rolling_snowfall"] = (
        df[snow_col].rolling(rolling_window, min_periods=1).sum()
    )
    df["rolling_temp"] = (
        df[temp_col].rolling(rolling_window, min_periods=1).mean()
    )

    # Snowpack index — computed day by day, resetting each water year
    df["Water_Year_Tmp"] = df[date_col].apply(
        lambda d: d.year + 1 if d.month >= 10 else d.year
    )

    snowpack_values = []
    snowpack = 0.0
    current_wy = None

    for _, row in df.iterrows():
        wy = row["Water_Year_Tmp"]

        # Reset snowpack at the start of each new water year
        if wy != current_wy:
            snowpack = 0.0
            current_wy = wy

        # Add today's snowfall
        snowpack += row[snow_col]

        # Subtract melt: positive melt only above threshold, capped at snowpack
        melt = max(0.0, (row[temp_col] - melt_temp_threshold) * melt_rate)
        snowpack = max(0.0, snowpack - melt)

        snowpack_values.append(snowpack)

    df["snowpack_index"] = snowpack_values
    df = df.drop(columns=["Water_Year_Tmp"])
    df = df.reset_index(drop=True)

    return df

def merge_climate_and_flow(flow_df, climate_df, date_col="Date"):
    flow_df = flow_df.copy()
    climate_df = climate_df.copy()

    flow_df[date_col] = pd.to_datetime(flow_df[date_col])
    climate_df[date_col] = pd.to_datetime(climate_df[date_col])

    merged = flow_df.merge(
        climate_df[[
            date_col,
            "rolling_precip",
            "rolling_snowfall",
            "rolling_temp",
            "snowpack_index"
        ]],
        on=date_col,
        how="left"
    )

    n_missing = merged[[
        "rolling_precip", "rolling_snowfall",
        "rolling_temp", "snowpack_index"
    ]].isna().any(axis=1).sum()

    if n_missing > 0:
        print(f"Warning: {n_missing} rows have missing climate data.")

    return merged

# ============================================================
# 2. CORE FORECAST FUNCTION
# ============================================================

def water_year_forecast_climate(
    df,
    forecast_date,
    date_col="Date",
    volume_col="Volume_m3",
    climate_features= ("rolling_precip", "rolling_temp"),
    slope_window=7,
    n_analogs=5,
    min_wy_day=14,
    forecast_horizon=90,
    analog_flow_weight=0.5,
):
    """
    90-day ahead cumulative volume forecast using:
      - Linear regression on climate features for volume projection
      - Combined flow-slope + climate similarity for analog selection
      - Analog-based shape for forecast curve timing

    Parameters
    ----------
    df : DataFrame
        Must contain flow columns, climate feature columns, WaterYear, WY_Day,
        CumVol_WY (added internally).
    forecast_date : datetime
    date_col : str
    volume_col : str
    climate_features : tuple of str
        Column names of derived climate features to use.
    slope_window : int
        Days of recent flow used to compute slope.
    n_analogs : int
        Number of analog years to use for shape.
    min_wy_day : int
        Minimum WY day before forecast is attempted.
    forecast_horizon : int
        Days ahead to forecast (default 90).
    analog_flow_weight : float
        Weight given to flow slope similarity vs climate similarity
        when scoring analogs. 0.5 = equal weight.

    Returns
    -------
    dict with observed_up_to_date, forecast_curve, true_curve,
    horizon_total_projection, regression_coefs, metadata
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    forecast_date = pd.to_datetime(forecast_date)

    # ---------------------------
    # 1. Water year structure
    # ---------------------------
    df["WaterYear"] = df[date_col].dt.year
    df.loc[df[date_col].dt.month >= 10, "WaterYear"] += 1

    wy_start = (
        pd.to_datetime(df["WaterYear"] - 1, format="%Y")
        + pd.DateOffset(months=9)
    )
    df["WY_Start"] = wy_start
    df["WY_Day"] = (df[date_col] - df["WY_Start"]).dt.days + 1
    df["CumVol_WY"] = df.groupby("WaterYear")[volume_col].cumsum()

    # ---------------------------
    # 2. Current state
    # ---------------------------
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

    # Check climate features are present
    missing_cols = [c for c in climate_features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing climate feature columns: {missing_cols}")

    # ---------------------------
    # 3. Historical pool (LOYO)
    # ---------------------------
    hist = df[df["WaterYear"] != test_wy].copy()

    # For each historical WY, get the state at current_wy_day
    hist_at_day = (
        hist[hist["WY_Day"] == current_wy_day]
        .drop_duplicates(subset="WaterYear")
        .set_index("WaterYear")
    )

    # Target: volume accumulated over the horizon window
    hist_at_horizon = (
        hist[hist["WY_Day"] == horizon_day]
        .drop_duplicates(subset="WaterYear")
        .set_index("WaterYear")["CumVol_WY"]
    )

    valid_wys = hist_at_day.index.intersection(hist_at_horizon.index)

    # Drop any years with missing climate features
    climate_feat_list = list(climate_features)
    valid_wys = valid_wys[
        hist_at_day.loc[valid_wys, climate_feat_list].notna().all(axis=1)
    ]

    if len(valid_wys) < 3:
        raise ValueError("Not enough historical years with complete data.")

    # Horizon volume = cumulative at horizon_day for each historical WY
    y_train = hist_at_horizon.loc[valid_wys].values

    # ---------------------------
    # 4. Linear regression for volume projection (LOYO-safe)
    # ---------------------------
    X_train = hist_at_day.loc[valid_wys, climate_feat_list].values

    # Current year's climate state at forecast date
    X_current = np.array([[test_row[f] for f in climate_feat_list]])

    if np.any(np.isnan(X_current)):
        raise ValueError(
            f"Current year (WY {test_wy}, day {current_wy_day}) has NaN climate "
            f"features: { {f: test_row[f] for f in climate_feat_list} }"
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_current_scaled = scaler.transform(X_current)


    reg = LinearRegression()
    reg.fit(X_train_scaled, y_train)
    horizon_total_projection = float(reg.predict(X_current_scaled)[0])
    hist_min = y_train.min()
    hist_max = y_train.max()
    horizon_total_projection = float(np.clip(horizon_total_projection, hist_min, hist_max))

    # Store regression coefficients for transparency
    regression_coefs = dict(zip(climate_feat_list, reg.coef_))
    regression_coefs["intercept"] = reg.intercept_

    # ---------------------------
    # 5. Combined analog scoring
    # ---------------------------
    # Flow slope similarity
    recent_current = df[
        (df["WaterYear"] == test_wy) &
        (df["WY_Day"] > current_wy_day - slope_window) &
        (df["WY_Day"] <= current_wy_day)
    ]
    current_slope = recent_current[volume_col].mean()

    hist_slopes = {}
    for wy in valid_wys:
        sub = hist[
            (hist["WaterYear"] == wy) &
            (hist["WY_Day"] > current_wy_day - slope_window) &
            (hist["WY_Day"] <= current_wy_day)
        ]
        if len(sub) == slope_window:
            hist_slopes[wy] = sub[volume_col].mean()

    hist_slopes = pd.Series(hist_slopes)
    valid_wys_slope = valid_wys.intersection(hist_slopes.index)

    slope_diff = (hist_slopes.loc[valid_wys_slope] - current_slope).abs()

    # Climate similarity: Euclidean distance in scaled feature space
    X_hist_scaled = pd.DataFrame(
        X_train_scaled,
        index=valid_wys,
        columns=climate_feat_list
    )
    climate_dist = X_hist_scaled.loc[valid_wys_slope].apply(
        lambda row: np.sqrt(np.sum((row.values - X_current_scaled[0]) ** 2)),
        axis=1
    )

    # Normalize both scores to [0, 1] so units don't bias the weighting
    def normalize(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    slope_score = normalize(slope_diff)
    climate_score = normalize(climate_dist)

    # Combined score — lower is better (more similar)
    combined_score = (
        analog_flow_weight * slope_score
        + (1 - analog_flow_weight) * climate_score
    )

    top_analogs = combined_score.nsmallest(n_analogs).index

    # ---------------------------
    # 6. Analog shapes (90-day window)
    # ---------------------------
    remainder_shapes = []

    for wy in top_analogs:
        sub = hist[hist["WaterYear"] == wy].copy()
        sub = sub[
            (sub["WY_Day"] >= current_wy_day) &
            (sub["WY_Day"] <= horizon_day)
        ]
        sub = sub.drop_duplicates(subset="WY_Day")

        if current_wy_day not in sub["WY_Day"].values:
            continue

        base_cum = sub.loc[
            sub["WY_Day"] == current_wy_day, "CumVol_WY"
        ].values[0]

        sub["Remainder"] = sub["CumVol_WY"] - base_cum
        total_remaining = sub["Remainder"].max()

        if total_remaining <= 0:
            continue

        sub["NormShape"] = sub["Remainder"] / total_remaining
        remainder_shapes.append(
            sub[["WY_Day", "NormShape"]]
            .drop_duplicates("WY_Day")
            .set_index("WY_Day")
        )

    if len(remainder_shapes) == 0:
        raise ValueError("No valid analog shapes found.")

    shape_df = pd.concat(remainder_shapes, axis=1)
    mean_shape = shape_df.mean(axis=1)

    horizon_volume = horizon_total_projection - current_cum
    forecast_cum = current_cum + horizon_volume * mean_shape

    forecast_curve = pd.DataFrame({
        "WY_Day": mean_shape.index,
        "ForecastCum": forecast_cum.values
    })

    # ---------------------------
    # 7. True curve for metrics
    # ---------------------------
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
        "regression_coefs": regression_coefs,
        "metadata": {
            "test_wy": test_wy,
            "forecast_date": forecast_date,
            "current_wy_day": current_wy_day,
            "horizon_day": horizon_day,
            "top_analogs": list(top_analogs),
        }
    }


# ============================================================
# 3. METRICS
# ============================================================

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def bias(a, b):
    return np.mean(a - b)

def nse(sim, obs):
    return 1 - np.sum((sim - obs)**2) / np.sum((obs - np.mean(obs))**2)


# ============================================================
# 4. LOYO EVALUATION
# ============================================================

def run_loyo_evaluation_climate(
    df,
    date_col="Date",
    volume_col="Volume_m3",
    climate_features=("rolling_precip", "rolling_snowfall",
                       "rolling_temp", "snowpack_index"),
    slope_window=7,
    n_analogs=5,
    forecast_horizon=90,
    analog_flow_weight=0.5,
    evaluation_wy_days=(30, 60, 90, 120, 150, 180)
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    df["WaterYear"] = df[date_col].dt.year
    df.loc[df[date_col].dt.month >= 10, "WaterYear"] += 1

    wy_start = (
        pd.to_datetime(df["WaterYear"] - 1, format="%Y")
        + pd.DateOffset(months=9)
    )
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
                fc = water_year_forecast_climate(
                    df,
                    forecast_date=forecast_date,
                    date_col=date_col,
                    volume_col=volume_col,
                    climate_features=climate_features,
                    slope_window=slope_window,
                    n_analogs=n_analogs,
                    forecast_horizon=forecast_horizon,
                    analog_flow_weight=analog_flow_weight,
                )

                forecast_curve = fc["forecast_curve"]
                true_curve = fc["true_curve"]

                merged = forecast_curve.merge(true_curve, on="WY_Day", how="inner")

                if len(merged) >= 5:
                    sim = merged["ForecastCum"].values
                    obs = merged["TrueCum"].values

                    results.append({
                        "WaterYear": test_wy,
                        "Evaluation_WY_Day": wy_day,
                        "Horizon_Day": wy_day + forecast_horizon,
                        "ForecastDate": forecast_date,
                        "TrueHorizonVol": true_curve["TrueCum"].max(),
                        "PredHorizonVol": fc["horizon_total_projection"],
                        "HorizonError": (
                            fc["horizon_total_projection"]
                            - true_curve["TrueCum"].max()
                        ),
                        "HorizonPctError": (
                            (fc["horizon_total_projection"]
                             - true_curve["TrueCum"].max())
                            / true_curve["TrueCum"].max() * 100
                        ),
                        "RMSE": rmse(sim, obs),
                        "Bias": bias(sim, obs),
                        "NSE": nse(sim, obs),
                        "TopAnalogs": str(fc["metadata"]["top_analogs"]),
                        "Coef_rolling_precip": fc["regression_coefs"].get("rolling_precip", np.nan),
                        "Coef_rolling_snowfall": fc["regression_coefs"].get("rolling_snowfall", np.nan),
                        "Coef_rolling_temp": fc["regression_coefs"].get("rolling_temp", np.nan),
                        "Coef_snowpack_index": fc["regression_coefs"].get("snowpack_index", np.nan),
                    })

                    forecast_store[(test_wy, wy_day)] = fc

            except Exception as e:
                import traceback
                traceback.print_exc()

    return pd.DataFrame(results), forecast_store


# ============================================================
# 5. MAIN
# ============================================================

# --- Load streamflow ---
flow_df = pd.read_csv(
    "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/"
    "Retrospective_Data/retrospective_trent.csv"
)
flow_df["Date"] = pd.to_datetime(flow_df["Date"])
flow_df = flow_df.sort_values("Date")

SECONDS_PER_DAY = 86400
flow_df["Volume_m3"] = flow_df["Discharge"] * SECONDS_PER_DAY

flow_df = prepare_water_year_dataframe(
    flow_df, date_col="Date", volume_col="Volume_m3"
)

# --- Load and prepare ERA-5 climate data ---
climate_raw = pd.read_csv(
    "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/Retrospective_Data/retrospective_era5/era5_trent_synthetic.csv"   # <-- update path as needed
)
climate_raw = climate_raw.rename(columns={"date": "Date"})  # add this

print("RAW CSV date range:", climate_raw["Date"].min(), climate_raw["Date"].max())
print("RAW CSV shape:", climate_raw.shape)
print("RAW CSV path used: era5_trent_synthetic.csv")  # just a label

climate_raw["Date"] = pd.to_datetime(climate_raw["Date"], format = "mixed")

print("RAW CSV date range:", climate_raw["Date"].min(), climate_raw["Date"].max())
print("RAW CSV shape:", climate_raw.shape)
print("RAW CSV path used: era5_trent_synthetic.csv")  # just a label

climate_df = compute_climate_features(
    climate_raw,
    date_col="Date",
    precip_col="precip_total",
    temp_col="temperature_2m",
    snow_col="snowfall",
    melt_temp_threshold=273.15,   # degrees C — adjust if temps are in Kelvin
    melt_rate=3.0,             # mm melt per degree above threshold per day
    rolling_window=7
)

print("Climate df length before merge:", len(climate_df))
print("NaT dates in climate:", climate_df["Date"].isna().sum())
print("Climate date min/max:", climate_df["Date"].min(), climate_df["Date"].max())
print("Climate index range:", climate_df.index.min(), climate_df.index.max())

flow_df["Date"] = flow_df["Date"].dt.normalize()
climate_df["Date"] = climate_df["Date"].dt.normalize()

# --- Merge flow and climate ---
df = merge_climate_and_flow(flow_df, climate_df, date_col="Date")





# Detailed merge diagnostic
print("Flow date dtype:", flow_df["Date"].dtype)
print("Climate date dtype:", climate_df["Date"].dtype)
print("Flow date sample:", flow_df["Date"].head(3).tolist())
print("Climate date sample:", climate_df["Date"].head(3).tolist())

# Check if any dates actually overlap
flow_dates = set(flow_df["Date"].dt.date)
climate_dates = set(climate_df["Date"].dt.date)
overlap = flow_dates.intersection(climate_dates)
print(f"\nFlow date range: {flow_df['Date'].min()} to {flow_df['Date'].max()}")
print(f"Climate date range: {climate_df['Date'].min()} to {climate_df['Date'].max()}")
print(f"Number of overlapping dates: {len(overlap)}")
print(f"Sample overlapping dates: {sorted(overlap)[:5]}")
print(f"Sample non-overlapping flow dates: {sorted(flow_dates - climate_dates)[:5]}")
print(df.head(10))
print(df.columns)

# --- Run LOYO evaluation ---
results_df, forecast_store = run_loyo_evaluation_climate(
    df,
    date_col="Date",
    volume_col="Volume_m3",
    climate_features=("rolling_precip", "rolling_snowfall",
                      "rolling_temp", "snowpack_index"),
    slope_window=7,
    n_analogs=5,
    forecast_horizon=90,
    analog_flow_weight=0.5,
    evaluation_wy_days=(30, 60, 90, 120, 150, 180)
)

print(results_df.head())
print(results_df.describe())

# --- Save results CSV ---
output_dir = (
    "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/"
    "Forecast_Results_Climate"
)
os.makedirs(output_dir, exist_ok=True)

results_df.to_csv(
    os.path.join(output_dir, "loyo_results_climate.csv"), index=False
)
print(f"Results saved to {output_dir}/loyo_results_climate.csv")

# --- Plot and save every water year ---
plots_dir = os.path.join(output_dir, "WY_Plots")
os.makedirs(plots_dir, exist_ok=True)

available_wys = sorted(set(wy for (wy, _) in forecast_store.keys()))

for wy in available_wys:
    wy_obs = df[df["Water_Year"] == wy]

    fig, ax = plt.subplots(figsize=(12, 6))

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
        analogs = fc["metadata"]["top_analogs"]

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

        line, = ax.plot(
            combined["WY_Day"],
            combined["Cum"],
            linestyle="--",
            label=f"Forecast @ Day {eval_day} (→{horizon_day}) | analogs: {analogs}"
        )

        ax.axvline(x=eval_day, color=line.get_color(), linestyle=":", alpha=0.4)
        ax.axvline(x=horizon_day, color=line.get_color(), linestyle=":", alpha=0.4)

    ax.set_xlabel("Water Year Day")
    ax.set_ylabel("Cumulative Volume (m³)")
    ax.set_title(f"90-Day Climate-Informed Forecast — WY {wy}")
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, f"WY_{wy}_climate_forecast.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot for WY {wy}")

print(f"All plots saved to {plots_dir}")
