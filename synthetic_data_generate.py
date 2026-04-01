"""
synthetic_era5_generator.py

Takes 1-2 years of real ERA-5 data and generates ~78 years of synthetic
data by resampling from the real data with added noise. The output is a
single CSV in the same format as the real ERA-5 data, covering a date
range that matches your streamflow record.

Usage:
    - Update the file paths and date range at the bottom of the script
    - Run the script
    - The output CSV can be dropped straight into the climate forecast script
"""

import pandas as pd
import numpy as np
import os


def generate_synthetic_era5(
        real_climate_path,
        output_path,
        target_start_year,
        target_end_year,
        date_col="Date",
        precip_col="precipitation",
        temp_col="temperature",
        snow_col="snowfall",
        noise_scale=0.05,
        random_seed=42,
):
    """
    Generates synthetic ERA-5 daily climate data by resampling from real
    data with added Gaussian noise.

    For each synthetic year, the method:
      1. Takes the real data and extracts the day-of-year signal (seasonal cycle)
      2. For each day in the synthetic year, finds the matching day-of-year
         in the real data and samples from those values
      3. Adds small Gaussian noise scaled to the variable's standard deviation
         to give each synthetic year a slightly different character

    Parameters
    ----------
    real_climate_path : str
        Path to your real ERA-5 CSV.
    output_path : str
        Where to save the synthetic CSV.
    target_start_year : int
        First year of the synthetic record (should match your streamflow record).
    target_end_year : int
        Last year of the synthetic record.
    date_col : str
    precip_col : str
    temp_col : str
    snow_col : str
    noise_scale : float
        Noise added as a fraction of each variable's standard deviation.
        0.05 = 5% noise, which keeps values realistic while adding variety.
    random_seed : int
        For reproducibility.
    """

    np.random.seed(random_seed)

    # ---------------------------
    # Load and prepare real data
    # ---------------------------
    real = pd.read_csv(real_climate_path)
    real[date_col] = pd.to_datetime(real[date_col])
    real = real.sort_values(date_col).reset_index(drop=True)

    climate_cols = [precip_col, temp_col, snow_col]

    # Add day-of-year to real data for seasonal matching
    real["DOY"] = real[date_col].dt.dayofyear

    # Compute per-variable noise standard deviations from real data
    noise_stds = {col: real[col].std() * noise_scale for col in climate_cols}

    print(f"Real data covers: {real[date_col].min().date()} to {real[date_col].max().date()}")
    print(f"Generating synthetic data from {target_start_year} to {target_end_year}")
    print(f"Noise scale: {noise_scale} (={noise_scale * 100:.0f}% of each variable's std dev)")
    for col in climate_cols:
        print(f"  {col}: noise std = {noise_stds[col]:.4f}")

    # ---------------------------
    # Build day-of-year lookup
    # from real data
    # ---------------------------
    # For each DOY, store the observed values across all real years
    doy_lookup = {}
    for doy in range(1, 367):
        matches = real[real["DOY"] == doy]
        if len(matches) == 0:
            # Edge case: DOY 366 may not exist in non-leap real data
            # Fall back to DOY 365
            matches = real[real["DOY"] == 365]
        doy_lookup[doy] = matches[climate_cols].values

    # ---------------------------
    # Generate synthetic date range
    # ---------------------------
    all_dates = pd.date_range(
        start=f"{target_start_year}-01-01",
        end=f"{target_end_year}-12-31",
        freq="D"
    )

    # Skip dates already in real data so we don't duplicate them
    real_dates = set(real[date_col].dt.date)
    synthetic_dates = [d for d in all_dates if d.date() not in real_dates]

    print(f"\nTotal synthetic days to generate: {len(synthetic_dates)}")

    # ---------------------------
    # Sample values for each
    # synthetic date
    # ---------------------------
    rows = []

    for date in synthetic_dates:
        doy = date.dayofyear
        pool = doy_lookup[doy]

        # Randomly pick one real observation for this DOY
        idx = np.random.randint(0, len(pool))
        sampled = pool[idx].copy().astype(float)

        # Add Gaussian noise to each variable
        for i, col in enumerate(climate_cols):
            noise = np.random.normal(0, noise_stds[col])
            sampled[i] += noise

            # Physical constraints — precip and snowfall can't be negative
            if col in (precip_col, snow_col):
                sampled[i] = max(0.0, sampled[i])

        rows.append({
            date_col: date,
            precip_col: sampled[0],
            temp_col: sampled[1],
            snow_col: sampled[2],
        })

    synthetic_df = pd.DataFrame(rows)

    # ---------------------------
    # Combine with real data and
    # sort by date
    # ---------------------------
    real_subset = real[[date_col, precip_col, temp_col, snow_col]].copy()
    real_subset["source"] = "real"
    synthetic_df["source"] = "synthetic"

    combined = pd.concat([real_subset, synthetic_df], ignore_index=True)
    combined = combined.sort_values(date_col).reset_index(drop=True)

    # Filter to target date range only
    combined = combined[
        (combined[date_col].dt.year >= target_start_year) &
        (combined[date_col].dt.year <= target_end_year)
        ]

    print(f"\nFinal combined dataset: {len(combined)} rows")
    print(f"  Real rows: {(combined['source'] == 'real').sum()}")
    print(f"  Synthetic rows: {(combined['source'] == 'synthetic').sum()}")

    # ---------------------------
    # Basic sanity checks
    # ---------------------------
    print("\nSanity checks:")
    for col in climate_cols:
        print(f"  {col}: min={combined[col].min():.4f}, "
              f"max={combined[col].max():.4f}, "
              f"mean={combined[col].mean():.4f}")

    # Save — drop the source column for clean output
    out = combined.drop(columns=["source"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"\nSynthetic ERA-5 data saved to: {output_path}")

    return combined


# ============================================================
# MAIN — update these paths and years to match your setup
# ============================================================

if __name__ == "__main__":
    # Path to your real 1-2 years of ERA-5 data
    REAL_CLIMATE_PATH = (
        "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/"
        "Retrospective_Data/era5_ohio_real.csv"  # <-- update this
    )

    # Where to save the synthetic + real combined output
    OUTPUT_PATH = (
        "/Users/bethlarsen/Downloads/Hydro Lab/stat_forecast_project/"
        "Retrospective_Data/era5_ohio_synthetic.csv"
    )

    # These should match the date range of your streamflow record
    TARGET_START_YEAR = 1940
    TARGET_END_YEAR = 2024

    combined = generate_synthetic_era5(
        real_climate_path=REAL_CLIMATE_PATH,
        output_path=OUTPUT_PATH,
        target_start_year=TARGET_START_YEAR,
        target_end_year=TARGET_END_YEAR,
        date_col="Date",
        precip_col="precipitation",
        temp_col="temperature",
        snow_col="snowfall",
        noise_scale=0.05,  # increase for more variety, decrease for closer to real
        random_seed=42,
    )