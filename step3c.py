# step_3c.py
# Test multiple regression forms (linear, polynomial, log, exponential, power)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# === USER SETTINGS ===
csv_path = "/Users/bethlarsen/Documents/Hydro Research/geoglows_stat_forecast/flow_data.csv"
ref_month = 6
ref_day = 15
window_months_list = [3, 6, 9, 12]
date_col = "Date"
flow_col = "Flow_m3s"

# === LOAD AND PREP DATA ===
df = pd.read_csv(csv_path)
df[date_col] = pd.to_datetime(df[date_col])
df = df.set_index(date_col).sort_index()
df["Volume_m3"] = df[flow_col] * 24 * 60 * 60

results = []

# === MAIN LOOP ===
for window_months in window_months_list:
    records = []
    for year in df.index.year.unique():
        ref_date = pd.Timestamp(f"{year}-{ref_month:02d}-{ref_day:02d}")
        start_past = ref_date - pd.DateOffset(months=window_months)
        end_future = ref_date + pd.DateOffset(months=window_months)

        if start_past < df.index.min() or end_future > df.index.max():
            continue

        past_sum = df.loc[start_past:ref_date, "Volume_m3"].sum()
        future_sum = df.loc[ref_date:end_future, "Volume_m3"].sum()
        records.append({"Year": year, "Past_m3": past_sum, "Future_m3": future_sum})

    rec_df = pd.DataFrame(records)
    if rec_df.empty:
        continue

    X = rec_df["Past_m3"].values.reshape(-1, 1)
    y = rec_df["Future_m3"].values

    models = {}

    # --- 1️⃣ Linear ---
    lin = LinearRegression().fit(X, y)
    y_pred_lin = lin.predict(X)
    models["Linear"] = (y_pred_lin, lin.coef_[0], lin.intercept_, r2_score(y, y_pred_lin))

    # --- 2️⃣ Polynomial (2nd degree) ---
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression().fit(X_poly, y)
    y_pred_poly = poly_model.predict(X_poly)
    models["Quadratic"] = (y_pred_poly, poly_model.coef_[1:], poly_model.intercept_, r2_score(y, y_pred_poly))

    # --- 3️⃣ Logarithmic ---
    X_log = np.log(X[X > 0])  # avoid negative/zero
    y_log = y[:len(X_log)]
    log_model = LinearRegression().fit(X_log, y_log)
    y_pred_log = log_model.predict(X_log)
    models["Logarithmic"] = (y_pred_log, log_model.coef_[0], log_model.intercept_, r2_score(y_log, y_pred_log))

    # --- 4️⃣ Exponential (log-transform y) ---
    y_pos = y[y > 0]
    X_exp = X[:len(y_pos)]
    exp_model = LinearRegression().fit(X_exp, np.log(y_pos))
    y_pred_exp = np.exp(exp_model.predict(X_exp))
    models["Exponential"] = (y_pred_exp, exp_model.coef_[0], exp_model.intercept_, r2_score(y_pos, y_pred_exp))

    # --- 5️⃣ Power-law (log-log) ---
    X_pow = np.log(X[X > 0])
    y_pow = np.log(y[:len(X_pow)])
    pow_model = LinearRegression().fit(X_pow, y_pow)
    y_pred_pow = np.exp(pow_model.predict(X_pow))
    models["Power"] = (y_pred_pow, pow_model.coef_[0], pow_model.intercept_, r2_score(y[:len(X_pow)], y_pred_pow))

    # --- Pick best model by R² ---
    best_form = max(models.items(), key=lambda kv: kv[1][3])
    best_name, (pred, coef, intercept, r2) = best_form

    results.append({
        "Window_months": window_months,
        "Best_Model": best_name,
        "R2": r2,
        "Coeff": coef,
        "Intercept": intercept
    })

    # --- Plot ---
    plt.figure(figsize=(6, 5))
    plt.scatter(X, y, color="blue", alpha=0.6)
    plt.scatter(X, pred, color="red", alpha=0.7, label=f"{best_name} fit (R²={r2:.3f})")
    plt.xlabel(f"Past {window_months} Months Volume (m³)")
    plt.ylabel(f"Future {window_months} Months Volume (m³)")
    plt.title(f"{window_months}-Month Window — Best: {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === SUMMARY ===
summary_df = pd.DataFrame(results)
print("\n=== Model Comparison Summary ===")
print(summary_df.round(3))

plt.figure(figsize=(7, 5))
for model_name in summary_df["Best_Model"].unique():
    subset = summary_df[summary_df["Best_Model"] == model_name]
    plt.scatter(subset["Window_months"], subset["R2"], label=model_name, s=80)
plt.plot(summary_df["Window_months"], summary_df["R2"], "k--", alpha=0.4)
plt.title("Best Fit vs. Window Length")
plt.xlabel("Window Length (months)")
plt.ylabel("R²")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()