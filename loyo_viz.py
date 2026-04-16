"""
LOYO River Forecast — Metric Visualizer
========================================
Run this script in any Python IDE (VS Code, Spyder, PyCharm, IDLE)
or double-click it if Python is associated with .py files on your system.

A file picker will open so you can select one or more CSV files.
All plots are displayed automatically when done.

Required packages (install once via pip if needed):
    pip install pandas matplotlib seaborn
"""

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # Use Tk window backend — avoids PyCharm's outdated plot viewer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ── Config ────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

WY_DAY_LABELS = {
    45:  "Nov 14\n(WY Day 45)",
    90:  "Dec 29\n(WY Day 90)",
    135: "Feb 12\n(WY Day 135)",
    180: "Mar 29\n(WY Day 180)",
    225: "May 13\n(WY Day 225)",
    270: "Jun 27\n(WY Day 270)",
}

SEASON_COLORS = {
    45:  "#4e79a7",
    90:  "#f28e2b",
    135: "#e15759",
    180: "#76b7b2",
    225: "#59a14f",
    270: "#edc948",
}

METRIC_META = {
    "RMSE":            {"label": "RMSE (ft³)",          "better": "lower"},
    "Bias":            {"label": "Bias (ft³)",           "better": "near zero"},
    "NSE":             {"label": "Nash–Sutcliffe (NSE)", "better": "higher"},
    "HorizonPctError": {"label": "Horizon % Error",      "better": "near zero"},
}


# ── File picker ───────────────────────────────────────────────────────────────

def pick_files():
    root = tk.Tk()
    root.withdraw()
    root.call("wm", "attributes", ".", "-topmost", True)

    files = filedialog.askopenfilenames(
        title="Select LOYO CSV file(s) — hold Ctrl/Cmd to select multiple",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()

    if not files:
        print("No files selected. Exiting.")
        raise SystemExit

    return list(files)


def pick_outdir():
    root = tk.Tk()
    root.withdraw()
    root.call("wm", "attributes", ".", "-topmost", True)

    folder = filedialog.askdirectory(
        title="Choose a folder to save figures (cancel to skip saving)",
    )
    root.destroy()
    return Path(folder) if folder else None


def ask_labels(files):
    defaults = [Path(f).stem for f in files]
    root = tk.Tk()
    root.withdraw()
    root.call("wm", "attributes", ".", "-topmost", True)

    labels = []
    for i, (f, default) in enumerate(zip(files, defaults)):
        label = simpledialog.askstring(
            title=f"River name ({i+1}/{len(files)})",
            prompt=f"Enter a friendly name for:\n{Path(f).name}",
            initialvalue=default,
            parent=root,
        )
        labels.append(label if label else default)

    root.destroy()
    return labels


def ask_metrics():
    choices = list(METRIC_META.keys())

    root = tk.Tk()
    root.title("Select metrics to plot")
    root.call("wm", "attributes", ".", "-topmost", True)
    root.resizable(False, False)

    # BooleanVars must be created after the root window exists
    selected = {m: tk.BooleanVar(master=root, value=True) for m in choices}

    tk.Label(root, text="Which metrics would you like to visualize?",
             font=("Helvetica", 11, "bold"), pady=8).pack()

    for m in choices:
        tk.Checkbutton(
            root, text=f"{m}  —  {METRIC_META[m]['label']}",
            variable=selected[m], anchor="w", font=("Helvetica", 10),
        ).pack(fill="x", padx=20)

    result = []

    def confirm():
        chosen = [m for m in choices if selected[m].get()]
        if not chosen:
            messagebox.showwarning("Nothing selected", "Pick at least one metric.")
            return
        result.extend(chosen)
        root.destroy()

    tk.Button(root, text="  Plot!  ", command=confirm,
              font=("Helvetica", 11, "bold"),
              bg="#4e79a7", fg="white", pady=4).pack(pady=12)

    root.mainloop()

    if not result:
        raise SystemExit
    return result


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(files, labels):
    frames = []
    for path, label in zip(files, labels):
        df = pd.read_csv(path, parse_dates=["ForecastDate"])
        df["River"] = label
        df["WY_Day_Label"] = df["Evaluation_WY_Day"].map(WY_DAY_LABELS)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def season_order(df):
    days = sorted(df["Evaluation_WY_Day"].unique())
    return [WY_DAY_LABELS[d] for d in days if d in WY_DAY_LABELS]


# ── Plot 1: Box plots ─────────────────────────────────────────────────────────

def plot_boxplots(df, metrics):
    rivers = df["River"].unique()
    order = season_order(df)
    palette = {WY_DAY_LABELS[k]: v for k, v in SEASON_COLORS.items()
               if k in df["Evaluation_WY_Day"].unique()}

    for metric in metrics:
        meta = METRIC_META[metric]
        n = len(rivers)
        fig, axes = plt.subplots(1, n, figsize=(max(7, 5 * n), 6),
                                 squeeze=False)

        for ax, river in zip(axes[0], rivers):
            sub = df[df["River"] == river]
            sns.boxplot(
                data=sub, x="WY_Day_Label", y=metric,
                order=order,
                hue="WY_Day_Label", hue_order=order,
                palette=palette, legend=False,
                linewidth=0.8,
                flierprops=dict(marker="o", markersize=3, alpha=0.5),
                ax=ax,
            )
            ax.set_title(river, fontweight="bold")
            ax.set_xlabel("Forecast Issue Date (Water Year Day)")
            ax.set_ylabel(meta["label"] if ax == axes[0][0] else "")
            ax.tick_params(axis="x", rotation=0)

            if metric == "NSE":
                ax.set_ylim(-1, 1)
                ax.axhline(0, color="red", linestyle="--",
                           linewidth=0.8, label="NSE = 0 (benchmark)")

                # Annotate each box with count of values below 0
                for i, lbl in enumerate(order):
                    group = sub[sub["WY_Day_Label"] == lbl]["NSE"]
                    n_below = int((group < 0).sum())
                    if n_below:
                        ax.text(
                            i, -0.97,
                            f"n<0: {n_below}",
                            ha="center", va="bottom",
                            fontsize=7.5, color="red", style="italic",
                        )

                n_clipped = int((sub["NSE"] < -1).sum())
                if n_clipped:
                    ax.annotate(
                        f"Note: {n_clipped} value(s) below −1 not shown",
                        xy=(0.5, 0.06), xycoords="axes fraction",
                        ha="center", fontsize=7.5, color="gray", style="italic",
                    )
                ax.legend(fontsize=8)
            elif metric in ("Bias", "HorizonPctError"):
                ax.axhline(0, color="gray", linestyle="--",
                           linewidth=0.8, alpha=0.7)

        fig.suptitle(
            f"{meta['label']} Distribution by Forecast Issue Date\n"
            f"(better = {meta['better']})",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()


# ── Plot 2: Time series ───────────────────────────────────────────────────────

def plot_timeseries(df, metrics):
    rivers = df["River"].unique()
    wy_days = sorted(df["Evaluation_WY_Day"].unique())

    for metric in metrics:
        meta = METRIC_META[metric]
        n = len(rivers)
        fig, axes = plt.subplots(n, 1, figsize=(13, 4 * n), squeeze=False)

        for ax, river in zip(axes[:, 0], rivers):
            sub = df[df["River"] == river].sort_values("ForecastDate")

            for day in wy_days:
                chunk = sub[sub["Evaluation_WY_Day"] == day]
                ax.plot(
                    chunk["ForecastDate"], chunk[metric],
                    marker="o", markersize=3, linewidth=1.2, alpha=0.85,
                    color=SEASON_COLORS.get(day),
                    label=WY_DAY_LABELS.get(day, str(day)),
                )

            if metric == "NSE":
                ax.set_ylim(-1, 1)
                ax.axhline(0, color="red", linestyle="--",
                           linewidth=0.8, alpha=0.7)
                n_clipped = int((sub["NSE"] < -1).sum())
                if n_clipped:
                    ax.annotate(
                        f"Note: {n_clipped} value(s) below −1 not shown",
                        xy=(0.01, 0.02), xycoords="axes fraction",
                        ha="left", fontsize=7.5, color="gray", style="italic",
                    )
            elif metric in ("Bias", "HorizonPctError"):
                ax.axhline(0, color="gray", linestyle="--",
                           linewidth=0.8, alpha=0.6)

            ax.set_title(river, fontweight="bold")
            ax.set_ylabel(meta["label"])
            ax.set_xlabel("Forecast Date")
            ax.legend(title="Issue Date (WY Day)", fontsize=7,
                      title_fontsize=8, loc="upper left", ncol=2)

        fig.suptitle(
            f"{meta['label']} Over Time by Forecast Issue Date\n"
            f"(better = {meta['better']})",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()


# ── Plot 3: Heatmap (multi-river only) ────────────────────────────────────────

def plot_heatmap(df, metrics):
    if df["River"].nunique() < 2:
        return

    order = season_order(df)

    for metric in metrics:
        meta = METRIC_META[metric]
        pivot = (
            df.groupby(["River", "WY_Day_Label"])[metric]
            .median()
            .unstack("WY_Day_Label")
            .reindex(columns=order)
        )

        fig, ax = plt.subplots(
            figsize=(max(8, len(order) * 1.5),
                     max(3, df["River"].nunique() * 0.9 + 1))
        )

        if metric in ("Bias", "HorizonPctError"):
            vmax = pivot.abs().max().max()
            sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0,
                        vmin=-vmax, vmax=vmax, annot=True, fmt=".1f",
                        linewidths=0.5, cbar_kws={"label": meta["label"]})
        else:
            cmap = "RdYlGn" if meta["better"] == "higher" else "RdYlGn_r"
            sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=".2f",
                        linewidths=0.5, cbar_kws={"label": meta["label"]})

        ax.set_title(
            f"Median {meta['label']} — Rivers × Forecast Issue Date\n"
            f"(better = {meta['better']})",
            fontweight="bold",
        )
        ax.set_xlabel("Forecast Issue Date (Water Year Day)")
        ax.set_ylabel("River / Basin")
        fig.tight_layout()


# ── Main ──────────────────────────────────────────────────────────────────────

def save_all_figures(outdir):
    for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
        title = fig.texts[0].get_text() if fig.texts else f"figure_{i+1}"
        # Make a safe filename from the figure title
        safe = title.split("\n")[0]
        for ch in r'/\:*?"<>| ':
            safe = safe.replace(ch, "_")
        safe = safe.strip("_")
        path = outdir / f"{safe}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")


def main():
    print("Opening file picker...")
    files = pick_files()
    print(f"Selected {len(files)} file(s).")

    labels = ask_labels(files)
    metrics = ask_metrics()
    outdir = pick_outdir()

    print("Loading data...")
    df = load_data(files, labels)
    print(f"  {len(df):,} rows | rivers: {list(df['River'].unique())}")
    print(f"  Plotting: {metrics}")

    print("Generating plots...")
    plot_boxplots(df, metrics)
    plot_timeseries(df, metrics)
    plot_heatmap(df, metrics)

    if outdir:
        print(f"Saving figures to: {outdir}")
        save_all_figures(outdir)
        print("All figures saved!")
    else:
        print("No save folder selected — skipping save.")

    print("Displaying plots...")
    plt.show()


if __name__ == "__main__":
    main()
