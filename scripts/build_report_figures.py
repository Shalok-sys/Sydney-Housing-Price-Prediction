"""
Builds the figures used in the LaTeX report.

Output is vector PDF so the figures stay sharp at any size in the
compiled document.

Colour choices were validated rather than guessed. The first instinct
for the correlation chart was green for positive and red for negative,
which fails a colour vision deficiency check at a separation of 6.7,
the classic red and green trap. The blue and orange pair used here
separates by more than 21 under every simulated deficiency.

Usage
    python3 scripts/build_report_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = "data/processed/predictions.csv"
OUT_DIR = "report/figures"

# One hue for magnitude, a validated pair for anything with a sign
HUE = "#2166ac"
POSITIVE = "#2166ac"
NEGATIVE = "#b2510a"

# Text wears ink, never the series colour
INK = "#1c1c19"
INK_SOFT = "#5c5a54"
GRID = "#d8d7d2"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 150,
})


def strip(ax):
    """Keep the frame recessive so the data carries the chart."""
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.8)


def price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    axes[0].hist(df["sale_price"] / 1e6, bins=40, color=HUE, linewidth=0)
    axes[0].set_xlabel("Sale price, million dollars")
    axes[0].set_ylabel("Properties")
    axes[0].set_title("As recorded", loc="left", fontsize=9.5, color=INK)

    axes[1].hist(np.log(df["sale_price"]), bins=40, color=HUE, linewidth=0)
    axes[1].set_xlabel("Log of sale price")
    axes[1].set_title("On a log scale", loc="left", fontsize=9.5, color=INK)

    for ax in axes:
        ax.grid(axis="x", visible=False)
        strip(ax)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_price_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def price_by_suburb(df):
    order = df.groupby("suburb")["sale_price"].median().sort_values().index.tolist()
    data = [df[df["suburb"] == s]["sale_price"].values for s in order]

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    parts = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.5)

    for box in parts["boxes"]:
        box.set(facecolor=HUE, alpha=0.22, edgecolor=HUE, linewidth=1.2)
    for element in ["whiskers", "caps", "medians"]:
        for item in parts[element]:
            item.set(color=HUE, linewidth=1.2)
    for flier in parts["fliers"]:
        flier.set(marker="o", markersize=3, markerfacecolor=HUE,
                  markeredgecolor="none", alpha=0.55)

    ax.set_yscale("log")
    ax.set_ylabel("Sale price, log scale")
    ax.grid(axis="x", visible=False)
    strip(ax)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_price_by_suburb.pdf", bbox_inches="tight")
    plt.close(fig)


def correlations(df):
    features = ["bedrooms", "bathrooms", "parking_spaces",
                "land_size_sqm", "floor_area_sqm"]
    labels = {
        "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms",
        "parking_spaces": "Parking spaces",
        "land_size_sqm": "Land size",
        "floor_area_sqm": "Floor area",
    }
    values = df[features + ["sale_price"]].corr()["sale_price"].drop("sale_price")
    values = values.sort_values()

    colours = [POSITIVE if v >= 0 else NEGATIVE for v in values]

    fig, ax = plt.subplots(figsize=(4.9, 2.6))
    positions = np.arange(len(values))
    ax.barh(positions, values.values, color=colours, height=0.6, linewidth=0)

    ax.set_yticks(positions)
    ax.set_yticklabels([labels[name] for name in values.index])
    ax.set_xlabel("Correlation with sale price")
    ax.axvline(0, color=INK_SOFT, linewidth=0.8)
    ax.set_xlim(-0.1, 0.75)
    ax.grid(axis="y", visible=False)

    # Direct labels double as the secondary encoding for the sign
    for position, value in zip(positions, values.values):
        ax.text(value + 0.02, position, f"{value:+.2f}",
                va="center", fontsize=8, color=INK_SOFT)

    strip(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_correlations.pdf", bbox_inches="tight")
    plt.close(fig)


def error_by_suburb(df):
    errors = df.groupby("suburb")["pct_error"].median().sort_values()

    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    positions = np.arange(len(errors))
    ax.barh(positions, errors.values, color=HUE, height=0.55, linewidth=0)

    ax.set_yticks(positions)
    ax.set_yticklabels(errors.index)
    ax.set_xlabel("Median error, percent of sale price")
    ax.set_xlim(0, max(errors.values) * 1.25)
    ax.grid(axis="y", visible=False)

    for position, value in zip(positions, errors.values):
        ax.text(value + 0.5, position, f"{value:.1f}%",
                va="center", fontsize=8.5, color=INK_SOFT)

    strip(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_error_by_suburb.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA)

    price_distribution(df)
    price_by_suburb(df)
    correlations(df)

    if "pct_error" in df.columns:
        error_by_suburb(df)
        made = 4
    else:
        print("No pct_error column, run the notebook first to create it")
        made = 3

    print(f"Wrote {made} figures to {OUT_DIR}")
    for name in sorted(os.listdir(OUT_DIR)):
        print(f"  {name}")


if __name__ == "__main__":
    main()
