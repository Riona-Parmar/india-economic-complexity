


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Input and output paths
INPUT = Path("software_results/evaluation/services_evaluation_metrics.csv")
OUTPUT = Path("software_results/evaluation/density_vs_rca_backtest.png")

df = pd.read_csv(INPUT)

# Convert year to integer
df["asof_year"] = df["asof_year"].astype(int)

# Metrics to plot
metrics = [
    ("Precision@3", "Precision@3", True),
    ("Recall@3", "Recall@3", True),
    ("Hit@3", "Hit@3", True),
    ("PR-AUC", "PR-AUC", False)
]

# Create a single 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

models = ["Density", "RCA Persistence"]

for ax, (column, title, percentage) in zip(axes, metrics):

    for model in models:
        subset = df[df["model"] == model].sort_values("asof_year")

        values = subset[column]
        if percentage:
            values = values * 100

        mean_value = values.mean()

        ax.plot(
            subset["asof_year"],
            values,
            marker="o",
            markersize=4,
            linewidth=2,
            label=f"{model} (Avg: {mean_value:.2f}{'%' if percentage else ''})"
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("As-of Year", fontsize=11)
    ax.set_ylabel("Percentage (%)" if percentage else "PR-AUC",
                  fontsize=11)

    ax.set_xticks(sorted(df["asof_year"].unique()))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)

# Main title
fig.suptitle(
    "Density vs RCA Persistence: Historical Prediction Performance",
    fontsize=18,
    fontweight="bold",
    y=0.99
)

fig.text(
    0.5, 0.015,
    "Five-year prediction horizon | Top-3 recommendations | "
    "Historical backtesting: 2005–2020",
    ha="center",
    fontsize=11
)

plt.tight_layout(rect=[0, 0.045, 1, 0.95])

# Save high-resolution image
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.show()

print(f"Graph saved at: {OUTPUT}")
