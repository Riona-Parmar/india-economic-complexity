
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Load labels and baseline predictions
labels = pd.read_parquet("labels_v1.parquet")
rca = pd.read_parquet("predictions_rca.parquet")
density = pd.read_parquet("predictions_density.parquet")

keys = ["asof_year", "country", "product"]

# Create output folder
os.makedirs("pr_curves_baselines", exist_ok=True)

# Keep only test rows
test_labels = labels[
    labels["split"].isin(["test_india", "test_peer"])
].dropna(subset=["label"]).copy()

# Merge both prediction scores with labels
data = test_labels.merge(
    rca[keys + ["score"]].rename(columns={"score": "rca_score"}),
    on=keys,
    how="left",
    validate="one_to_one"
)

data = data.merge(
    density[keys + ["score"]].rename(columns={"score": "density_score"}),
    on=keys,
    how="left",
    validate="one_to_one"
)

# Validate prediction scores
if data[["rca_score", "density_score"]].isna().any().any():
    raise ValueError("Missing prediction scores!")

if not data[["rca_score", "density_score"]].map(
    lambda x: pd.notna(x) and abs(x) != float("inf")
).all().all():
    raise ValueError("Invalid prediction scores!")

subsets = {
    "india": lambda x: x[x["split"] == "test_india"],
    "peers": lambda x: x[x["split"] == "test_peer"],
    "all": lambda x: x
}

for year in sorted(data["asof_year"].unique()):

    year_data = data[data["asof_year"] == year]

    for subset_name, filter_func in subsets.items():

        subset = filter_func(year_data)

        if len(subset) == 0:
            continue

        y_true = subset["label"].astype(int)

        if y_true.nunique() < 2:
            print(f"Skipping {year} {subset_name}: only one label class")
            continue

        plt.figure(figsize=(8, 6))

        # RCA persistence curve
        precision_rca, recall_rca, _ = precision_recall_curve(
            y_true, subset["rca_score"]
        )
        ap_rca = average_precision_score(
            y_true, subset["rca_score"]
        )

        plt.plot(
            recall_rca,
            precision_rca,
            label=f"RCA Persistence (AP = {ap_rca:.4f})"
        )

        # Density curve
        precision_density, recall_density, _ = precision_recall_curve(
            y_true, subset["density_score"]
        )
        ap_density = average_precision_score(
            y_true, subset["density_score"]
        )

        plt.plot(
            recall_density,
            precision_density,
            label=f"Density (AP = {ap_density:.4f})"
        )

        # Base-rate reference
        base_rate = y_true.mean()
        plt.axhline(
            y=base_rate,
            linestyle="--",
            label=f"Base rate = {base_rate:.4f}"
        )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - A{year} - {subset_name}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)

        filename = f"pr_curves_baselines/PR_A{year}_{subset_name}.png"

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print("Saved:", filename)

print("\nAll baseline PR curves generated!")
