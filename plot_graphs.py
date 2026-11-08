
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Results from your model comparison
data = {
    "Model": [
        "RCA Persistence",
        "Proximity Density",
        "XGB-A Baseline",
        "XGB-A Weight 0.25",
        "XGB-F",
        "Equal Rank Ensemble"
    ],

    "Precision@20": [
        0.0426,
        0.0300,
        0.0553,
        0.0571,
        0.0567,
        0.0568
    ],

    "PR-AUC": [
        0.0242,
        0.0172,
        0.0396,
        0.0409,
        0.0403,
        0.0314
    ],

    "ROC-AUC": [
        0.5869,
        0.6168,
        0.7280,
        0.7324,
        0.7325,
        0.6664
    ]
}

df = pd.DataFrame(data)

# Separate XGBoost variants
xgb = df[df["Model"].str.startswith("XGB")].copy()

# Choose the best XGBoost based on Precision@20
best_xgb = xgb.sort_values(
    by=["Precision@20", "PR-AUC"],
    ascending=False
).iloc[0]

print("\nMODEL COMPARISON")
print(df.to_string(index=False))

print("\nBEST XGBOOST MODEL")
print(best_xgb["Model"])

print("Precision@20:", best_xgb["Precision@20"])
print("PR-AUC:", best_xgb["PR-AUC"])
print("ROC-AUC:", best_xgb["ROC-AUC"])


# ----------------------------------
# GRAPH 1: Precision@20
# ----------------------------------

plt.figure(figsize=(11, 6))

plt.bar(df["Model"], df["Precision@20"])

plt.title("Model Comparison - Precision@20")
plt.xlabel("Model")
plt.ylabel("Precision@20")

plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("precision_comparison.png", dpi=300)
plt.show()


# ----------------------------------
# GRAPH 2: PR-AUC
# ----------------------------------

plt.figure(figsize=(11, 6))

plt.bar(df["Model"], df["PR-AUC"])

plt.title("Model Comparison - PR-AUC")
plt.xlabel("Model")
plt.ylabel("PR-AUC")

plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("pr_auc_comparison.png", dpi=300)
plt.show()


# ----------------------------------
# GRAPH 3: ROC-AUC
# ----------------------------------

plt.figure(figsize=(11, 6))

plt.bar(df["Model"], df["ROC-AUC"])

plt.title("Model Comparison - ROC-AUC")
plt.xlabel("Model")
plt.ylabel("ROC-AUC")

plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("roc_auc_comparison.png", dpi=300)
plt.show()


# ----------------------------------
# GRAPH 4: All metrics for XGBoost
# ----------------------------------

xgb_plot = xgb.set_index("Model")[
    ["Precision@20", "PR-AUC", "ROC-AUC"]
]

ax = xgb_plot.plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title("Comparison of XGBoost Variants")
ax.set_xlabel("XGBoost Model")
ax.set_ylabel("Metric Score")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

plt.legend(title="Metrics")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("xgboost_variants_comparison.png", dpi=300)
plt.show()
