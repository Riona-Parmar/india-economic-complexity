
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


# ============================================================
# ARGUMENTS
# ============================================================

ap = argparse.ArgumentParser()

ap.add_argument("--preds", required=True)
ap.add_argument("--labels")
ap.add_argument("--trade")
ap.add_argument("--density")
ap.add_argument("--models")
ap.add_argument("--ref", default="rca_persistence")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--out", default="results")

a = ap.parse_args()


# ============================================================
# COLUMN ALIASES
# ============================================================

ALIAS = dict(
    asof_year=["asof_year", "validation_year", "val_year",
               "fold_year", "fold", "year"],

    country=["country", "country_iso3", "iso3", "country_code"],

    product=["product", "hs4", "product_code", "hs4_code"],

    score=["score", "probability", "pred", "prediction",
           "y_score", "pred_proba", "proba",
           "predicted_probability"],

    label=["label", "y_true", "target", "y", "outcome"],

    model=["model", "model_name", "candidate", "variant"],
)

KEYS = ["asof_year", "country", "product"]


# ============================================================
# FILE READING
# ============================================================

def read(path):
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)

    return pd.read_csv(path)


def read_multiple(pattern):
    """
    Read one or multiple files.
    Supports wildcard paths such as density_A*.parquet.
    """

    files = sorted(glob.glob(pattern, recursive=True))

    if not files and os.path.isfile(pattern):
        files = [pattern]

    if not files:
        sys.exit(f"No files found for: {pattern}")

    frames = []

    for file in files:

        df = read(file)

        # Infer year from filenames like density_A2000.parquet
        if "asof_year" not in df.columns and "year" not in df.columns:

            match = re.search(
                r"A(20\d{2})",
                os.path.basename(file),
                re.IGNORECASE
            )

            if match:
                df["asof_year"] = int(match.group(1))

        frames.append(df)

        print(f"[read] {file}")

    return pd.concat(frames, ignore_index=True)


# ============================================================
# COLUMN STANDARDIZATION
# ============================================================

def norm(df, required, name):

    df = df.copy()

    rename = {}

    for col in required:

        if col in df.columns:
            continue

        matches = [
            c for c in ALIAS.get(col, [])
            if c in df.columns
        ]

        if matches:
            rename[matches[0]] = col

    df = df.rename(columns=rename)

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        sys.exit(
            f"\n{name}: Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def standardize_product(df):

    df = df.copy()

    df["product"] = (
        df["product"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(4)
    )

    return df


def find_file(argument, patterns, name):

    if argument:

        if not os.path.exists(argument):
            sys.exit(f"{name} file not found: {argument}")

        return argument

    hits = sorted({
        path
        for pattern in patterns
        for path in glob.glob(
            os.path.join("**", pattern),
            recursive=True
        )
    })

    if len(hits) == 1:

        print(f"[auto] {name}: {hits[0]}")
        return hits[0]

    if len(hits) > 1:

        print(
            f"[skip] Multiple {name} files found. "
            f"Pass --{name} explicitly."
        )

    else:

        print(f"[skip] No {name} file found.")

    return None


# ============================================================
# 1. READ MODEL PREDICTIONS
# ============================================================

print("\n========== READING PREDICTIONS ==========")

pred = read(a.preds)

has_model = any(
    col in pred.columns
    for col in ALIAS["model"]
)

pred = norm(
    pred,
    KEYS + ["score"] + (["model"] if has_model else []),
    "predictions"
)

if not has_model:

    pred["model"] = os.path.splitext(
        os.path.basename(a.preds)
    )[0]


pred = standardize_product(pred)

if a.models:

    requested_models = [
        m.strip()
        for m in a.models.split(",")
    ]

    pred = pred[
        pred["model"].isin(requested_models)
    ].copy()

if pred.empty:
    sys.exit("No prediction rows found.")


# ============================================================
# 2. ATTACH LABELS
# ============================================================

print("\n========== LOADING LABELS ==========")

if not any(
    col in pred.columns
    for col in ALIAS["label"]
):

    label_path = find_file(
        a.labels,
        ["labels_v*.parquet", "labels*.parquet"],
        "labels"
    )

    if not label_path:
        sys.exit("No labels found. Pass --labels.")

    labels = read(label_path)

    labels = norm(
        labels,
        KEYS + ["label"],
        "labels"
    )

    labels = standardize_product(labels)

    labels = labels[
        KEYS + ["label"]
    ].drop_duplicates(KEYS)

    pred = pred.merge(
        labels,
        on=KEYS,
        how="inner"
    )

else:

    pred = norm(
        pred,
        ["label"],
        "predictions"
    )


pred = pred[pred["label"].notna()].copy()

pred["label"] = pd.to_numeric(
    pred["label"],
    errors="coerce"
)

pred["score"] = pd.to_numeric(
    pred["score"],
    errors="coerce"
)

pred = pred.dropna(
    subset=["label", "score"]
)

if not pred["label"].isin([0, 1]).all():
    sys.exit("Labels must be binary (0 or 1).")

if pred.empty:
    sys.exit("No labelled prediction rows remain.")

# Check label consistency
label_counts = pred.groupby(KEYS)["label"].nunique()

if (label_counts > 1).any():
    sys.exit("Conflicting labels found for the same observation.")

pred = pred[
    KEYS + ["model", "score", "label"]
].drop_duplicates(KEYS + ["model"])

base_keys = pred[
    KEYS + ["label"]
].drop_duplicates(KEYS)

frames = [pred]


# ============================================================
# 3. RCA PERSISTENCE BASELINE
# ============================================================

print("\n========== RCA PERSISTENCE BASELINE ==========")

trade_path = find_file(
    a.trade,
    ["trade_rca_v*.parquet", "trade_rca*.parquet"],
    "trade"
)

if trade_path:

    trade = read(trade_path)

    trade = norm(
        trade,
        ["country", "product"],
        "trade_rca"
    )

    if "asof_year" not in trade.columns:

        if "year" in trade.columns:
            trade = trade.rename(
                columns={"year": "asof_year"}
            )

        else:
            sys.exit("trade_rca needs a year or asof_year column.")

    if "rca3" not in trade.columns:
        sys.exit("trade_rca does not contain rca3.")

    trade = standardize_product(trade)

    trade["asof_year"] = pd.to_numeric(
        trade["asof_year"],
        errors="coerce"
    )

    trade["rca3"] = pd.to_numeric(
        trade["rca3"],
        errors="coerce"
    )

    trade = trade.dropna(
        subset=["asof_year", "rca3"]
    )

    trade = trade[
        KEYS + ["rca3"]
    ].drop_duplicates(KEYS)

    baseline = base_keys.merge(
        trade,
        on=KEYS,
        how="inner"
    )

    baseline = baseline.rename(
        columns={"rca3": "score"}
    )

    baseline["model"] = "rca_persistence"

    frames.append(
        baseline[
            KEYS + ["model", "score", "label"]
        ]
    )

    print(
        "RCA baseline rows:",
        len(baseline)
    )


# ============================================================
# 4. PROXIMITY DENSITY BASELINE
# ============================================================

print("\n========== DENSITY BASELINE ==========")

if a.density:

    density = read_multiple(a.density)

    density = norm(
        density,
        ["country", "product"],
        "density"
    )

    if "asof_year" not in density.columns:

        if "year" in density.columns:

            density = density.rename(
                columns={"year": "asof_year"}
            )

        else:

            sys.exit(
                "Density files need asof_year/year "
                "or filenames like density_A2000.parquet."
            )

    density_col = next(
        (
            col for col in ["density", "score"]
            if col in density.columns
        ),
        None
    )

    if density_col is None:

        sys.exit(
            "Density column not found. "
            f"Available columns: {list(density.columns)}"
        )

    density = standardize_product(density)

    density["asof_year"] = pd.to_numeric(
        density["asof_year"],
        errors="coerce"
    )

    density[density_col] = pd.to_numeric(
        density[density_col],
        errors="coerce"
    )

    density = density.dropna(
        subset=["asof_year", density_col]
    )

    density = density[
        KEYS + [density_col]
    ].drop_duplicates(KEYS)

    baseline = base_keys.merge(
        density,
        on=KEYS,
        how="inner"
    )

    baseline = baseline.rename(
        columns={density_col: "score"}
    )

    baseline["model"] = "density_proximity"

    frames.append(
        baseline[
            KEYS + ["model", "score", "label"]
        ]
    )

    print(
        "Density baseline rows:",
        len(baseline)
    )


# ============================================================
# 5. IDENTICAL ROW COMPARISON
# ============================================================

print("\n========== IDENTICAL ROW FILTER ==========")

allp = pd.concat(
    frames,
    ignore_index=True
)

allp = allp.dropna(
    subset=["score", "label"]
)

models = sorted(
    allp["model"].unique()
)

if len(models) < 2:

    sys.exit(
        f"Only {len(models)} model(s) found: {models}. "
        "Check whether the baselines were loaded."
    )

if allp.duplicated(KEYS + ["model"]).any():

    sys.exit(
        "Duplicate observations found for a model. "
        "Check duplicate rows in the input files."
    )

counts = allp.groupby(KEYS)["model"].nunique()

common = counts[
    counts == len(models)
].index

allp = (
    allp.set_index(KEYS)
    .loc[common]
    .reset_index()
)

if allp.empty:

    sys.exit(
        "No identical rows across all models. "
        "Check years, country codes and product codes."
    )

print("\nModels:", models)

print(
    "Rows per model:",
    len(allp) // len(models)
)

print(
    "Countries:",
    allp["country"].nunique()
)

print(
    "Windows:",
    sorted(allp["asof_year"].unique())
)


# ============================================================
# 6. COUNTRY-LEVEL RANKING METRICS
# ============================================================

print("\n========== CALCULATING METRICS ==========")

KS = [10, 20, 50, 100]

unit = []

for (model, year, country), group in allp.groupby(
    ["model", "asof_year", "country"]
):

    group = group.sort_values(
        ["score", "product"],
        ascending=[False, True]
    )

    y = group["label"].to_numpy(dtype=float)

    positives = y.sum()

    if positives == 0:
        continue

    row = {
        "model": model,
        "asof_year": year,
        "country": country,
        "rows": len(y),
        "pos": positives,
        "base": y.mean()
    }

    for k in KS:

        row[f"p{k}"] = y[:k].mean()

        row[f"r{k}"] = (
            y[:k].sum() / positives
        )

    row["hit20"] = float(
        y[:20].sum() > 0
    )

    unit.append(row)


U = pd.DataFrame(unit)

if U.empty:
    sys.exit("No country/year groups have positive labels.")

METRICS = [
    "p10", "p20", "p50", "p100",
    "r20", "r50", "r100", "hit20"
]


# ============================================================
# 7. SUMMARY TABLE
# ============================================================

def make_table(group_cols, data):

    output = []

    for group_key, group in data.groupby(group_cols):

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        model = group["model"].iloc[0]

        row = dict(
            zip(group_cols, group_key)
        )

        row.update(
            countries=group["country"].nunique(),
            units=len(group),
            positives=int(group["pos"].sum()),
            base_rate=group["base"].mean()
        )

        for metric in METRICS:

            row[metric] = group[metric].mean()

        row["lift20"] = (
            row["p20"] / row["base_rate"]
            if row["base_rate"] > 0
            else np.nan
        )

        selected = allp[
            allp["model"] == model
        ]

        if "asof_year" in group_cols:

            year = group_key[
                group_cols.index("asof_year")
            ]

            selected = selected[
                selected["asof_year"] == year
            ]

        selected = selected[
            selected["country"].isin(
                group["country"].unique()
            )
        ]

        try:

            row["pr_auc"] = average_precision_score(
                selected["label"],
                selected["score"]
            )

            row["roc_auc"] = roc_auc_score(
                selected["label"],
                selected["score"]
            )

        except ValueError:

            row["pr_auc"] = np.nan
            row["roc_auc"] = np.nan

        output.append(row)

    return pd.DataFrame(output)


U["window"] = U["asof_year"].astype(str)

per = make_table(
    ["model", "asof_year"],
    U
)

U2 = U.copy()

U2["pooled"] = "ALL"

pool = make_table(
    ["model", "pooled"],
    U2
)

pool = pool.rename(
    columns={"pooled": "asof_year"}
)

res = pd.concat(
    [per, pool],
    ignore_index=True
)

os.makedirs(
    a.out,
    exist_ok=True
)

res.to_csv(
    os.path.join(
        a.out,
        "model_comparison_summary.csv"
    ),
    index=False
)

show = res.copy()

for col in [
    "base_rate", "p10", "p20", "p50", "p100",
    "r20", "r50", "r100", "hit20",
    "pr_auc", "roc_auc"
]:

    show[col] = show[col].round(4)

show["lift20"] = show["lift20"].round(2)

print("\n========== MODEL COMPARISON ==========")

print(
    show.sort_values(
        ["asof_year", "model"],
        key=lambda s: s.astype(str)
    ).to_string(index=False)
)


# ============================================================
# 8. PAIRED COUNTRY BOOTSTRAP
# ============================================================

print("\n========== COUNTRY BOOTSTRAP ==========")

ref = a.ref if a.ref in models else None

if ref is None:

    print(
        f"Reference model '{a.ref}' not found in {models}."
    )

    sys.exit(0)

if a.boot <= 0:
    sys.exit("--boot must be greater than zero.")

rng = np.random.default_rng(42)

rows = []

for window in (
    sorted(U["asof_year"].unique()) + ["ALL"]
):

    sub = (
        U if window == "ALL"
        else U[U["asof_year"] == window]
    )

    countries = sorted(
        sub["country"].unique()
    )

    if len(countries) < 3:
        continue

    cidx = {
        country: i
        for i, country in enumerate(countries)
    }

    S = {}

    cnt_c = np.zeros(
        len(countries)
    )

    for model in models:

        s = sub[
            sub["model"] == model
        ]

        mat = np.zeros(
            (len(countries), len(METRICS))
        )

        for country, group in s.groupby("country"):

            i = cidx[country]

            mat[i] = group[
                METRICS
            ].sum().to_numpy()

            if model == models[0]:
                cnt_c[i] = len(group)

        S[model] = mat

    n = len(countries)

    W = rng.multinomial(
        n,
        np.ones(n) / n,
        size=a.boot
    ).astype(float)

    denominator = (
        W @ cnt_c
    )[:, None]

    means = {
        model: (W @ S[model]) / denominator
        for model in models
    }

    for model in models:

        if model == ref:
            continue

        d = (
            means[model] - means[ref]
        )

        for j, metric in enumerate(METRICS):

            if metric not in (
                "p20", "r50", "r100", "hit20"
            ):
                continue

            low, high = np.percentile(
                d[:, j],
                [2.5, 97.5]
            )

            point = (
                S[model][:, j].sum()
                - S[ref][:, j].sum()
            ) / cnt_c.sum()

            rows.append({

                "window": window,

                "model": model,

                "vs": ref,

                "metric": metric,

                "delta": round(point, 4),

                "ci_low": round(low, 4),

                "ci_high": round(high, 4),

                "share_boot_gt0": round(
                    (d[:, j] > 0).mean(),
                    3
                ),

                "countries": n

            })


D = pd.DataFrame(rows)

D.to_csv(
    os.path.join(
        a.out,
        "model_paired_differences.csv"
    ),
    index=False
)

print(
    f"\nPaired differences vs {ref}:"
)

if D.empty:

    print("No paired differences available.")

else:

    print(
        D.to_string(index=False)
    )


# ============================================================
# 9. FINAL OUTPUT
# ============================================================

print("\n========================================")
print("COMPARISON COMPLETED")
print("========================================")

print(
    f"Summary: {a.out}/model_comparison_summary.csv"
)

print(
    f"Bootstrap: {a.out}/model_paired_differences.csv"
)

print(
    "\nA confidence interval excluding zero indicates "
    "evidence of a difference from the reference model."
)
