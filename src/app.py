"""P5 Streamlit dashboard: India's product diversification opportunities.

Tabs: Ranking, Backtest, Model Comparison, Product Space, RCA Trend, Data checks.

Run from the repository root:
    streamlit run src/app.py
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
INDIA = "IND"

# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------
# Each entry lists the contract path first, followed by any older file names
# currently in the repo. The first one that exists is loaded.
CONTRACT_FILES: dict[str, list[str]] = {
    "products": ["data/processed/products_v1.parquet"],
    "trade_rca": [
        "data/processed/trade_rca.parquet",
        "data/processed/trade_rca_v1.parquet",
    ],
    "metrics": [
        "outputs/metrics.parquet",
        "outputs/metrics_xgboost_v1.parquet",
    ],
    # Ground-truth outcomes for the backtest (P1 labels; NaN for forecast years).
    "labels": ["data/processed/labels_v1.parquet"],
}

# P4 interactive product space (Plotly figure saved with fig.write_html).
NETWORK_HTML = "outputs/network/product_space_2010_p3.html"

# SHAP explanations from src/explain_xgboost.py / src/combine_explanations.py.
EXPLANATION_FILES = [
    "outputs/india_final_explanations.csv",
    "outputs/india_shap_explanations.csv",
]
SHAP_FEATURE_NAMES = {
    "density": "Proximity density to India's current exports",
    "pci_current": "Product complexity (PCI)",
    "rca3_current": "India's current 3-yr RCA",
    "rca3_change_3y": "Change in India's 3-yr RCA over 3 years",
    "eci_current": "India's economic complexity (ECI)",
    "f_ubiquity": "Ubiquity (number of countries exporting it)",
    "f_world_growth": "World export growth of the product",
}

FORECAST_LABEL = "Forecast: no outcome yet"
BACKTEST_TOP_K = 20

# Predictions may be split across one file per model: outputs/predictions_*.parquet
PREDICTIONS_GLOB = "outputs/predictions_*.parquet"

PREDICTION_COLUMNS = ["asof_year", "country", "product", "model", "score"]
ALLOWED_MODELS = {"rca_persistence", "proximity", "xgboost"}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_parquet(path: str, filters: tuple | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, filters=list(filters) if filters else None)


@st.cache_data(show_spinner=False)
def _read_parquets(paths: tuple[str, ...]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _resolve(candidates: list[str]) -> Path | None:
    for rel in candidates:
        path = ROOT / rel
        if path.is_file():
            return path
    return None


def load_contract_file(name: str, filters: tuple | None = None) -> pd.DataFrame | None:
    """Load one contract file; show a message and return None on failure.

    `filters` is passed to pyarrow as row filters (as a tuple so it can be cached).
    """
    candidates = CONTRACT_FILES[name]
    path = _resolve(candidates)
    if path is None:
        st.warning(
            f"Missing file for **{name}**: expected `{candidates[0]}`"
            + (
                f" (also checked {', '.join(f'`{c}`' for c in candidates[1:])})"
                if len(candidates) > 1
                else ""
            )
        )
        return None
    try:
        return _read_parquet(str(path), filters)
    except Exception as exc:  # corrupt file, missing parquet engine, etc.
        st.error(f"Could not read `{path.relative_to(ROOT).as_posix()}`: {exc}")
        return None


def load_predictions() -> pd.DataFrame | None:
    """Load and stack every outputs/predictions_*.parquet file."""
    paths = sorted(ROOT.glob(PREDICTIONS_GLOB))
    if not paths:
        st.warning(f"Missing file for **predictions**: no file matches `{PREDICTIONS_GLOB}`")
        return None
    try:
        return _read_parquets(tuple(str(p) for p in paths))
    except Exception as exc:
        names = ", ".join(f"`{p.relative_to(ROOT).as_posix()}`" for p in paths)
        st.error(f"Could not read predictions from {names}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def validate_predictions(df: pd.DataFrame) -> list[str]:
    """Return a list of contract violations; an empty list means the test passed."""
    errors: list[str] = []

    columns = list(df.columns)
    missing = [c for c in PREDICTION_COLUMNS if c not in columns]
    extra = [c for c in columns if c not in PREDICTION_COLUMNS]
    if missing:
        errors.append(f"Missing columns: {missing}")
    if extra:
        errors.append(f"Unexpected columns: {extra}")

    if "score" in df.columns:
        n_nan = int(df["score"].isna().sum())
        if n_nan:
            errors.append(f"`score` has {n_nan:,} NaN value(s)")

    if "model" in df.columns:
        unknown = sorted(set(df["model"].dropna().astype(str)) - ALLOWED_MODELS)
        if unknown:
            errors.append(
                f"Unknown model name(s): {unknown}; allowed: {sorted(ALLOWED_MODELS)}"
            )
        if df["model"].isna().any():
            errors.append("`model` has missing values")

    return errors


@st.cache_data(show_spinner=False)
def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"product": str, "candidate_product": str})


def load_explanations() -> pd.DataFrame | None:
    """SHAP reasons as columns asof_year, product, probability, top_3_reasons (+ related)."""
    path = _resolve(EXPLANATION_FILES)
    if path is None:
        return None
    df = _read_csv(str(path)).rename(
        columns={"candidate_product": "product", "candidate_probability": "probability"}
    )
    df["product"] = df["product"].str.zfill(4)
    return df


# ---------------------------------------------------------------------------
# Shared helpers: outcomes, forecast years, backtest precision
# ---------------------------------------------------------------------------
def outcome_years(labels: pd.DataFrame | None) -> set[int] | None:
    """As-of years whose outcome is known (at least one non-null label).

    Returns None when labels are unavailable, meaning "unknown".
    """
    if labels is None:
        return None
    return set(labels.dropna(subset=["label"])["asof_year"].astype(int).unique())


def is_forecast(year: int, known: set[int] | None) -> bool:
    """A year is a forecast when labels exist but none of them has an outcome yet."""
    return known is not None and int(year) not in known


def year_label(year: int, known: set[int] | None) -> str:
    return f"{int(year)} · {FORECAST_LABEL}" if is_forecast(year, known) else str(int(year))


def fmt_ci(value: float, lo: float, hi: float) -> str:
    if pd.isna(value):
        return "—"
    if pd.isna(lo) or pd.isna(hi):
        return f"{value:.2f}"
    return f"{value:.2f} [{lo:.2f}–{hi:.2f}]"


def normalise_metrics(metrics: pd.DataFrame | None) -> pd.DataFrame | None:
    """Older metrics files have no `model` column; they were written for xgboost."""
    if metrics is None:
        return None
    if "model" not in metrics.columns:
        metrics = metrics.assign(model="xgboost")
    return metrics


def backtest_precision(
    metrics: pd.DataFrame | None, model: str, year: int
) -> tuple[str, int | None]:
    """India Precision@20 (with 95% CI) for this model at `year`.

    For a forecast year (no metrics row), falls back to the latest earlier
    backtest year. Returns (formatted text, year used).
    """
    if metrics is None:
        return "—", None
    rows = metrics[
        (metrics["model"] == model)
        & (metrics["subset"] == "india")
        & (metrics["asof_year"] <= year)
    ].sort_values("asof_year")
    if rows.empty:
        return "—", None
    r = rows.iloc[-1]
    return (
        fmt_ci(r["Precision@20"], r.get("P20_CI_lo"), r.get("P20_CI_hi")),
        int(r["asof_year"]),
    )


def show_smoke_test(df: pd.DataFrame) -> bool:
    errors = validate_predictions(df)
    if errors:
        st.error(
            "Predictions smoke test failed:\n\n" + "\n".join(f"- {e}" for e in errors)
        )
        return False
    st.success(
        f"Predictions smoke test passed: {len(df):,} rows, "
        f"models {sorted(df['model'].unique())}."
    )
    return True


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_sidebar(
    predictions: pd.DataFrame | None,
    products: pd.DataFrame | None,
    known_years: set[int] | None,
) -> dict:
    """Draw the sidebar controls and return the current selections."""
    sb = st.sidebar
    sb.header("Filters")

    models = sorted(predictions["model"].dropna().unique()) if predictions is not None else []
    model = sb.selectbox("Model", models, disabled=not models)

    # Only offer years the chosen model actually has predictions for.
    years: list[int] = []
    if predictions is not None and model is not None:
        years = sorted(predictions.loc[predictions["model"] == model, "asof_year"].unique())
    year = sb.selectbox(
        "As-of year",
        years,
        index=len(years) - 1 if years else None,
        format_func=lambda y: year_label(y, known_years),
        disabled=not years,
    )

    sectors = sorted(products["sector"].dropna().unique()) if products is not None else []
    categories = sb.multiselect(
        "Product category",
        sectors,
        placeholder="All categories",
        disabled=not sectors,
    )

    top_n = sb.number_input("Top-N", min_value=1, max_value=1000, value=20, step=1)

    return {"model": model, "year": year, "categories": categories, "top_n": int(top_n)}


# What the `score` column means for each model, shown as the Feasibility tooltip.
SCORE_MEANING = {
    "xgboost": "predicted probability that India develops RCA in this product",
    "proximity": "proximity density: how close the product is to India's current export basket",
    "rca_persistence": "India's current 3-year average RCA in the product",
}


def pci_percentiles(products: pd.DataFrame) -> pd.Series:
    """PCI percentile (0–1) of each product among *all* products in products_v1.

    Computed on the full catalogue, so a product's percentile doesn't change
    when the sector filter changes.
    """
    return products.set_index("product")["pci_latest"].rank(pct=True)


def build_ranking(
    predictions: pd.DataFrame,
    products: pd.DataFrame,
    model: str,
    year: int,
    sectors: list[str],
    top_n: int,
) -> pd.DataFrame:
    """India's top-N products for one model and as-of year, ranked by rank_score.

    rank_score = score (feasibility) × PCI percentile.
    Names, sectors and PCI come only from products_v1.
    """
    preds = predictions.loc[
        (predictions["model"] == model)
        & (predictions["asof_year"] == year)
        & (predictions["country"] == INDIA),
        ["product", "score"],
    ]

    catalog = products[["product", "name", "sector", "pci_latest"]].copy()
    catalog["pci_pct"] = catalog["product"].map(pci_percentiles(products))

    ranked = preds.merge(catalog, on="product", how="left", validate="one_to_one")
    if sectors:
        ranked = ranked[ranked["sector"].isin(sectors)]

    ranked["rank_score"] = ranked["score"] * ranked["pci_pct"]
    ranked = (
        ranked.sort_values(["rank_score", "product"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    # Placeholder: Atlas complexity outlook gain, to be filled for the forecast year only.
    ranked["atlas_cog"] = float("nan")

    return ranked[
        ["rank", "product", "name", "sector", "pci_latest",
         "score", "pci_pct", "rank_score", "atlas_cog"]
    ]


def feasibility_column(model: str, max_score: float):
    return st.column_config.ProgressColumn(
        "Feasibility",
        format="%.3f",
        min_value=0.0,
        max_value=max(1.0, max_score),
        help=f"Model score for {model}: {SCORE_MEANING.get(model, 'model score')}.",
    )


def backtest_p20_column(model: str, used_year: int | None):
    return st.column_config.TextColumn(
        "Backtest P@20 [95% CI]",
        help=(
            f"How often {model}'s top-20 for India was right in the {used_year} backtest "
            "(Precision@20 from metrics.parquet). Read the Feasibility next to it in that light."
        ),
    )


def render_ranking_tab(
    predictions: pd.DataFrame | None,
    products: pd.DataFrame | None,
    metrics: pd.DataFrame | None,
    explanations: pd.DataFrame | None,
    known_years: set[int] | None,
    filters: dict,
) -> None:
    st.subheader(f"{INDIA} product ranking")
    if predictions is None or products is None:
        st.warning("Needs both valid predictions and products to be loaded.")
        return
    if filters["model"] is None or filters["year"] is None:
        st.info("Choose a model and as-of year in the sidebar.")
        return

    model, year = filters["model"], int(filters["year"])
    table = build_ranking(
        predictions, products, model, year, filters["categories"], filters["top_n"]
    )
    if table.empty:
        st.info("No candidate products match the current filters.")
        return

    if is_forecast(year, known_years):
        st.warning(f"As-of {year}: **{FORECAST_LABEL}**. Atlas cog applies to this year.")

    p20_text, p20_year = backtest_precision(metrics, model, year)
    table.insert(table.columns.get_loc("score"), "backtest_p20", p20_text)

    st.markdown(
        f"Top **{len(table)}** for **{model}**, as of **{year_label(year, known_years)}**. "
        "`rank_score = Feasibility × PCI percentile`"
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "product": st.column_config.TextColumn("HS code", width="small"),
            "name": st.column_config.TextColumn("Product Name", width="large"),
            "sector": st.column_config.TextColumn("Sector"),
            "pci_latest": st.column_config.NumberColumn(
                "PCI", format="%.3f", help="Product Complexity Index (products_v1)"
            ),
            "backtest_p20": backtest_p20_column(model, p20_year),
            "score": st.column_config.ProgressColumn(
                "Feasibility",
                format="%.3f",
                min_value=0.0,
                max_value=max(1.0, float(table["score"].max())),
                help=f"Model score for {model}: {SCORE_MEANING.get(model, 'model score')}.",
            ),
            "pci_pct": st.column_config.ProgressColumn(
                "× PCI percentile",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
                help="Share of all products in products_v1 with PCI at or below this one.",
            ),
            "rank_score": st.column_config.NumberColumn(
                "= rank_score", format="%.4f", help="Feasibility × PCI percentile"
            ),
            "atlas_cog": st.column_config.NumberColumn(
                "Atlas cog",
                format="%.3f",
                help="Placeholder: Atlas complexity outlook gain, filled for the forecast year.",
            ),
        },
    )
    if model != "xgboost":
        st.caption(
            f"Note: for **{model}**, Feasibility is {SCORE_MEANING[model]}, "
            "not a probability."
        )
    if p20_year is not None and p20_year != year:
        st.caption(f"Backtest P@20 is from {p20_year}, the latest year with known outcomes.")

    render_reasons(table, model, year, explanations)


def parse_reasons(text: str) -> list[tuple[str, str]]:
    """'density increases; pci_current decreases' -> [(readable feature, direction)]."""
    out = []
    for part in str(text).split(";"):
        words = part.strip().rsplit(" ", 1)
        if len(words) == 2:
            feature, direction = words
            out.append((SHAP_FEATURE_NAMES.get(feature, feature), direction))
    return out


def render_reasons(
    table: pd.DataFrame, model: str, year: int, explanations: pd.DataFrame | None
) -> None:
    """'Reasons' expander: why one ranked product got its score."""
    with st.expander("Reasons: why did this product get its score?"):
        labels = dict(zip(table["product"], table["product"] + " · " + table["name"].fillna("")))
        product = st.selectbox(
            "Product", list(labels), format_func=lambda p: labels.get(p, p), key="reason_product"
        )
        row = table.loc[table["product"] == product].iloc[0]

        st.markdown(
            f"**Score breakdown:** Feasibility {row['score']:.3f} × "
            f"PCI percentile {row['pci_pct']:.3f} = rank_score **{row['rank_score']:.4f}** "
            f"(PCI {row['pci_latest']:.3f})"
        )

        if model != "xgboost":
            st.info(
                f"SHAP reasons exist only for xgboost. For **{model}** the score is "
                f"{SCORE_MEANING.get(model, 'the model score')}."
            )
            return
        if explanations is None:
            st.info(
                "No SHAP explanations found. Expected "
                + " or ".join(f"`{p}`" for p in EXPLANATION_FILES)
                + " (made by `src/explain_xgboost.py` and `src/combine_explanations.py`)."
            )
            return

        match = explanations[
            (explanations["asof_year"] == year) & (explanations["product"] == product)
        ]
        if match.empty:
            st.info(f"No SHAP explanation saved for {product} in {year}.")
            return

        first = match.iloc[0]
        st.markdown(f"**Top SHAP drivers** (xgboost probability {first['probability']:.3f}):")
        for feature, direction in parse_reasons(first["top_3_reasons"]):
            arrow = "▲ raises" if direction == "increases" else "▼ lowers"
            st.markdown(f"- {arrow} the score: {feature}")

        if "related_product" in match.columns and match["related_product"].notna().any():
            st.markdown("**Closest products India already exports:**")
            st.dataframe(
                match[["related_product", "related_name", "proximity"]]
                .dropna(subset=["related_product"])
                .rename(columns={"related_product": "HS code", "related_name": "Product",
                                 "proximity": "Proximity"}),
                hide_index=True,
                width="stretch",
            )


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def build_backtest(
    predictions: pd.DataFrame,
    products: pd.DataFrame,
    labels: pd.DataFrame,
    india_rca: pd.DataFrame | None,
    model: str,
    year: int,
) -> pd.DataFrame:
    """Every India candidate for (model, year) with its model rank and actual outcome.

    Ranked by raw score, the way Precision@20 is computed in the P2 harness.
    `hit` = label 1: RCA3 rose from below 0.25 to ≥ 1 by the target year.
    """
    preds = predictions.loc[
        (predictions["model"] == model)
        & (predictions["asof_year"] == year)
        & (predictions["country"] == INDIA),
        ["product", "score"],
    ]
    outcome = labels.loc[labels["asof_year"] == year, ["product", "target_year", "label"]]
    df = (
        preds.merge(outcome, on="product", how="left", validate="one_to_one")
        .merge(products[["product", "name", "sector"]], on="product", how="left")
        .sort_values(["score", "product"], ascending=[False, True])
        .reset_index(drop=True)
    )
    df.insert(0, "rank", range(1, len(df) + 1))
    df["hit"] = df["label"].eq(1)

    df["rca3_target"] = float("nan")
    if india_rca is not None and df["target_year"].notna().any():
        target = int(df["target_year"].dropna().iloc[0])
        rca3 = india_rca.loc[india_rca["year"] == target].set_index("product")["rca3"]
        df["rca3_target"] = df["product"].map(rca3)
    return df


def render_backtest_tab(
    predictions: pd.DataFrame | None,
    products: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    india_rca: pd.DataFrame | None,
    metrics: pd.DataFrame | None,
    known_years: set[int] | None,
    filters: dict,
) -> None:
    st.subheader("Backtest: top-20 vs. what actually happened")
    if predictions is None or products is None or labels is None:
        st.warning("Needs predictions, products and labels to be loaded.")
        return
    model, year = filters["model"], filters["year"]
    if model is None or year is None:
        st.info("Choose a model and as-of year in the sidebar.")
        return
    year = int(year)
    if is_forecast(year, known_years):
        st.info(
            f"As-of {year}: **{FORECAST_LABEL}**. Pick an earlier as-of year in the sidebar "
            "to compare predictions with outcomes."
        )
        return

    df = build_backtest(predictions, products, labels, india_rca, model, year)
    target = int(df["target_year"].dropna().iloc[0])
    top = df.head(BACKTEST_TOP_K).copy()
    hits = int(top["hit"].sum())
    actual = df[df["hit"]]
    p20_text, _ = backtest_precision(metrics, model, year)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Hits in top {BACKTEST_TOP_K}", f"{hits} / {len(top)}")
    c2.metric("Backtest P@20 [95% CI]", p20_text)
    c3.metric(f"Products India actually reached ({target})", f"{len(actual)} / {len(df)}")
    st.caption(
        f"Model **{model}**, as of **{year}**, outcome in **{target}**. A hit is a product whose "
        "3-yr RCA was below 0.25 through the as-of year and reached ≥ 1 by the target year "
        "(the labels_v1 rule). The ranking uses the raw model score, as Precision@20 does, "
        "and ignores the sector filter."
    )

    top["outcome"] = top["hit"].map({True: "✅ Hit", False: "—"})
    top.insert(top.columns.get_loc("score"), "backtest_p20", p20_text)
    view = top[["rank", "product", "name", "sector", "backtest_p20", "score",
                "rca3_target", "outcome"]]
    styled = view.style.apply(
        lambda r: ["background-color: rgba(34, 197, 94, 0.25)" if top.loc[r.name, "hit"]
                   else "" for _ in r],
        axis=1,
    ).format({"score": "{:.3f}", "rca3_target": "{:.2f}"}, na_rep="—")
    st.dataframe(
        styled,
        hide_index=True,
        width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "product": st.column_config.TextColumn("HS code", width="small"),
            "name": st.column_config.TextColumn("Product Name", width="large"),
            "sector": st.column_config.TextColumn("Sector"),
            "backtest_p20": backtest_p20_column(model, year),
            "score": st.column_config.NumberColumn(
                "Feasibility", help=SCORE_MEANING.get(model, "model score")
            ),
            "rca3_target": st.column_config.NumberColumn(
                f"Actual 3-yr RCA ({target})", help="From trade_rca; ≥ 1 is a hit."
            ),
            "outcome": st.column_config.TextColumn("Outcome"),
        },
    )

    st.markdown(f"**Every product India actually reached by {target}**, with the model's rank:")
    if actual.empty:
        st.info("None. India reached no new products in this window.")
    else:
        st.dataframe(
            actual[["rank", "product", "name", "sector", "score", "rca3_target"]],
            hide_index=True,
            width="stretch",
            column_config={
                "rank": st.column_config.NumberColumn(
                    f"Model rank (of {len(df)})", help=f"Inside the top {BACKTEST_TOP_K} = hit"
                ),
                "product": st.column_config.TextColumn("HS code"),
                "name": st.column_config.TextColumn("Product Name", width="large"),
                "sector": st.column_config.TextColumn("Sector"),
                "score": st.column_config.NumberColumn("Feasibility", format="%.3f"),
                "rca3_target": st.column_config.NumberColumn(
                    f"Actual 3-yr RCA ({target})", format="%.2f"
                ),
            },
        )


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def build_comparison(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame | None,
    known_years: set[int] | None,
    subset: str,
) -> pd.DataFrame:
    """One row per model × as-of year; forecast years appear with no metrics."""
    m = metrics[metrics["subset"] == subset].copy()
    if predictions is not None:
        pairs = predictions[["model", "asof_year"]].drop_duplicates()
        m = pairs.merge(m, on=["model", "asof_year"], how="outer")
    m["status"] = [
        FORECAST_LABEL if is_forecast(y, known_years) else "Backtest" for y in m["asof_year"]
    ]
    m["P@20 [95% CI]"] = [
        fmt_ci(r["Precision@20"], r.get("P20_CI_lo"), r.get("P20_CI_hi")) for _, r in m.iterrows()
    ]
    m["PR-AUC [95% CI]"] = [
        fmt_ci(r["PR-AUC"], r.get("PRAUC_CI_lo"), r.get("PRAUC_CI_hi")) for _, r in m.iterrows()
    ]
    return m.sort_values(["asof_year", "model"]).reset_index(drop=True)


def ci_chart(df: pd.DataFrame, value: str, lo: str, hi: str, title: str) -> alt.LayerChart:
    base = alt.Chart(df).encode(
        x=alt.X("model:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        color=alt.Color("model:N", title="Model", legend=alt.Legend(orient="top")),
    )
    bars = base.mark_rule(strokeWidth=2).encode(
        y=alt.Y(f"{lo}:Q", title=title), y2=f"{hi}:Q"
    )
    points = base.mark_point(filled=True, size=80).encode(
        y=f"{value}:Q",
        tooltip=["model", "asof_year", alt.Tooltip(f"{value}:Q", format=".3f"),
                 alt.Tooltip(f"{lo}:Q", format=".3f"), alt.Tooltip(f"{hi}:Q", format=".3f")],
    )
    return (bars + points).properties(width=110, height=260).facet(
        column=alt.Column("asof_year:O", title="As-of year")
    )


def render_comparison_tab(
    metrics: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    known_years: set[int] | None,
) -> None:
    st.subheader("Model comparison")
    if metrics is None:
        st.warning("Needs metrics to be loaded.")
        return

    subset = st.radio(
        "Evaluated on", ["india", "peers", "all"], horizontal=True,
        format_func={"india": "India", "peers": "Peer countries", "all": "All"}.get,
    )
    table = build_comparison(metrics, predictions, known_years, subset)

    backtested = table[table["status"] == "Backtest"].dropna(subset=["Precision@20"])
    st.markdown("**Precision@20 with 95% bootstrap CI**")
    st.altair_chart(
        ci_chart(backtested, "Precision@20", "P20_CI_lo", "P20_CI_hi", "Precision@20")
    )
    st.markdown("**PR-AUC with 95% bootstrap CI**")
    st.altair_chart(ci_chart(backtested, "PR-AUC", "PRAUC_CI_lo", "PRAUC_CI_hi", "PR-AUC"))

    st.dataframe(
        table[["asof_year", "model", "status", "P@20 [95% CI]", "Precision@10", "Recall@20",
               "PR-AUC [95% CI]", "base_rate", "lift"]],
        hide_index=True,
        width="stretch",
        column_config={
            "asof_year": st.column_config.NumberColumn("As-of year", format="%d"),
            "model": "Model",
            "status": "Status",
            "Precision@10": st.column_config.NumberColumn("P@10", format="%.2f"),
            "Recall@20": st.column_config.NumberColumn("Recall@20", format="%.2f"),
            "base_rate": st.column_config.NumberColumn(
                "Base rate", format="%.3f", help="Share of candidates that actually emerged"
            ),
            "lift": st.column_config.NumberColumn(
                "Lift", format="%.1f×", help="P@20 ÷ base rate"
            ),
        },
    )
    st.caption(
        "P@20 averages each country's top-20 hit rate; for India it is hits ÷ 20. "
        "CIs: 1,000 bootstrap resamples (seed 42). Wide CIs reflect very few positives "
        "per year, so overlapping intervals mean models are not reliably different."
    )


# ---------------------------------------------------------------------------
# Product space
# ---------------------------------------------------------------------------
def render_product_space_tab() -> None:
    st.subheader("Product space (2010)")
    path = ROOT / NETWORK_HTML
    if not path.is_file():
        st.warning(f"Missing file for **product space**: expected `{NETWORK_HTML}`")
        return
    st.caption(
        "P4 interactive network for as-of 2010 (`src/day7_bootstrap_ci_and_network.py`). "
        "It is fixed to 2010 and does not follow the sidebar. Zoom and hover inside the plot. "
        "Plotly loads from its CDN, so an internet connection is needed."
    )
    # Our own P4 artifact (trusted), embedded as HTML so its Plotly JS can run.
    st.iframe(_read_text(str(path)), height=940)


def render_rca_trend_tab(
    india_rca: pd.DataFrame | None, products: pd.DataFrame | None, categories: list[str]
) -> None:
    st.subheader(f"{INDIA} RCA trend")
    if india_rca is None or products is None:
        st.warning("Needs both trade_rca and products to be loaded.")
        return

    catalog = products[products["product"].isin(india_rca["product"].unique())]
    if categories:
        catalog = catalog[catalog["sector"].isin(categories)]
    if catalog.empty:
        st.info("No products in the selected categories.")
        return

    catalog = catalog.sort_values("product")
    names = catalog["name"].fillna("(unnamed)")
    labels = dict(zip(catalog["product"], catalog["product"] + " · " + names))
    product = st.selectbox(
        "Product", list(labels), format_func=lambda p: labels.get(p, p), key="rca_product"
    )

    trend = (
        india_rca.loc[india_rca["product"] == product, ["year", "rca", "rca3", "export_value"]]
        .sort_values("year")
        .reset_index(drop=True)
    )
    if trend.empty:
        st.info("No trade records for this product.")
        return

    sector = catalog.loc[catalog["product"] == product, "sector"].iloc[0]
    st.caption(f"Sector: {sector} · RCA ≥ 1 means India has a revealed comparative advantage.")

    latest = trend.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"RCA ({int(latest['year'])})", f"{latest['rca']:.2f}")
    c2.metric("3-year avg RCA", f"{latest['rca3']:.2f}")
    c3.metric("Years with RCA ≥ 1", f"{int((trend['rca'] >= 1).sum())} / {len(trend)}")

    long = trend.melt(
        id_vars="year", value_vars=["rca", "rca3"], var_name="series", value_name="value"
    )
    long["series"] = long["series"].map({"rca": "RCA", "rca3": "RCA (3-yr avg)"})
    lines = (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("value:Q", title="RCA"),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top")),
            tooltip=["year", "series", alt.Tooltip("value:Q", format=".3f")],
        )
    )
    threshold = (
        alt.Chart(pd.DataFrame({"y": [1.0]}))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(y="y:Q")
    )
    st.altair_chart(lines + threshold, width="stretch")

    with st.expander("Show data"):
        st.dataframe(
            trend.rename(
                columns={"year": "Year", "rca": "RCA", "rca3": "RCA (3-yr avg)",
                         "export_value": "Export value (USD)"}
            ),
            hide_index=True,
            width="stretch",
        )


def render_data_checks_tab(data: dict[str, pd.DataFrame | None]) -> None:
    st.subheader("Data contract")
    loaded = {k: v for k, v in data.items() if v is not None}
    if loaded:
        st.caption(
            "Loaded: " + ", ".join(f"{k} ({len(v):,} rows)" for k, v in loaded.items())
        )
    st.subheader("Smoke test")
    if data["predictions"] is None:
        st.error("Smoke test skipped: predictions could not be loaded.")
    else:
        show_smoke_test(data["predictions"])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="India Economic Complexity", layout="wide")
    st.title("India Economic Complexity Dashboard")

    data = {
        "products": load_contract_file("products"),
        # Only India's rows are needed; filtering at read time skips ~6M rows.
        "trade_rca": load_contract_file("trade_rca", (("country", "==", INDIA),)),
        "metrics": load_contract_file("metrics"),
        "labels": load_contract_file("labels", (("country", "==", INDIA),)),
        "predictions": load_predictions(),
    }
    metrics = normalise_metrics(data["metrics"])
    explanations = load_explanations()
    known_years = outcome_years(data["labels"])

    # Surface a failed smoke test above the tabs, not only inside Data checks.
    predictions = data["predictions"]
    if predictions is not None and validate_predictions(predictions):
        st.error("Predictions failed the smoke test. See the Data checks tab.")
        predictions = None  # don't drive the UI from data that breaks the contract

    filters = render_sidebar(predictions, data["products"], known_years)

    ranking_tab, backtest_tab, compare_tab, space_tab, rca_tab, checks_tab = st.tabs(
        ["Ranking", "Backtest", "Model Comparison", "Product Space", "RCA Trend", "Data checks"]
    )
    with ranking_tab:
        render_ranking_tab(
            predictions, data["products"], metrics, explanations, known_years, filters
        )
    with backtest_tab:
        render_backtest_tab(
            predictions, data["products"], data["labels"], data["trade_rca"],
            metrics, known_years, filters,
        )
    with compare_tab:
        render_comparison_tab(metrics, predictions, known_years)
    with space_tab:
        render_product_space_tab()
    with rca_tab:
        render_rca_trend_tab(data["trade_rca"], data["products"], filters["categories"])
    with checks_tab:
        render_data_checks_tab(data)


if __name__ == "__main__":
    main()
