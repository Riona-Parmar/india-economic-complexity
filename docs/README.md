
# P2 — Product Proximity, Density & Baseline Evaluation

## Overview

This module implements the product-space analysis and baseline evaluation pipeline for **What Should India Export Next?**

The objective is to quantify how closely related different export products are and how feasible it is for a country to develop a comparative advantage in a new product.

P2 is responsible for:

- Constructing product presence matrices using Revealed Comparative Advantage (RCA).
- Computing pairwise product proximity.
- Calculating country-product density scores.
- Implementing two baseline prediction models.
- Evaluating predictions using a common evaluation harness.
- Generating baseline metrics and Precision-Recall curves.

## 1. Methodology

### 1.1 Product Presence Matrix

For each as-of year, a binary country-product matrix is constructed:

\[
m_{cp} =
\begin{cases}
1 & \text{if } RCA3_{cp} \geq 1\\
0 & \text{otherwise}
\end{cases}
\]

Only data available up to the corresponding as-of year is used.

The matrices are generated for:

- A2000
- A2005
- A2010
- A2015

### 1.2 Product Proximity

Product proximity measures how frequently two products are exported with comparative advantage by the same countries.

For products \(i\) and \(j\):

\[
\phi(i,j)=\min\left(
P(m_i=1\mid m_j=1),
P(m_j=1\mid m_i=1)
\right)
\]

Equivalently, using country-level co-occurrence counts:

\[
\phi(i,j)=\frac{C(i,j)}{\max(N_i,N_j)}
\]

Where:

- \(C(i,j)\) is the number of countries having comparative advantage in both products.
- \(N_i\) and \(N_j\) are the number of countries having comparative advantage in products \(i\) and \(j\), respectively.

The resulting proximity matrix is symmetric, with values between 0 and 1.

### 1.3 Product Density

Density estimates how closely a country's existing export basket is related to a potential new product.

\[
Density(c,p)=
\frac{\sum_j \phi(p,j)\times m(c,j)}
{\sum_j \phi(p,j)}
\]

Where:

- \(c\) represents a country.
- \(p\) represents a product being evaluated.
- \(j\) iterates over all products.
- \(m(c,j)\) indicates whether country \(c\) has comparative advantage in product \(j\).

A higher density indicates that the country already exports products closely related to the target product.

## 2. Baseline Models

Two baseline approaches are implemented to establish reference performance for subsequent predictive models.

### Baseline 1: RCA Persistence

Uses the latest available RCA3 value at the as-of year as the prediction score.

\[
Score_{RCA}(c,p,y)=RCA3(c,p,y)
\]

This baseline tests whether existing comparative advantage is a useful indicator of future export performance.

### Baseline 2: Product Proximity

Uses the calculated density score as the prediction score.

\[
Score_{Proximity}(c,p,y)=Density(c,p,y)
\]

This baseline tests whether relatedness to a country's existing export basket can help identify products in which it may develop future comparative advantage.

## 3. Evaluation Framework

Both baselines are evaluated using a common evaluation harness.

Evaluation is performed for each as-of year and for three subsets:

- **India:** Indian test observations.
- **Peers:** Peer-country test observations.
- **All:** Combined test observations.

The evaluation metrics include:

| Metric | Description |
|---|---|
| Precision@10 | Precision among the top 10 ranked products |
| Precision@20 | Precision among the top 20 ranked products |
| Recall@20 | Fraction of positive observations captured in the top 20 |
| Average Precision (AP) | Ranking quality measured using the precision-recall curve |
| Base Rate | Fraction of positive observations in the evaluated subset |
| Lift | Improvement in top-ranked precision relative to the base rate |

Products are ranked within each country for the corresponding as-of year.

The evaluation harness also validates prediction inputs to detect missing scores, invalid values, and duplicate prediction keys.

## 4. Project Structure

```text
p2/
│
├── README.md
│
└── data-p2/
    ├── matrices/
    │   ├── matrix_A2000.parquet
    │   ├── matrix_A2005.parquet
    │   ├── matrix_A2010.parquet
    │   └── matrix_A2015.parquet
    │
    ├── proximity/
    │   ├── proximity_A2000.parquet
    │   ├── proximity_A2005.parquet
    │   ├── proximity_A2010.parquet
    │   └── proximity_A2015.parquet
    │
    ├── density/
    │   ├── density_A2000.parquet
    │   ├── density_A2005.parquet
    │   ├── density_A2010.parquet
    │   └── density_A2015.parquet
    │
    ├── predictions/
    │   ├── predictions_rca.parquet
    │   └── predictions_density.parquet
    │
    ├── results/
    │   ├── metrics_rca.csv
    │   ├── metrics_density.csv
    │   └── pr_curves_baselines/
    │
    └── src/
        ├── matrix.py
        ├── proximity.py
        ├── density.py
        ├── baseline.py
        ├── baseline_density.py
        ├── evaluation.py
        ├── check_density.py
        └── pr-curves_baseline.py
```

## 5. Generated Outputs

| Output | Purpose |
|---|---|
| `matrix_A*.parquet` | Binary country-product presence matrices |
| `proximity_A*.parquet` | Pairwise product proximity values |
| `density_A*.parquet` | Country-product density scores |
| `predictions_rca.parquet` | RCA persistence baseline predictions |
| `predictions_density.parquet` | Proximity-density baseline predictions |
| `metrics_rca.csv` | Evaluation metrics for RCA persistence |
| `metrics_density.csv` | Evaluation metrics for proximity-density baseline |
| `pr_curves_baselines/` | Precision-Recall plots for baseline evaluation |

All generated datasets are stored in Parquet format, while evaluation summaries are stored as CSV files.

## 6. Sanity Checks

The density implementation is checked against future comparative-advantage labels.

A positive correlation between density and the target label is expected because products related to a country's existing export basket should, in general, be more feasible to develop.

The observed correlations were positive across all evaluated as-of years:

| As-of Year | Density-Label Correlation |
|---|---:|
| 2000 | 0.0249 |
| 2005 | 0.0212 |
| 2010 | 0.0845 |
| 2015 | 0.0432 |

These checks provide a directional sanity check for the implementation; they do not independently establish predictive performance.

## 7. Technology Stack

- **Python**
- **Pandas** — Data manipulation and processing
- **NumPy** — Matrix operations and numerical computation
- **Scikit-learn** — Evaluation metrics and precision-recall calculations
- **Matplotlib** — Precision-Recall visualizations
- **Parquet** — Efficient storage of intermediate and generated datasets

## 8. Reproducibility & Data Dependencies

P2 uses upstream P1 outputs, particularly the processed trade/RCA dataset and the generated labels.

These input datasets must be available at the paths expected by the scripts before executing the pipeline.

The proximity calculation must use only information available up to the corresponding as-of year to avoid future-data leakage.

The evaluation harness is the common source of reported baseline metrics. Both baselines are evaluated using the same test labels and evaluation procedure.

## 9. Contribution Summary

P2 establishes the product-space foundation and reference prediction performance for the overall project.

It transforms historical export data into product proximity and density features, implements RCA persistence and proximity-density baselines, and provides a consistent evaluation framework for comparing subsequent machine-learning models against these reference approaches.
