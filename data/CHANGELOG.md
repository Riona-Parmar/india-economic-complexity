## P1 validation update — 2026-10-29

### Artifacts validated

- `data/processed/trade_rca_v1.parquet`
  - Rows: 6,339,300
  - Countries: 170
  - Products: 1,243
  - Years: 1995–2024
  - Missing values in `rca`, `rca3`, and `m`: 0
  - Product codes not four characters long: 0

- `data/processed/labels_v1.parquet`
  - Rows: 559,645
  - Historical rows: 466,856
  - Forecast rows: 92,789
  - Historical missing labels: 0
  - Historical eligibility violations detected by validation script: 0

- `data/processed/products_v1.parquet`
  - Rows: 1,243
  - Required columns present

- `outputs/reports/data_report.md`
  - File exists

### Export-total cross-check

- Atlas Locations view: India → World, 2024, All Products (HS92), Gross; displayed total approximately USD 417 billion.
- Processed dataset: India 2024 export value USD 416,457,390,569.
- Assessment: Consistent at the dashboard's displayed precision. This does not establish record-level correctness.

### India emergence-label positives

| As-of year | Target year | Positive labels |
|---:|---:|---:|
| 2000 | 2005 | 12 |
| 2005 | 2010 | 5 |
| 2010 | 2015 | 4 |
| 2015 | 2020 | 1 |

### Outstanding caveat

The independently calculated standard Balassa RCA does not fully reconcile with the source Atlas `export_rca` values. The discrepancy remains unresolved and must be retained in the audit documentation. Do not describe the source RCA comparison as fully reconciled.

### Handoff status

Core artifact and label-eligibility checks passed. Confirm the Drive upload and team handoff separately. Do not mark every P1 requirement complete until the remaining documentation and unresolved caveats have been addressed.

### P1 validation and RCA sensitivity audit

- Validated the trade panel, product metadata keys, historical labels, forecast rows, country splits, target-year alignment, and strict historical RCA3 eligibility.
- All final P1 integrity checks passed.
- Reconciled export values against the source comparison file: 5,706,695 matched records had zero export-value mismatches. The 632,605 source-absent country–product–year rows had zero project exports.
- Compared the Atlas source `export_rca` with the project's independently calculated standard Balassa RCA. The sensitivity audit found 6,325 historical eligibility disagreements and 750 historical label disagreements across 466,856 historical labeled rows.
- Current project decision: retain calculated standard Balassa RCA and its associated labels as the primary methodology. Preserve the sensitivity audit outputs and document the source-field discrepancy as unresolved.
- Limitation: the sensitivity analysis is not evidence that the two RCA definitions are interchangeable. The precise cause of the source `export_rca` discrepancy remains unverified.
- Status: P1 data construction and internal validation passed, subject to the documented RCA methodology limitation.

## P4 network pruning, interactive visualization, and confidence intervals — 2026-11-07

### Artifacts generated

- `outputs/network/network_A2010_nodes.parquet`
  - Rows: 1,243 products
  - Columns: `product`, `x`, `y`, `name`, `sector`, `pci`
  - Fixed-seed spring layout (`seed=42`, `k=0.3`, `weight=phi`)

- `outputs/network/network_A2010_edges.parquet`
  - Rows: 19,418 pruned edges
  - Columns: `product_i`, `product_j`, `phi`
  - Pruned using NetworkX: Maximum Spanning Tree (1,242 edges) union high-proximity edges ($\phi \ge 0.40$, 19,364 edges), achieving a 97.5% reduction from 771,903 initial undirected edges

- `outputs/network/network_A2010.parquet`
  - Combined sentinel file with `table` tag column containing nodes and edges

- `outputs/network/product_space_2010_p3.html`
  - Standalone interactive Plotly visualization of the product space
  - 1,243 nodes sized by PCI and colored across 20 distinct sectors
  - 19,418 thin pruned edges
  - 373 products exported with comparative advantage by India in 2010 ($m=1$) highlighted with bold 2.5px borders
  - XGBoost top-20 predicted emergence products for India overlaid as gold star markers with probability scores

- `outputs/predictions_xgboost_v1.parquet`
  - Rows: 5,205
  - Columns: `asof_year`, `country`, `product`, `score`
  - Evaluated across test windows 2010 and 2015 for India and peer countries

- `outputs/metrics_xgboost_v1.parquet` & `outputs/metrics_xgboost_v1.csv`
  - Point-estimate evaluation metrics calculated via P2 exact evaluation harness
  - 95% bootstrap confidence intervals (1,000 iterations with replacement, seed 42) for Precision@20 and PR-AUC:
    - 2010 India: Precision@20 = 0.0500 [0.0000, 0.2000]; PR-AUC = 0.1584 [0.0089, 0.6667]
    - 2010 Peers: Precision@20 = 0.0200 [0.0000, 0.0400]; PR-AUC = 0.0093 [0.0053, 0.0189]
    - 2010 All: Precision@20 = 0.0250 [0.0000, 0.0583]; PR-AUC = 0.0208 [0.0069, 0.0903]
    - 2015 India: Precision@20 = 0.0500 [0.0000, 0.1500]; PR-AUC = 0.0667 [0.0476, 0.2308]
    - 2015 Peers: Precision@20 = 0.0100 [0.0000, 0.0400]; PR-AUC = 0.0219 [0.0063, 0.0855]
    - 2015 All: Precision@20 = 0.0167 [0.0000, 0.0417]; PR-AUC = 0.0210 [0.0068, 0.0733]

### Scripts added

- `src/verify_india_trade_and_labels.py`: Verification of India trade RCA exports and historical emergence labels against `data_report.md`.
- `src/build_network_A2010.py`: Proximity matrix smoke tests, MST + $\phi \ge 0.40$ network pruning, fixed-seed spring layout, and node metadata attachment.
- `src/plot_product_space_2010.py`: Day 6 interactive Plotly visualization generation.
- `src/day7_bootstrap_ci_and_network.py`: Day 7 1,000-iteration bootstrap resampling for 95% CIs and XGBoost top-20 prediction map overlay.
