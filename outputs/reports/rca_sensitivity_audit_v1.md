# RCA Sensitivity Audit

## Scope

Compared the project's calculated standard Balassa RCA and rolling RCA3
against the Atlas source `export_rca` field, for the countries in the project
trade panel.

## Assumptions and limitations

- The source comparison file has no row for some country-product-year keys
  in the complete project grid. These absent rows were assigned source RCA = 0
  for this sensitivity exercise.
- Explicitly missing source `export_rca` values were replaced with the
  calculated RCA for those records. This affects 0 rows.
- These are sensitivity assumptions, not proof of the Atlas source's intended
  missing-value semantics.
- RCA3 uses a rolling mean of up to three years, with `min_periods=1`.
- The sensitivity labels apply the project's strict historical eligibility
  rule and target RCA3 >= 1.0.

## Results

- Project trade-panel rows: 6,339,300
- Source rows absent from the project grid: 632,605
- Explicitly missing source RCA values using calculated-RCA fallback: 0
- Historical labeled rows compared: 466,856
- Eligibility status disagreements: 6,325
- Source-vs-calculated rebuilt label disagreements: 750
- Sensitivity labels vs existing project labels: 750

See `data/interim/rca_sensitivity_label_disagreements.csv` for affected rows.
