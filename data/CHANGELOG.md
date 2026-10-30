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