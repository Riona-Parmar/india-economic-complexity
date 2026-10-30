# Trade RCA and Emergence Label Data Report

## Configuration

- Data years: 1995–2024
- RCA3 eligibility threshold: below 0.25
- Emergence threshold: at least 1.0
- Prediction horizon: 5 years
- Historical windows configured: [2000, 2005, 2010, 2015]
- India test country: IND
- Peer test countries: BGD, IDN, MEX, TUR, VNM

## Product metadata

- Products: 1,243
- Missing names: 1
- Missing sectors: 1
- Special code XXXX remains unmapped; it is not assigned an invented sector.

## Label counts

| As-of year | Target year | Split | Rows | Positive labels |
|---:|---:|---|---:|---:|
| 2000 | 2005 | test_india | 392 | 12 |
| 2000 | 2005 | test_peer | 3,312 | 56 |
| 2000 | 2005 | train | 131,242 | 2,205 |
| 2005 | 2010 | test_india | 279 | 5 |
| 2005 | 2010 | test_peer | 2,805 | 39 |
| 2005 | 2010 | train | 117,713 | 2,205 |
| 2010 | 2015 | test_india | 237 | 4 |
| 2010 | 2015 | test_peer | 2,477 | 17 |
| 2010 | 2015 | train | 106,471 | 1,136 |
| 2015 | 2020 | test_india | 207 | 1 |
| 2015 | 2020 | test_peer | 2,284 | 14 |
| 2015 | 2020 | train | 99,437 | 850 |
| 2024 | Forecast | forecast | 92,789 | Not observed |

## India positive labels by historical window

| As-of year | Target year | Eligible India pairs | Positive labels |
|---:|---:|---:|---:|
| 2000 | 2005 | 392 | 12 |
| 2005 | 2010 | 279 | 5 |
| 2010 | 2015 | 237 | 4 |
| 2015 | 2020 | 207 | 1 |

## Interpretation cautions

- Labels represent transitions in the calculated RCA3 measure, not causal evidence.
- Historical eligibility requires RCA3 to remain below the threshold in every year since the first data year.
- Forecast rows have no observed target outcomes and must not be treated as labeled training examples.
- A small number of India positives limits the reliability of India-only evaluation metrics.
- The Atlas source `export_rca` differs from the independently calculated standard Balassa RCA for some records; the cause remains unresolved. For Mali (MLI), HS4 product 5203, in 2022, source RCA was 15,895.525391 while calculated RCA was 2,308.718728. The calculated result agrees with the RCA implied by the source `global_market_share` (approximately 2,308.717252). Keep the calculated RCA as the project metric and retain the source value for audit; do not claim the cause is established.
- Merchandise exports and RCA are not direct measures of GNP or domestic value added.
