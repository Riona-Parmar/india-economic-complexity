# Recovery Label Report

## Definition

- Previously specialized: RCA3 >= configured recovery threshold in at least one year before the as-of year.
- Currently lost advantage: RCA3 < configured recovery threshold at the as-of year.
- Recovered: RCA3 >= configured recovery threshold at the target year.
- Recovery threshold: 1.0
- Prediction horizon: 5 years
- This baseline does not require the loss to persist for two consecutive years.

## Counts by window and split

| As-of | Target | Split | Eligible | Recovered |
|---:|---:|---|---:|---:|
| 2000 | 2005 | test_india | 59 | 21 |
| 2000 | 2005 | test_peer | 301 | 56 |
| 2000 | 2005 | train | 9,309 | 1,479 |
| 2005 | 2010 | test_india | 118 | 17 |
| 2005 | 2010 | test_peer | 547 | 98 |
| 2005 | 2010 | train | 17,734 | 2,603 |
| 2010 | 2015 | test_india | 217 | 46 |
| 2010 | 2015 | test_peer | 787 | 101 |
| 2010 | 2015 | train | 24,009 | 2,740 |
| 2015 | 2020 | test_india | 246 | 49 |
| 2015 | 2020 | test_peer | 990 | 111 |
| 2015 | 2020 | train | 30,430 | 2,970 |

## India recovery outcomes

| As-of | Target | Eligible | Recovered | Rate (%) |
|---:|---:|---:|---:|---:|
| 2000 | 2005 | 59 | 21 | 35.6 |
| 2005 | 2010 | 118 | 17 | 14.4 |
| 2010 | 2015 | 217 | 46 | 21.2 |
| 2015 | 2020 | 246 | 49 | 19.9 |

## Limitations

- These labels describe observed RCA3 transitions; they do not establish causality.
- Features must be restricted to information available by the as-of year.
- Recovery labels must be modeled separately from new-emergence labels.
- A sensitivity analysis should test a two-consecutive-year loss definition.
