# Feature Readiness Audit

## Configuration

- Data period: 1995–2024
- Historical as-of years: [2000, 2005, 2010, 2015]
- Prediction horizon: 5 years
- India evaluation country: IND
- Peer evaluation countries: BGD, IDN, MEX, TUR, VNM

## Product metadata

- Product rows: 1,243
- Missing names: 1
- Missing sectors: 1
- Distinct goods sectors represented: 20
- Software and other commercial services are not covered by this merchandise panel.

## Label balance

| pathway       |   asof_year |   target_year | split      |   rows |   positives |   negatives |   positive_rate_pct |
|:--------------|------------:|--------------:|:-----------|-------:|------------:|------------:|--------------------:|
| new_emergence |        2000 |          2005 | test_india |    392 |          12 |         380 |            3.06122  |
| new_emergence |        2000 |          2005 | test_peer  |   3312 |          56 |        3256 |            1.69082  |
| new_emergence |        2000 |          2005 | train      | 131242 |        2205 |      129037 |            1.6801   |
| new_emergence |        2005 |          2010 | test_india |    279 |           5 |         274 |            1.79211  |
| new_emergence |        2005 |          2010 | test_peer  |   2805 |          39 |        2766 |            1.39037  |
| new_emergence |        2005 |          2010 | train      | 117713 |        2205 |      115508 |            1.8732   |
| new_emergence |        2010 |          2015 | test_india |    237 |           4 |         233 |            1.68776  |
| new_emergence |        2010 |          2015 | test_peer  |   2477 |          17 |        2460 |            0.686314 |
| new_emergence |        2010 |          2015 | train      | 106471 |        1136 |      105335 |            1.06696  |
| new_emergence |        2015 |          2020 | test_india |    207 |           1 |         206 |            0.483092 |
| new_emergence |        2015 |          2020 | test_peer  |   2284 |          14 |        2270 |            0.61296  |
| new_emergence |        2015 |          2020 | train      |  99437 |         850 |       98587 |            0.854813 |
| recovery      |        2000 |          2005 | test_india |     59 |          21 |          38 |           35.5932   |
| recovery      |        2000 |          2005 | test_peer  |    301 |          56 |         245 |           18.6047   |
| recovery      |        2000 |          2005 | train      |   9309 |        1479 |        7830 |           15.8879   |
| recovery      |        2005 |          2010 | test_india |    118 |          17 |         101 |           14.4068   |
| recovery      |        2005 |          2010 | test_peer  |    547 |          98 |         449 |           17.9159   |
| recovery      |        2005 |          2010 | train      |  17734 |        2603 |       15131 |           14.678    |
| recovery      |        2010 |          2015 | test_india |    217 |          46 |         171 |           21.1982   |
| recovery      |        2010 |          2015 | test_peer  |    787 |         101 |         686 |           12.8335   |
| recovery      |        2010 |          2015 | train      |  24009 |        2740 |       21269 |           11.4124   |
| recovery      |        2015 |          2020 | test_india |    246 |          49 |         197 |           19.9187   |
| recovery      |        2015 |          2020 | test_peer  |    990 |         111 |         879 |           11.2121   |
| recovery      |        2015 |          2020 | train      |  30430 |        2970 |       27460 |            9.76011  |

## As-of feature coverage

| pathway       |   asof_year | split      |   label_rows |   feature_rows_found |   feature_rows_missing |   export_value_missing |   export_value_missing_pct |   rca_missing |   rca_missing_pct |   rca3_missing |   rca3_missing_pct |   m_missing |   m_missing_pct |   pci_missing |   pci_missing_pct |   eci_missing |   eci_missing_pct |   atlas_distance_missing |   atlas_distance_missing_pct |   atlas_cog_missing |   atlas_cog_missing_pct |
|:--------------|------------:|:-----------|-------------:|---------------------:|-----------------------:|-----------------------:|---------------------------:|--------------:|------------------:|---------------:|-------------------:|------------:|----------------:|--------------:|------------------:|--------------:|------------------:|-------------------------:|-----------------------------:|--------------------:|------------------------:|
| new_emergence |        2000 | test_india |          392 |                  392 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| new_emergence |        2000 | test_peer  |         3312 |                 3312 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| new_emergence |        2000 | train      |       131242 |               131242 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| new_emergence |        2005 | test_india |          279 |                  279 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| new_emergence |        2005 | test_peer  |         2805 |                 2805 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| new_emergence |        2005 | train      |       117713 |               117713 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| new_emergence |        2010 | test_india |          237 |                  237 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| new_emergence |        2010 | test_peer  |         2477 |                 2477 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| new_emergence |        2010 | train      |       106471 |               106471 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| new_emergence |        2015 | test_india |          207 |                  207 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| new_emergence |        2015 | test_peer  |         2284 |                 2284 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| new_emergence |        2015 | train      |        99437 |                99437 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| recovery      |        2000 | test_india |           59 |                   59 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| recovery      |        2000 | test_peer  |          301 |                  301 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| recovery      |        2000 | train      |         9309 |                 9309 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| recovery      |        2005 | test_india |          118 |                  118 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| recovery      |        2005 | test_peer  |          547 |                  547 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| recovery      |        2005 | train      |        17734 |                17734 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| recovery      |        2010 | test_india |          217 |                  217 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| recovery      |        2010 | test_peer  |          787 |                  787 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| recovery      |        2010 | train      |        24009 |                24009 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| recovery      |        2015 | test_india |          246 |                  246 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| recovery      |        2015 | test_peer  |          990 |                  990 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |
| recovery      |        2015 | train      |        30430 |                30430 |                      0 |                      0 |                          0 |             0 |                 0 |              0 |                  0 |           0 |               0 |             0 |                 0 |             0 |                 0 |                        0 |                            0 |                   0 |                       0 |

## Overall feature missingness

| feature        |   missing_values |   total_label_rows |   missing_pct |
|:---------------|-----------------:|-------------------:|--------------:|
| export_value   |                0 |             551603 |             0 |
| rca            |                0 |             551603 |             0 |
| rca3           |                0 |             551603 |             0 |
| m              |                0 |             551603 |             0 |
| pci            |                0 |             551603 |             0 |
| eci            |                0 |             551603 |             0 |
| atlas_distance |                0 |             551603 |             0 |
| atlas_cog      |                0 |             551603 |             0 |

## Validation checks

- India split violations: 0
- Peer split violations: 0
- Training split violations: 0
- Incorrect prediction horizon rows: 0
- Labels without matching as-of panel row: 0

## Interpretation and safeguards

- Features are joined using country, product, and as-of year.
- Target-year values are not joined as predictive features.
- Missing feature values must be handled using training-only imputation rules.
- The label eligibility rules differ between new emergence and recovery; compare metrics within each pathway.
- India-only metrics may be unstable because of small positive counts in some windows.
- ECI is a proxy based on the current panel construction, not an official national ECI series.
- Atlas source RCA and calculated Balassa RCA have a known unresolved discrepancy.
- This is a readiness audit, not evidence of predictive performance or causal effects.
