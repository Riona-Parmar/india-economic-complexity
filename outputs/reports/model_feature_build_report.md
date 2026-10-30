# Model Feature Build Report

- Output rows: 551,603
- Output columns: 32
- Historical windows: [2000, 2005, 2010, 2015]
- Prediction horizon: 5 years
- India: IND
- Peers: BGD, IDN, MEX, TUR, VNM

## Label counts

      pathway  asof_year      split   rows  positives  negatives
new_emergence       2000 test_india    392         12        380
new_emergence       2000  test_peer   3312         56       3256
new_emergence       2000      train 131242       2205     129037
new_emergence       2005 test_india    279          5        274
new_emergence       2005  test_peer   2805         39       2766
new_emergence       2005      train 117713       2205     115508
new_emergence       2010 test_india    237          4        233
new_emergence       2010  test_peer   2477         17       2460
new_emergence       2010      train 106471       1136     105335
new_emergence       2015 test_india    207          1        206
new_emergence       2015  test_peer   2284         14       2270
new_emergence       2015      train  99437        850      98587
     recovery       2000 test_india     59         21         38
     recovery       2000  test_peer    301         56        245
     recovery       2000      train   9309       1479       7830
     recovery       2005 test_india    118         17        101
     recovery       2005  test_peer    547         98        449
     recovery       2005      train  17734       2603      15131
     recovery       2010 test_india    217         46        171
     recovery       2010  test_peer    787        101        686
     recovery       2010      train  24009       2740      21269
     recovery       2015 test_india    246         49        197
     recovery       2015  test_peer    990        111        879
     recovery       2015      train  30430       2970      27460

## Feature columns

export_value_current, log_export_value, rca_current, rca3_current, m_current, pci_current, eci_current, atlas_distance_current, atlas_cog_current, export_value_lag1, rca_lag1, rca3_lag1, m_lag1, rca_lag3, rca3_lag3, m_lag3, log_export_change_1y, log_export_change_3y, rca3_change_3y, lag1_available, lag3_available

## Leakage safeguards

- Each label is joined to features for its as-of year only.
- Lag features use the as-of year or earlier.
- Target-year RCA, exports and other target-year variables are not model features.
- Product names and sector classifications are static metadata.
- Labels from new emergence and recovery remain separated by the pathway column.

## Important limitations

- Lag values unavailable at the beginning of the dataset are set to zero and accompanied by availability indicators.
- The current ECI variable is a project proxy, not an official ECI series.
- Atlas source RCA and calculated Balassa RCA have an unresolved discrepancy.
- The product mapping covers 20 merchandise sectors; software and other commercial services need a separate dataset.
- This build prepares features but does not train or evaluate models.
