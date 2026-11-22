# Model card — Autoencoder anomaly detector

**Version:** `v20261120-171353` · **Task:** flag unusual SKU × warehouse weeks for investigation
**Data:** synthetic (nexus_v2); 7-day windows of 8 daily features (demand_z, is_promo, is_holiday, unexplained_inv, receipt_delay, stockout, days_of_supply, pipeline_cover)

## Method
- Dense autoencoder 56 → 64 → 8 → 64 → 56, trained without labels on windows ending ≤ train end (8.6 s on CPU)
- Score = mean squared reconstruction error of the behavioural features (demand_z, unexplained_inv, receipt_delay) on the
  window's last day; promotions, holidays, stock cover and pipeline are context inputs only. Threshold 3.823 chosen on the validation quarter
  (best F1 against injected anomalies within an alert budget of 0.5% of series-days)

## Test quarter (Q4 2025)
| Detector | Alerts | Precision (injected) | Precision (incl. recorded events) | Event recall | Day-level FPR |
|---|---|---|---|---|---|
| Autoencoder | 217 | 0.05 | 0.50 | 0.38 (6/16) | 0.0056 |
| Robust z-score | 23 | 0.22 | 0.74 | 0.31 (5/16) | 0.0005 |

## Head-to-head at the same alert volume (top 100 test days each)
| Detector | Precision (injected) | Precision (incl. recorded events) | Event recall |
|---|---|---|---|
| Autoencoder | 0.10 | 0.65 | 0.38 |
| Robust z-score | 0.05 | 0.68 | 0.31 |

Only 16 injected anomalies start in the test quarter, so these rates are uncertain (one event = 6% of recall).

## Limitations
- Evaluated against injected synthetic anomalies; real anomalies are more varied.
- An alert means "investigation required", not a confirmed disruption.
- Per-feature reconstruction errors show which inputs looked unusual; they are not causes.
