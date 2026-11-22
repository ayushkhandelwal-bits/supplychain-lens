# Model card — LSTM (PyTorch)

**Version:** `v20261120-170453` · **Task:** 28-day daily demand forecast per SKU × warehouse with P10/P50/P90
**Data:** synthetic (NEXUS-SCCT data model, nexus_v2), 400 series × 1096 days

## Training
- Split: train ≤ 2025-06-30 · validation ≤ 2025-09-30 · test 2025-10-01 → 2025-12-31
- Inputs: past 56 days of demand_scaled, is_promo, price_ratio, dow_sin, dow_cos, doy_sin, doy_cos, is_black_friday, is_cyber_monday, is_holiday_week; known future is_promo, price_ratio, dow_sin, dow_cos, doy_sin, doy_cos, is_black_friday, is_cyber_monday, is_holiday_week;
  static SKU / warehouse / category embeddings and log demand scale
- Loss: pinball (quantile) loss; scaling by train-period mean per series
- Parameters: 54,257 · epochs run: 19 (best 14) · training time 237.1 s on CPU
- Hyperparameters: `{'lr': 0.002, 'batch_size': 256, 'windows_per_epoch': 40000, 'max_epochs': 30, 'patience': 5}`

## Results (demand units)
| Split | WAPE | MAE | RMSE | Bias | P10–P90 coverage |
|---|---|---|---|---|---|
| Validation | 0.141 | 87.0 | 182.8 | -0.007 | 79.3% |
| Test (Q4 2025) | 0.136 | 98.4 | 221.3 | -0.033 | 76.2% |

Comparison with baselines and the other neural model: `reports/forecast_comparison.md`.

## Serving
- ONNX file 280.4 KB; max |ONNX − PyTorch| = 9.5e-07
- ONNX Runtime latency for all 400 series: 26.4 ms (CPU)

## Limitations
- Trained and evaluated on synthetic data only; results are not evidence of real-world accuracy.
- P10/P90 are forecast quantiles; their empirical coverage is reported above rather than assumed.
- No promotions are known after the snapshot date, so live forecasts assume none.
