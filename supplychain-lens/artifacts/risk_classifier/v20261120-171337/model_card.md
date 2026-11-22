# Model card — Supplier disruption-risk classifier

**Version:** `v20261120-171337` · **Selected:** Neural network (PyTorch MLP) (best validation PR-AUC)
**Task:** probability that a disruption affecting a supplier starts within the next 14 days.
**Labels:** synthetic, from the generator's event log — this is synthetic supervised learning, not validated on real outcomes.
**Class balance:** 2.7% positive in training, 3.6% in test.

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test Brier | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Neural network (PyTorch MLP) | 0.179 | 0.058 | 0.713 | 0.0358 | 0.04 | 0.04 | 0.04 |
| Logistic regression | 0.151 | 0.065 | 0.747 | 0.0351 | 0.03 | 0.09 | 0.04 |
| XGBoost | 0.117 | 0.077 | 0.719 | 0.0349 | 0.04 | 0.15 | 0.06 |

Precision / recall / F1 use each model's threshold chosen on validation (best F1). PR-AUC is the headline
metric because positives are rare; a random model's PR-AUC equals the positive rate (0.036).

## What drives the selected model (permutation importance on test, associational)
- health_score: PR-AUC drop +0.017
- country_risk_base: PR-AUC drop +0.012
- region_active_events: PR-AUC drop +0.004
- country_active_events: PR-AUC drop +0.003
- delay_mean_90: PR-AUC drop +0.001
- days_since_own_event: PR-AUC drop +0.001

## Limitations
- Weather events are random in the generator and cannot be predicted from supplier data; part of the
  positive class is therefore unpredictable by design.
- Feature importance shows association with predictions, not causes of disruptions.
