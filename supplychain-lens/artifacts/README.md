# Model artifacts

Versioned outputs of training runs, one folder per model and version:
`artifacts/<model>/<version>/` with `model.onnx`, `model.pt`, `config.json`, `scaler.json`,
`metrics.json`, `loss_curve.csv`, `test_predictions.parquet`, `model_card.md`, `SHA256SUMS`.
See docs/adr/0009-training-vs-inference-environments.md. Empty until Phase 5.
