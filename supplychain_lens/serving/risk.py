"""Inference for the supplier disruption-risk classifier (no PyTorch needed)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from supplychain_lens.serving import registry


@dataclass
class RiskScorer:
    selected: str
    version: str
    threshold: float
    mean: np.ndarray
    scale: np.ndarray
    run: registry.Run
    _session: object = None

    @classmethod
    def load(cls) -> RiskScorer:
        run = registry.load_run("risk_classifier")
        if run is None:
            raise FileNotFoundError("risk_classifier artifact not found")
        cfg, metrics, sc = run.json("config.json"), run.json("metrics.json"), run.json("scaler.json")
        sel = cfg["selected_model"]
        return cls(sel, run.version, metrics["models"][sel]["test"]["threshold"], np.array(sc["mean"]),
                   np.array(sc["scale"]), run)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Probabilities for raw feature rows (n, F) using the selected model."""
        Z = ((X - self.mean) / self.scale).astype(np.float32)
        if self.selected.startswith("Neural"):
            if self._session is None:
                import onnxruntime as ort

                self._session = ort.InferenceSession(str(self.run.file("model.onnx")), providers=["CPUExecutionProvider"])
            return self._session.run(None, {"features": Z})[0].reshape(-1)
        if self.selected == "Logistic regression":
            lr = self.run.json("logistic_regression.json")
            return 1 / (1 + np.exp(-(Z @ np.array(lr["coef"]) + lr["intercept"])))
        from xgboost import XGBClassifier  # training environment only

        m = XGBClassifier()
        m.load_model(str(self.run.file("xgboost.json")))
        return m.predict_proba(X)[:, 1]

    def current_scores(self) -> pd.DataFrame:
        return pd.read_parquet(self.run.file("current_scores.parquet"))
