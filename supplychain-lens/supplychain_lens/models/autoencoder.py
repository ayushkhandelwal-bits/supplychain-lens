"""Model C — dense autoencoder for operational anomaly detection (PyTorch).

Input: a 7-day window of 8 standardised daily features (56 numbers) for one SKU × warehouse.
The bottleneck (8 numbers) forces the network to learn what *normal* weeks look like.
Anomaly score = mean squared reconstruction error; per-feature errors explain which
signal looked unusual (associational, not causal).
"""

from __future__ import annotations

import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, n_inputs: int, hidden: int = 64, bottleneck: int = 8, dropout: float = 0.05):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, bottleneck), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, n_inputs))

    def forward(self, x):
        return self.decoder(self.encoder(x))
