"""Model A — sequence-to-sequence LSTM demand forecaster (PyTorch).

    static: SKU, warehouse and category embeddings + log demand scale
        │                 └──> initial hidden/cell state of the encoder
    past 56 days ──> LSTM encoder ──(state)──> LSTM decoder <── known future 28 days
                                                   │            (promotions, price, calendar)
                                                   v
                                     quantile head -> P10, P50, P90 per day

Quantiles are produced as P50 ± softplus offsets, so P10 ≤ P50 ≤ P90 always holds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LSTMConfig:
    n_past: int = 10
    n_future: int = 9
    n_sku: int = 100
    n_warehouse: int = 12
    n_category: int = 10
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.1
    emb_sku: int = 8
    emb_warehouse: int = 4
    emb_category: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


def monotone_quantiles(raw: torch.Tensor) -> torch.Tensor:
    """(…, 3) unconstrained -> (…, 3) ordered P10 ≤ P50 ≤ P90."""
    mid = raw[..., 1]
    lo = mid - F.softplus(raw[..., 0])
    hi = mid + F.softplus(raw[..., 2])
    return torch.stack([lo, mid, hi], dim=-1)


class LSTMForecaster(nn.Module):
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.emb_sku = nn.Embedding(cfg.n_sku, cfg.emb_sku)
        self.emb_wh = nn.Embedding(cfg.n_warehouse, cfg.emb_warehouse)
        self.emb_cat = nn.Embedding(cfg.n_category, cfg.emb_category)
        static_dim = cfg.emb_sku + cfg.emb_warehouse + cfg.emb_category + 1
        self.init_h = nn.Linear(static_dim, cfg.hidden * cfg.layers)
        self.init_c = nn.Linear(static_dim, cfg.hidden * cfg.layers)
        drop = cfg.dropout if cfg.layers > 1 else 0.0
        self.encoder = nn.LSTM(cfg.n_past + static_dim, cfg.hidden, cfg.layers, batch_first=True, dropout=drop)
        self.decoder = nn.LSTM(cfg.n_future + static_dim, cfg.hidden, cfg.layers, batch_first=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.hidden, 3))

    def forward(self, past, future, codes, log_scale):
        static = torch.cat([self.emb_sku(codes[:, 0]), self.emb_wh(codes[:, 1]), self.emb_cat(codes[:, 2]),
                            log_scale], dim=-1)
        B, L = past.shape[0], self.cfg.layers
        h0 = torch.tanh(self.init_h(static)).view(B, L, -1).transpose(0, 1).contiguous()
        c0 = torch.tanh(self.init_c(static)).view(B, L, -1).transpose(0, 1).contiguous()
        enc_in = torch.cat([past, static.unsqueeze(1).expand(-1, past.shape[1], -1)], dim=-1)
        _, state = self.encoder(enc_in, (h0, c0))
        dec_in = torch.cat([future, static.unsqueeze(1).expand(-1, future.shape[1], -1)], dim=-1)
        out, _ = self.decoder(dec_in, state)
        return monotone_quantiles(self.head(out))  # (B, H, 3) in scaled units
