# Architecture follows Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon
# Time Series Forecasting" (2021). NEXUS-SCCT planned a TFT via pytorch-forecasting (tft_model.py);
# this is an independent, compact implementation written for SupplyChain Lens.
"""Model B — compact Temporal Fusion Transformer (PyTorch, implemented here layer by layer).

Building blocks
  GLU / GateAddNorm     gated skip connections: the network can switch a block off
  GRN                   gated residual network, optionally conditioned on a static context
  VariableSelection     softmax weights over input variables (interpretable importance)
  Static encoders       four context vectors from SKU / warehouse / category / scale:
                        c_select (variable selection), c_enrich (static enrichment),
                        c_h, c_c (initial LSTM state)
  LSTM encoder-decoder  local sequence processing (past -> future)
  Interpretable MHA     multi-head attention with a SHARED value projection, averaged over
                        heads, so one attention matrix can be read as "which days mattered"
  Quantile head         P10 / P50 / P90 (monotone)

`forward(..., return_weights=True)` also returns the variable-selection and attention weights.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from supplychain_lens.models.lstm import monotone_quantiles


@dataclass(frozen=True)
class TFTConfig:
    n_past: int = 10
    n_future: int = 9
    n_sku: int = 100
    n_warehouse: int = 12
    n_category: int = 10
    d_model: int = 32
    heads: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class GLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, 2 * d_out)

    def forward(self, x):
        a, b = self.lin(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GateAddNorm(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x, skip):
        return self.norm(self.glu(self.drop(x)) + skip)


class GRN(nn.Module):
    """Gated residual network: LayerNorm(skip(x) + GLU(W2 · ELU(W1 x + W3 c)))."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float, d_context: int | None = None):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ctx = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.gate = GateAddNorm(d_hidden, d_out, dropout)

    def forward(self, x, context=None):
        h = self.fc1(x)
        if self.ctx is not None and context is not None:
            h = h + self.ctx(context)
        h = self.fc2(F.elu(h))
        return self.gate(h, self.skip(x))


class VariableSelection(nn.Module):
    """Embeds each scalar variable, then mixes them with softmax weights (conditioned on context)."""

    def __init__(self, n_vars: int, d: int, dropout: float, d_context: int | None = None):
        super().__init__()
        self.n_vars = n_vars
        self.embed = nn.ModuleList([nn.Linear(1, d) for _ in range(n_vars)])
        self.var_grns = nn.ModuleList([GRN(d, d, d, dropout) for _ in range(n_vars)])
        self.weight_grn = GRN(n_vars * d, d, n_vars, dropout, d_context)

    def forward(self, x, context=None):  # x (B, T, n)
        emb = [lin(x[..., i : i + 1]) for i, lin in enumerate(self.embed)]       # n × (B, T, d)
        ctx = context.unsqueeze(1).expand(-1, x.shape[1], -1) if context is not None else None
        weights = torch.softmax(self.weight_grn(torch.cat(emb, dim=-1), ctx), dim=-1)  # (B, T, n)
        processed = torch.stack([g(e) for g, e in zip(self.var_grns, emb, strict=True)], dim=-1)  # (B, T, d, n)
        return (processed * weights.unsqueeze(-2)).sum(-1), weights


class InterpretableMultiHeadAttention(nn.Module):
    """Per-head queries/keys, one shared value projection, heads averaged (TFT paper, eq. 14–16)."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.d_head = heads, d // heads
        self.q = nn.ModuleList([nn.Linear(d, self.d_head) for _ in range(heads)])
        self.k = nn.ModuleList([nn.Linear(d, self.d_head) for _ in range(heads)])
        self.v = nn.Linear(d, self.d_head)
        self.out = nn.Linear(self.d_head, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key, mask):
        v = self.v(key)
        outs, attns = [], []
        for qh, kh in zip(self.q, self.k, strict=True):
            scores = qh(query) @ kh(key).transpose(-2, -1) / math.sqrt(self.d_head)
            attn = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
            attns.append(attn)
            outs.append(self.drop(attn) @ v)
        return self.out(torch.stack(outs).mean(0)), torch.stack(attns).mean(0)


class TemporalFusionTransformer(nn.Module):
    def __init__(self, cfg: TFTConfig):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.d_model, cfg.dropout
        self.emb_sku = nn.Embedding(cfg.n_sku, d)
        self.emb_wh = nn.Embedding(cfg.n_warehouse, d)
        self.emb_cat = nn.Embedding(cfg.n_category, d)
        self.emb_scale = nn.Linear(1, d)
        self.static_weights = GRN(4 * d, d, 4, p)
        self.static_grns = nn.ModuleList([GRN(d, d, d, p) for _ in range(4)])
        self.c_select, self.c_enrich, self.c_h, self.c_c = (GRN(d, d, d, p) for _ in range(4))
        self.vsn_past = VariableSelection(cfg.n_past, d, p, d_context=d)
        self.vsn_future = VariableSelection(cfg.n_future, d, p, d_context=d)
        self.encoder = nn.LSTM(d, d, batch_first=True)
        self.decoder = nn.LSTM(d, d, batch_first=True)
        self.post_lstm = GateAddNorm(d, d, p)
        self.enrich = GRN(d, d, d, p, d_context=d)
        self.attention = InterpretableMultiHeadAttention(d, cfg.heads, p)
        self.post_attn = GateAddNorm(d, d, p)
        self.position_ff = GRN(d, d, d, p)
        self.pre_output = GateAddNorm(d, d, p)
        self.head = nn.Linear(d, 3)

    def static_context(self, codes, log_scale):
        vars_ = [self.emb_sku(codes[:, 0]), self.emb_wh(codes[:, 1]), self.emb_cat(codes[:, 2]),
                 self.emb_scale(log_scale)]
        w = torch.softmax(self.static_weights(torch.cat(vars_, dim=-1)), dim=-1)          # (B, 4)
        processed = torch.stack([g(v) for g, v in zip(self.static_grns, vars_, strict=True)], dim=-1)
        static = (processed * w.unsqueeze(1)).sum(-1)
        return static, w

    def forward(self, past, future, codes, log_scale, return_weights: bool = False):
        static, w_static = self.static_context(codes, log_scale)
        c_sel, c_enr = self.c_select(static), self.c_enrich(static)
        h0, c0 = self.c_h(static).unsqueeze(0), self.c_c(static).unsqueeze(0)

        x_past, w_past = self.vsn_past(past, c_sel)
        x_fut, w_fut = self.vsn_future(future, c_sel)
        enc, state = self.encoder(x_past, (h0, c0))
        dec, _ = self.decoder(x_fut, state)
        lstm_out = torch.cat([enc, dec], dim=1)
        x_all = torch.cat([x_past, x_fut], dim=1)
        temporal = self.post_lstm(lstm_out, x_all)

        enriched = self.enrich(temporal, c_enr.unsqueeze(1).expand(-1, temporal.shape[1], -1))
        L, T = past.shape[1], temporal.shape[1]
        # causal mask: each future step may attend to all past steps and earlier future steps
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=past.device), diagonal=1)[L:]
        attn_out, attn = self.attention(enriched[:, L:], enriched, mask)
        x = self.post_attn(attn_out, enriched[:, L:])
        x = self.position_ff(x)
        x = self.pre_output(x, temporal[:, L:])
        q = monotone_quantiles(self.head(x))
        if return_weights:
            return q, {"static": w_static, "past": w_past, "future": w_fut, "attention": attn}
        return q
