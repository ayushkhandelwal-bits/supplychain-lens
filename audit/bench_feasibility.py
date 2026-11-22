# Scratch feasibility benchmark used for docs/change_impact_deep_learning.md §G.
# Sizing prototypes on stand-in data only; NOT the SupplyChain Lens models. Needs: torch, numpy, onnx, onnxruntime, statsmodels.
"""Feasibility benchmark (scratch, not project code).
Measures CPU training/inference cost of candidate SupplyChain Lens models at the proposed data scale.
"""
import json, time, math, os, sys, io
import numpy as np, torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(os.cpu_count())
R = {"cpu_threads": torch.get_num_threads(), "torch": torch.__version__}

# ---------- proposed scale: 100 SKUs x 4 warehouses (spread over 12) x 3 years daily ----------
N_SKU, WH_PER_SKU, N_WH, T = 100, 4, 12, 1095
S = N_SKU * WH_PER_SKU
t = np.arange(T)
dow = (t + 3) % 7; doy = t % 365
base = np.random.uniform(100, 2000, S)[:, None]
trend = 1 + np.random.uniform(-0.05, 0.15, S)[:, None] * t / 365
wk_amp = np.random.uniform(0.05, 0.35, S)[:, None]
yr_amp = np.random.uniform(0.0, 0.3, S)[:, None]
promo = (np.random.rand(S, T) < 0.04).astype(float)
price_ratio = 1 - 0.2 * promo
y = base * trend * (1 + wk_amp * np.cos(2 * np.pi * dow / 7)) * (1 + yr_amp * np.sin(2 * np.pi * doy / 365)) \
    * (1 + 0.6 * promo) * np.random.lognormal(0, 0.15, (S, T))
R["series"], R["days"], R["rows"] = S, T, S * T

sku_id = np.repeat(np.arange(N_SKU), WH_PER_SKU); wh_id = np.random.randint(0, N_WH, S)
known = np.stack([np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7), np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)], -1)  # (T,4)

L, H = 56, 28
test_start, val_start = T - 56, T - 112
scale = y[:, :val_start].mean(1, keepdims=True)  # fit on train only
ys = y / scale

def windows(end, stride):
    xs, idx = [], []
    for s in range(S):
        for e in range(L, end - H + 1, stride):
            idx.append((s, e))
    return idx

train_idx = windows(val_start, 1)
R["train_windows_stride1"] = len(train_idx)

def batch(idx):
    s = np.array([i[0] for i in idx]); e = np.array([i[1] for i in idx])
    past = np.stack([np.concatenate([ys[si, ei-L:ei, None], promo[si, ei-L:ei, None], price_ratio[si, ei-L:ei, None], known[ei-L:ei]], -1) for si, ei in zip(s, e)])
    fut = np.stack([np.concatenate([promo[si, ei:ei+H, None], price_ratio[si, ei:ei+H, None], known[ei:ei+H]], -1) for si, ei in zip(s, e)])
    tgt = np.stack([ys[si, ei:ei+H] for si, ei in zip(s, e)])
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    return f(past), f(fut), torch.tensor(sku_id[s]), torch.tensor(wh_id[s]), f(tgt)

QUANT = torch.tensor([0.1, 0.5, 0.9])
def pinball(pred, tgt):  # pred (B,H,3)
    d = tgt.unsqueeze(-1) - pred
    return torch.maximum(QUANT * d, (QUANT - 1) * d).mean()

class LSTMForecaster(nn.Module):
    def __init__(s, hid=64):
        super().__init__()
        s.e_sku, s.e_wh = nn.Embedding(N_SKU, 8), nn.Embedding(N_WH, 4)
        s.enc = nn.LSTM(7 + 12, hid, batch_first=True)
        s.dec = nn.LSTM(6, hid, batch_first=True)
        s.head = nn.Linear(hid, 3)
    def forward(s, past, fut, sk, wh):
        st = torch.cat([s.e_sku(sk), s.e_wh(wh)], -1)
        _, hc = s.enc(torch.cat([past, st.unsqueeze(1).expand(-1, past.shape[1], -1)], -1))
        out, _ = s.dec(fut, hc)
        return s.head(out)

class GRN(nn.Module):
    def __init__(s, d_in, d, d_out=None):
        super().__init__(); d_out = d_out or d
        s.l1, s.l2, s.g = nn.Linear(d_in, d), nn.Linear(d, d_out * 2), nn.LayerNorm(d_out)
        s.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
    def forward(s, x):
        h = nn.functional.glu(s.l2(nn.functional.elu(s.l1(x))))
        return s.g(h + s.skip(x))

class VSN(nn.Module):
    def __init__(s, n_vars, d):
        super().__init__()
        s.proj = nn.ModuleList([nn.Linear(1, d) for _ in range(n_vars)])
        s.grns = nn.ModuleList([GRN(d, d) for _ in range(n_vars)])
        s.w = GRN(n_vars * d, d, n_vars)
    def forward(s, x):  # x (B,T,n)
        e = [p(x[..., i:i+1]) for i, p in enumerate(s.proj)]
        w = torch.softmax(s.w(torch.cat(e, -1)), -1).unsqueeze(-2)
        v = torch.stack([g(ei) for g, ei in zip(s.grns, e)], -1)
        return (v * w).sum(-1)

class TFTProxy(nn.Module):
    """Compact TFT-style model: variable selection, GRNs, LSTM enc/dec, gated skip, multi-head attention, quantile head."""
    def __init__(s, d=32, heads=4):
        super().__init__()
        s.e_sku, s.e_wh = nn.Embedding(N_SKU, d), nn.Embedding(N_WH, d)
        s.static = GRN(2 * d, d)
        s.vsn_p, s.vsn_f = VSN(7, d), VSN(6, d)
        s.enc, s.dec = nn.LSTM(d, d, batch_first=True), nn.LSTM(d, d, batch_first=True)
        s.post = GRN(d, d); s.att = nn.MultiheadAttention(d, heads, batch_first=True)
        s.ff = GRN(d, d); s.head = nn.Linear(d, 3)
    def forward(s, past, fut, sk, wh):
        c = s.static(torch.cat([s.e_sku(sk), s.e_wh(wh)], -1))
        hp, hf = s.vsn_p(past) + c.unsqueeze(1), s.vsn_f(fut) + c.unsqueeze(1)
        oe, hc = s.enc(hp); od, _ = s.dec(hf, hc)
        seq = s.post(torch.cat([oe, od], 1))
        Tp = past.shape[1]; Tt = seq.shape[1]
        mask = torch.triu(torch.ones(Tt, Tt, dtype=torch.bool), 1)
        a, _ = s.att(seq[:, Tp:], seq, seq, attn_mask=mask[Tp:])
        return s.head(s.ff(a + seq[:, Tp:]))

def nparams(m): return sum(p.numel() for p in m.parameters())
def size_kb(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return round(len(b.getvalue()) / 1024, 1)

def bench(name, m, n_train=20000, bs=256):
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    rng = np.random.default_rng(0); sel = rng.choice(len(train_idx), n_train, replace=False)
    m.train(); t0 = time.perf_counter(); losses = []
    for i in range(0, n_train, bs):
        p, f, sk, wh, tg = batch([train_idx[j] for j in sel[i:i+bs]])
        loss = pinball(m(p, f, sk, wh), tg); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    dt = time.perf_counter() - t0
    m.eval(); p, f, sk, wh, _ = batch([(s, test_start) for s in range(S)])
    with torch.no_grad():
        m(p, f, sk, wh); t1 = time.perf_counter(); m(p, f, sk, wh); inf = time.perf_counter() - t1
    R[name] = {"params": nparams(m), "state_dict_kb": size_kb(m),
               "train_sec_per_20k_windows": round(dt, 1),
               "est_sec_per_epoch_stride1": round(dt * len(train_idx) / n_train, 0),
               "est_sec_per_epoch_stride7": round(dt * len(train_idx) / n_train / 7, 0),
               "inference_ms_all_400_series_28d": round(inf * 1000, 1),
               "loss_first_last": [round(losses[0], 3), round(losses[-1], 3)]}
    return m, (p, f, sk, wh)

lstm, ex = bench("lstm_h64", LSTMForecaster(64))
bench("tft_proxy_d32", TFTProxy(32))

# ONNX export + onnxruntime latency (deployment option without torch)
try:
    import onnxruntime as ort
    path = "lstm.onnx"
    torch.onnx.export(lstm, ex, path, input_names=["past", "fut", "sku", "wh"], output_names=["q"], dynamo=False)
    sess = ort.InferenceSession(path)
    feeds = {"past": ex[0].numpy(), "fut": ex[1].numpy(), "sku": ex[2].numpy(), "wh": ex[3].numpy()}
    sess.run(None, feeds); t1 = time.perf_counter(); out = sess.run(None, feeds); ms = (time.perf_counter() - t1) * 1000
    with torch.no_grad(): diff = float(np.abs(out[0] - lstm(*ex).numpy()).max())
    R["onnx_lstm"] = {"file_kb": round(os.path.getsize(path) / 1024, 1), "ort_inference_ms_400_series": round(ms, 1), "max_abs_diff_vs_torch": diff}
except Exception as e:
    R["onnx_lstm"] = f"failed: {type(e).__name__}: {e}"[:300]

# Autoencoder on daily operational feature vectors (e.g. 50 suppliers x 3 yrs x 12 features)
X = torch.randn(50 * T, 12)
ae = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 4), nn.ReLU(), nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 12))
opt = torch.optim.Adam(ae.parameters(), 1e-3); t0 = time.perf_counter()
for ep in range(5):
    for i in range(0, len(X), 256):
        xb = X[i:i+256]; l = ((ae(xb) - xb) ** 2).mean(); opt.zero_grad(); l.backward(); opt.step()
R["autoencoder"] = {"rows": len(X), "params": nparams(ae), "sec_per_epoch": round((time.perf_counter() - t0) / 5, 2)}

# Statistical baseline cost: Holt-Winters for all series
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    t0 = time.perf_counter()
    for s in range(40):
        ExponentialSmoothing(y[s, :val_start], trend="add", seasonal="mul", seasonal_periods=7).fit()
    R["ets_sec_all_400_series_est"] = round((time.perf_counter() - t0) * 10, 1)
except Exception as e:
    R["ets"] = str(e)[:200]

print(json.dumps(R, indent=2))
json.dump(R, open("bench_results.json", "w"), indent=2)
