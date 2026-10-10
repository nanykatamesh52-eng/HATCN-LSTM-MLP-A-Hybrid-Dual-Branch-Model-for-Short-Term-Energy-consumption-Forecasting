"""
HATCN-LSTM-MLP  --  reproducible pipeline (rebuilt from Manuscript Table 1 / Table 2)

Run on Colab / Kaggle (GPU).  Sections are separated by "# %%" so you can paste
them into notebook cells.

Inputs you must download yourself:
  1) PJME_hourly.csv  (Kaggle: robikscube/hourly-energy-consumption)  columns: Datetime, PJME_MW
  2) weather.csv      (NOAA hourly, one station in the PJM-East area)  columns: Datetime, temp, humidity
     -> if missing, the code runs with calendar features only (set USE_WEATHER=False).

Everything reported in the paper must come from ONE run of this script (fixed seeds),
including the abstract, tables, loss curves and figures.
"""
# %% 0. Config ---------------------------------------------------------------
import os, json, math, random, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
LOAD_CSV = "PJME_hourly.csv"
WEATHER_CSV = "weather.csv"
USE_WEATHER = os.path.exists(WEATHER_CSV)
OUT = "results"; os.makedirs(OUT, exist_ok=True)

LOOKBACK, HORIZON = 168, 24
QUANTILES = (0.1, 0.5, 0.9)
EPOCHS, BATCH, LR, PATIENCE = 60, 64, 1e-3, 8
SEEDS = [0, 1, 2]                      # report mean +- std over seeds
USE_EXOG = True                        # also run once with False (ablation) for fairness
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# %% 1. Data -----------------------------------------------------------------
def load_data():
    df = pd.read_csv(LOAD_CSV, parse_dates=["Datetime"]).drop_duplicates("Datetime")
    df = df.set_index("Datetime").sort_index()
    df = df.rename(columns={df.columns[0]: "load"})
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    df["load"] = df["load"].interpolate(limit=6)          # fill short gaps only
    if USE_WEATHER:
        w = pd.read_csv(WEATHER_CSV, parse_dates=["Datetime"]).sort_values("Datetime")
        df = pd.merge_asof(df.reset_index().rename(columns={"index": "Datetime"}), w,
                           on="Datetime", tolerance=pd.Timedelta("1h"), direction="nearest").set_index("Datetime")
        df[["temp", "humidity"]] = df[["temp", "humidity"]].interpolate(limit=6)
    df = df.dropna()
    return df


def exog_features(idx, df):
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hol = USFederalHolidayCalendar().holidays(idx.min(), idx.max())
    f = pd.DataFrame(index=idx)
    f["h_sin"] = np.sin(2 * np.pi * idx.hour / 24);   f["h_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["d_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7); f["d_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    f["m_sin"] = np.sin(2 * np.pi * idx.month / 12);  f["m_cos"] = np.cos(2 * np.pi * idx.month / 12)
    f["weekend"] = (idx.dayofweek >= 5).astype(float)
    f["holiday"] = idx.normalize().isin(hol).astype(float)
    if USE_WEATHER:
        f["temp"] = df["temp"].values; f["humidity"] = df["humidity"].values
    return f


def make_windows(load, ex, start, end):
    """Windows whose *forecast targets* lie in [start, end).  load: (T,), ex: (T,F)."""
    X, E, Y, T0 = [], [], [], []
    for t in range(max(start, LOOKBACK), end - HORIZON + 1):
        X.append(load[t - LOOKBACK:t]); Y.append(load[t:t + HORIZON])
        E.append(ex[t:t + HORIZON].reshape(-1)); T0.append(t)   # calendar (and weather *forecast*) of target hours
    return (np.asarray(X, np.float32)[..., None], np.asarray(E, np.float32),
            np.asarray(Y, np.float32), np.asarray(T0))


def prepare():
    df = load_data()
    n = len(df); i1, i2 = int(.70 * n), int(.85 * n)                  # chronological 70/15/15
    lo, hi = df["load"].iloc[:i1].min(), df["load"].iloc[:i1].max()      # scaler fitted on TRAIN only (no leakage)
    load = ((df["load"].values - lo) / (hi - lo)).astype(np.float32)
    ex = exog_features(df.index, df)
    ex_tr = ex.iloc[:i1]; mn, mx = ex_tr.min(), ex_tr.max()
    ex = ((ex - mn) / (mx - mn + 1e-9)).values.astype(np.float32)
    sets = {"train": make_windows(load, ex, 0, i1), "val": make_windows(load, ex, i1, i2),
            "test": make_windows(load, ex, i2, n)}
    meta = dict(lo=float(lo), hi=float(hi), n=n, idx=df.index, i1=i1, i2=i2, ex_dim=ex.shape[1] * HORIZON)
    return sets, meta, load, ex


# %% 2. Models ---------------------------------------------------------------
class TCNBlock(nn.Module):
    def __init__(self, cin, cout, k, d, p=0.2):
        super().__init__()
        self.pad = (k - 1) * d                                           # left padding only => causal
        self.conv = nn.Conv1d(cin, cout, k, dilation=d)
        self.drop = nn.Dropout(p)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):                                                # x: (B,C,T)
        y = self.drop(F.relu(self.conv(F.pad(x, (self.pad, 0)))))
        return y + self.res(x)                                           # Eq. (3)


class TCN(nn.Module):
    def __init__(self, cin=1, ch=64, k=3, dilations=(1, 2, 4)):
        super().__init__()
        self.net = nn.Sequential(*[TCNBlock(cin if i == 0 else ch, ch, k, d) for i, d in enumerate(dilations)])

    def forward(self, x): return self.net(x)


class Attention(nn.Module):                                              # Eq. (10)-(11)
    def __init__(self, d):
        super().__init__(); self.W = nn.Linear(d, d); self.v = nn.Linear(d, 1, bias=False)

    def forward(self, h):                                                # h: (B,T,D)
        a = torch.softmax(self.v(torch.tanh(self.W(h))).squeeze(-1), dim=1)
        return (a.unsqueeze(-1) * h).sum(1), a


class QuantileHead(nn.Module):
    """Predicts median + two POSITIVE offsets  =>  lower <= median <= upper by construction
    (answers Reviewer 1, comment 3: quantile crossing)."""
    def __init__(self, din, hidden=128, p=0.3):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, HORIZON * 3))

    def forward(self, z):
        o = self.fc(z).view(-1, HORIZON, 3)
        med = o[..., 0]
        return torch.stack([med - F.softplus(o[..., 1]), med, med + F.softplus(o[..., 2])], dim=-1)  # (B,24,3)


class Net(nn.Module):
    """kind: tcn | tcn_lstm | tcn_attn | cnn_lstm_attn | hatcn"""
    def __init__(self, kind, ex_dim, use_exog=True):
        super().__init__()
        self.kind, self.use_exog = kind, use_exog
        if kind == "cnn_lstm_attn":
            self.enc = nn.Sequential(nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        else:
            self.enc = TCN()
        if kind in ("tcn_lstm", "cnn_lstm_attn", "hatcn"):
            self.lstm = nn.LSTM(64, 128, batch_first=True); d = 128
        else:
            d = 64
        if kind in ("tcn_attn", "cnn_lstm_attn", "hatcn"):
            self.att = Attention(d)
        if use_exog:
            self.mlp = nn.Sequential(nn.Linear(ex_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())  # Eq. (12)
        self.head = QuantileHead(d + (32 if use_exog else 0))

    def forward(self, x, e):
        h = self.enc(x.transpose(1, 2)).transpose(1, 2)                  # (B,T,64)
        if hasattr(self, "lstm"): h, _ = self.lstm(h)
        ctx = self.att(h)[0] if hasattr(self, "att") else h[:, -1]
        if self.use_exog: ctx = torch.cat([ctx, self.mlp(e)], dim=1)     # Eq. (13)
        return self.head(ctx)


def pinball(pred, y, qs=QUANTILES):                                      # Eq. (14)
    q = torch.tensor(qs, device=pred.device).view(1, 1, -1)
    err = y.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1) * err).mean()


# %% 3. Training -------------------------------------------------------------
def loader(s, shuffle):
    X, E, Y, _ = s
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(E), torch.from_numpy(Y))
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


@torch.no_grad()
def predict(model, s):
    model.eval(); out = []
    for x, e, _ in loader(s, False):
        out.append(model(x.to(DEVICE), e.to(DEVICE)).cpu())
    return torch.cat(out).numpy()                                        # (N,24,3) scaled


def train(model, sets):
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=LR)
    hist, best, bad, state = [], 1e9, 0, None
    for ep in range(EPOCHS):
        model.train(); tr = []
        for x, e, y in loader(sets["train"], True):
            x, e, y = x.to(DEVICE), e.to(DEVICE), y.to(DEVICE)
            loss = pinball(model(x, e), y); opt.zero_grad(); loss.backward(); opt.step(); tr.append(loss.item())
        va = pinball(torch.from_numpy(predict(model, sets["val"])), torch.from_numpy(sets["val"][2])).item()
        hist.append((float(np.mean(tr)), va))                            # REAL per-epoch values -> use for Fig 4/5
        if va < best: best, bad, state = va, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(state)
    return hist


# %% 4. Metrics (original MW scale + normalised) ----------------------------
def metrics(pred, y, meta, alpha=0.2):
    """pred: (N,24,3) scaled; y: (N,24) scaled."""
    lo_, med, hi_ = pred[..., 0], pred[..., 1], pred[..., 2]
    rng = meta["hi"] - meta["lo"]
    inv = lambda a: a * rng + meta["lo"]
    Y, M, L, U = inv(y), inv(med), inv(lo_), inv(hi_)
    err = Y - M
    wink = (U - L) + (2 / alpha) * np.maximum(L - Y, 0) + (2 / alpha) * np.maximum(Y - U, 0)
    q = np.array(QUANTILES).reshape(1, 1, -1); e3 = y[..., None] - pred
    return {
        "RMSE_MW": float(np.sqrt((err ** 2).mean())), "MAE_MW": float(np.abs(err).mean()),
        "MAPE_%": float(100 * (np.abs(err) / Y).mean()),
        "RMSE_norm": float(np.sqrt(((y - med) ** 2).mean())), "MAE_norm": float(np.abs(y - med).mean()),
        "PICP_%": float(100 * ((Y >= L) & (Y <= U)).mean()), "MPIW_MW": float((U - L).mean()),
        "Winkler_MW": float(wink.mean()), "Pinball_norm": float(np.maximum(q * e3, (q - 1) * e3).mean()),
        "crossing_rate_%": float(100 * ((lo_ > med) | (med > hi_)).mean()),
    }


def metrics_by_horizon(pred, y, meta, hs=(1, 12, 24)):
    """Paper's '60/720/1440 min' = forecast step 1h / 12h / 24h ahead (data is hourly)."""
    return {f"h{h}": metrics(pred[:, h - 1:h], y[:, h - 1:h], meta) for h in hs}


# %% 5. Classical baselines: ARIMA, simplified ESN ensemble ------------------
def arima_baseline(load, meta, n_origins=150, hist_len=24 * 28):
    from statsmodels.tsa.arima.model import ARIMA
    i2 = meta["i2"]; rng = np.random.RandomState(0)
    origins = np.sort(rng.choice(np.arange(i2, meta["n"] - HORIZON), n_origins, replace=False))
    P, Y = [], []
    for t in origins:
        try:
            r = ARIMA(load[t - hist_len:t], order=(2, 1, 2)).fit()
            fc = r.get_forecast(HORIZON); m = fc.predicted_mean; ci = fc.conf_int(alpha=0.2)
            P.append(np.stack([ci[:, 0], m, ci[:, 1]], -1)); Y.append(load[t:t + HORIZON])
        except Exception:
            continue
    return np.asarray(P, np.float32), np.asarray(Y, np.float32)


def esn_baseline(sets, n_res=10, size=200, rho=0.9, ridge=1e-3, stride=4, seed=0):
    """SIMPLIFIED ESN ensemble (independent reservoirs + ridge readout, Gaussian intervals).
    It is NOT the collectively-trained ensemble of Mallea et al. [16]; say so in the paper
    or use the authors' code if available."""
    rs = np.random.RandomState(seed); z = 1.2816                          # 80% interval
    def states(X, Win, W):
        s = np.zeros((X.shape[0], size), np.float32)
        for t in range(X.shape[1]): s = np.tanh(X[:, t] @ Win + s @ W)
        return s
    Xtr, _, Ytr, _ = sets["train"]; Xtr, Ytr = Xtr[::stride], Ytr[::stride]
    preds_val, preds_te = [], []
    for _ in range(n_res):
        Win = rs.uniform(-0.5, 0.5, (1, size)).astype(np.float32)
        W = rs.randn(size, size).astype(np.float32); W *= rho / max(abs(np.linalg.eigvals(W)))
        S = states(Xtr, Win, W); R = np.linalg.solve(S.T @ S + ridge * np.eye(size), S.T @ Ytr)
        preds_val.append(states(sets["val"][0], Win, W) @ R); preds_te.append(states(sets["test"][0], Win, W) @ R)
    mv, mt = np.mean(preds_val, 0), np.mean(preds_te, 0)
    sd = (sets["val"][2] - mv).std(0, keepdims=True)                     # per-horizon residual std from validation
    return np.stack([mt - z * sd, mt, mt + z * sd], -1).astype(np.float32)


# %% 6. Experiment runner ----------------------------------------------------
MODELS = {"TCN": "tcn", "TCN-LSTM": "tcn_lstm", "TCN-Attention": "tcn_attn",
          "CNN-LSTM-Attention": "cnn_lstm_attn", "HATCN-LSTM-MLP": "hatcn"}


def run_all():
    sets, meta, load, ex = prepare()
    rows, histories, keep = [], {}, {}
    Yte = sets["test"][2]
    for name, kind in MODELS.items():
        for seed in SEEDS:
            set_seed(seed)
            m = Net(kind, meta["ex_dim"], USE_EXOG); h = train(m, sets)
            p = predict(m, sets["test"])
            histories[(name, seed)] = h
            if seed == SEEDS[0]: keep[name] = p
            for hk, mt in metrics_by_horizon(p, Yte, meta).items():
                rows.append(dict(model=name, seed=seed, horizon=hk, **mt))
            rows.append(dict(model=name, seed=seed, horizon="all24", **metrics(p, Yte, meta)))
            print(name, seed, "done")
    # classical baselines (single run)
    pe = esn_baseline(sets); keep["ESN-Ensemble (simplified)"] = pe
    for hk, mt in metrics_by_horizon(pe, Yte, meta).items(): rows.append(dict(model="ESN-Ensemble (simplified)", seed=0, horizon=hk, **mt))
    rows.append(dict(model="ESN-Ensemble (simplified)", seed=0, horizon="all24", **metrics(pe, Yte, meta)))
    pa, ya = arima_baseline(load, meta)
    for hk, mt in metrics_by_horizon(pa, ya, meta).items(): rows.append(dict(model="ARIMA(2,1,2)", seed=0, horizon=hk, **mt))
    rows.append(dict(model="ARIMA(2,1,2)", seed=0, horizon="all24", **metrics(pa, ya, meta)))

    res = pd.DataFrame(rows); res.to_csv(f"{OUT}/all_runs.csv", index=False)
    summ = res.groupby(["model", "horizon"]).agg(["mean", "std"]).drop(columns="seed")
    summ.round(4).to_csv(f"{OUT}/summary_mean_std.csv")      # use ONE decimal precision when building Tables 3-4
    json.dump({f"{k[0]}|{k[1]}": v for k, v in histories.items()}, open(f"{OUT}/loss_histories.json", "w"))
    return sets, meta, keep, histories


# %% 7. Figures: loss curves (Fig 4/5), forecast + prediction intervals (Fig 6) --
def plot_losses(histories, names, fname, seed=0):
    plt.figure(figsize=(7, 4))
    for n in names:
        h = np.array(histories[(n, seed)])
        l, = plt.plot(h[:, 0], label=f"{n} (train)"); plt.plot(h[:, 1], "--", color=l.get_color(), label=f"{n} (val)")
    plt.xlabel("Epoch"); plt.ylabel("Pinball loss (normalised)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(f"{OUT}/{fname}", dpi=300); plt.close()


def plot_forecast(sets, meta, keep, start=0, days=7, fname="fig6_forecast.png", models=None):
    """Median of every model + 10-90% band of the proposed model. One-step-per-day stitching of 24h windows."""
    Y = sets["test"][2]; n = days; rng = meta["hi"] - meta["lo"]; inv = lambda a: a * rng + meta["lo"]
    sl = slice(start, start + 24 * n, 24)
    y = inv(Y[sl]).reshape(-1); plt.figure(figsize=(11, 4)); plt.plot(y, "k", lw=1.5, label="Actual")
    for name in (models or list(keep)):
        p = inv(keep[name][sl][..., 1]).reshape(-1); plt.plot(p, lw=1, label=name)
    P = keep["HATCN-LSTM-MLP"][sl]
    plt.fill_between(range(len(y)), inv(P[..., 0]).reshape(-1), inv(P[..., 2]).reshape(-1), alpha=.2, label="HATCN 10-90% band")
    plt.xlabel("Hour"); plt.ylabel("Load (MW)"); plt.legend(fontsize=7, ncol=3); plt.tight_layout()
    plt.savefig(f"{OUT}/{fname}", dpi=300); plt.close()


if __name__ == "__main__":
    sets, meta, keep, histories = run_all()
    plot_losses(histories, ["TCN", "TCN-LSTM", "TCN-Attention", "HATCN-LSTM-MLP"], "fig4_loss_tcn_family.png")
    plot_losses(histories, ["CNN-LSTM-Attention", "HATCN-LSTM-MLP"], "fig5_loss_baselines.png")
    plot_forecast(sets, meta, keep)
    print("Done. See ./results")