"""
Step 1 of the revision: run HATCN-LSTM-MLP and the baselines on the SAME
pipeline (same data, split, normalization, windows, head, loss, seeds)
and save the test predictions so tables/figures can be built from them.

Put this file next to hatcn_rebuild.py and prob_tools.py, then:

    python baselines.py train --models hatcn tcn tcn_lstm tcn_attn cnn_lstm_attn
    python baselines.py naive
    python make_tables.py

Ablation / head study (separate table: tables/table_ablation_*.md):
    python baselines.py train --models hatcn_dense hatcn_noattn hatcn_noweather hatcn_nomlp

Output (resumable: existing files are skipped):
    preds/y_test.npy                  normalized test targets  (N, 24)
    preds/meta.json                   train min/max of the load (for MW scale)
    preds/<model>_seed<s>.npz         key "pred": (N, 24, 3) = [q0.1, q0.5, q0.9]
    histories/<model>_seed<s>.csv     train/val pinball loss per epoch

Design choice (important for fairness): every neural model gets the SAME inputs
(168 h of load + future calendar + latest weather) through the SAME MLP branch,
and the SAME NonCrossingHead + pinball loss. So differences come only from the
temporal encoder. Ablations without the MLP branch are a separate step.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from hatcn_rebuild import (
    build_features, make_windows, TCNBlock, HATCN_LSTM_MLP,
    predict, DEV, HORIZON, LOOKBACK,
)
from prob_tools import NonCrossingHead, pinball_loss

PRED_DIR, HIST_DIR = "preds", "histories"
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)


# ------------------------------------------------------------------ models
class Variant(nn.Module):
    """TCN / TCN-LSTM / TCN-Attention / CNN-LSTM-Attention with the same
    exogenous MLP branch, fusion block and non-crossing head as HATCN."""

    def __init__(self, kind, exog_dim, ch=64, lstm=128):
        super().__init__()
        self.kind = kind
        if kind == "cnn_lstm_attn":      # plain (non-dilated) CNN encoder
            self.enc = nn.Sequential(
                nn.Conv1d(1, ch, 3, padding=1), nn.ReLU(), nn.Dropout(0.2),
                nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(), nn.Dropout(0.2),
            )
        else:                            # same TCN as the proposed model
            self.enc = nn.Sequential(
                TCNBlock(1, ch, 3, 1), TCNBlock(ch, ch, 3, 2), TCNBlock(ch, ch, 3, 4)
            )
        self.use_lstm = kind in ("tcn_lstm", "cnn_lstm_attn")
        self.use_attn = kind in ("tcn_attn", "cnn_lstm_attn")
        d = ch
        if self.use_lstm:
            self.lstm = nn.LSTM(ch, lstm, batch_first=True)
            d = lstm
        if self.use_attn:
            self.att_W = nn.Linear(d, d)
            self.att_v = nn.Linear(d, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(exog_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        self.fuse = nn.Sequential(nn.Linear(d + 32, 128), nn.ReLU(), nn.Dropout(0.3))
        self.head = NonCrossingHead(128, HORIZON)

    def forward(self, x, e):
        h = self.enc(x.unsqueeze(1)).transpose(1, 2)        # (B, T, C)
        if self.use_lstm:
            h, _ = self.lstm(h)
        if self.use_attn:
            a = torch.softmax(self.att_v(torch.tanh(self.att_W(h))), dim=1)
            ctx = (a * h).sum(dim=1)
        else:
            ctx = h[:, -1]                                   # last (causal) step
        z = self.fuse(torch.cat([ctx, self.mlp(e)], dim=1))
        return self.head(z)



N_CAL = 8   # calendar features per future hour; E = [24*8 calendar | weather]


class DenseQuantileHead(nn.Module):
    """The ORIGINAL head: Dense(72) -> reshape (24, 3). Quantiles may cross."""

    def __init__(self, in_dim, horizon=24):
        super().__init__()
        self.h = horizon
        self.fc = nn.Linear(in_dim, horizon * 3)

    def forward(self, z):
        return self.fc(z).view(-1, self.h, 3)          # [q0.1, q0.5, q0.9]


class HATCNAbl(nn.Module):
    """HATCN-LSTM-MLP with switches for the ablation / head study.
    With all defaults it is the proposed model."""

    def __init__(self, exog_dim, head="noncross", use_mlp=True,
                 use_weather=True, use_attn=True, ch=64, lstm=128):
        super().__init__()
        self.use_mlp, self.use_attn = use_mlp, use_attn
        self.exog_in = exog_dim if use_weather else min(exog_dim, HORIZON * N_CAL)
        self.tcn = nn.Sequential(
            TCNBlock(1, ch, 3, 1), TCNBlock(ch, ch, 3, 2), TCNBlock(ch, ch, 3, 4)
        )
        self.lstm = nn.LSTM(ch, lstm, batch_first=True)
        if use_attn:
            self.att_W = nn.Linear(lstm, lstm)
            self.att_v = nn.Linear(lstm, 1, bias=False)
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.exog_in, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
            )
        self.fuse = nn.Sequential(
            nn.Linear(lstm + (32 if use_mlp else 0), 128), nn.ReLU(), nn.Dropout(0.3)
        )
        self.head = (NonCrossingHead(128, HORIZON) if head == "noncross"
                     else DenseQuantileHead(128, HORIZON))

    def forward(self, x, e):
        h = self.tcn(x.unsqueeze(1)).transpose(1, 2)
        h, _ = self.lstm(h)
        if self.use_attn:
            a = torch.softmax(self.att_v(torch.tanh(self.att_W(h))), dim=1)
            ctx = (a * h).sum(dim=1)
        else:
            ctx = h[:, -1]                               # last LSTM state
        z = torch.cat([ctx, self.mlp(e[:, :self.exog_in])], 1) if self.use_mlp else ctx
        return self.head(self.fuse(z))


# name -> kwargs. These are NOT part of the main comparison table.
ABL = {
    "hatcn_dense":     dict(head="dense"),          # crossing allowed (old head)
    "hatcn_noattn":    dict(use_attn=False),
    "hatcn_noweather": dict(use_weather=False),     # calendar only
    "hatcn_nomlp":     dict(use_mlp=False),         # no calendar, no weather
}

def make_model(name, exog_dim):
    if name == "hatcn":
        return HATCN_LSTM_MLP(exog_dim)
    if name in ABL:
        return HATCNAbl(exog_dim, **ABL[name])
    return Variant(name, exog_dim)


NAMES = ["hatcn", "tcn", "tcn_lstm", "tcn_attn", "cnn_lstm_attn"]
ALL_NAMES = NAMES + list(ABL)


# ---------------------------------------------------------------- training
def train_one(name, seed, data, epochs=60, patience=10, bs=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    (Xtr, Etr, Ytr), (Xva, Eva, Yva), (Xte, Ete, Yte) = data
    model = make_model(name, Etr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best, bad, state, hist = float("inf"), 0, None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        losses = []
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            loss = pinball_loss(model(Xtr[j].to(DEV), Etr[j].to(DEV)), Ytr[j].to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        tr = float(np.mean(losses))
        va = pinball_loss(torch.tensor(predict(model, Xva, Eva)), Yva).item()
        hist.append({"model": name, "seed": seed, "epoch": ep + 1,
                     "train_pinball_loss": tr, "val_pinball_loss": va})
        print(f"{name} seed {seed} ep {ep + 1:02d} train {tr:.5f} val {va:.5f}")
        if va < best:
            best, bad = va, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep + 1}")
                break

    model.load_state_dict(state)
    return predict(model, Xte, Ete), pd.DataFrame(hist)


def load_data():
    df, cal = build_features()
    data, lo, hi = make_windows(df, cal)
    # shared artefacts for the table/figure scripts
    np.save(f"{PRED_DIR}/y_test.npy", data[2][2].numpy())
    with open(f"{PRED_DIR}/meta.json", "w") as f:
        json.dump({"load_lo": lo, "load_hi": hi}, f)
    return data


def cmd_train(args):
    data = load_data()
    for name in args.models:
        for s in args.seeds:
            out = f"{PRED_DIR}/{name}_seed{s}.npz"
            if os.path.exists(out):
                print(f"skip {out}")
                continue
            pred, hist = train_one(name, s, data)
            np.savez_compressed(out, pred=pred.astype(np.float32))
            hist.to_csv(f"{HIST_DIR}/{name}_seed{s}.csv", index=False)


# ------------------------------------------------------------ seasonal naive
def cmd_naive(args):
    """Seasonal-naive (same hour, previous week) as a sanity-check baseline.
    Median = value 168 h earlier. Interval = naive + per-step empirical
    residual quantiles (0.1 / 0.9) estimated on the TRAIN windows only."""
    data = load_data()
    (Xtr, _, Ytr), _, (Xte, _, _) = data
    naive_tr = Xtr[:, :HORIZON].numpy()          # X[:, h] = load[t + h - 168]
    res = Ytr.numpy() - naive_tr
    q_lo = np.minimum(np.quantile(res, 0.1, axis=0), 0.0)
    q_hi = np.maximum(np.quantile(res, 0.9, axis=0), 0.0)
    med = Xte[:, :HORIZON].numpy()
    pred = np.stack([med + q_lo, med, med + q_hi], axis=-1)
    for s in args.seeds[:1]:                      # deterministic: one "seed"
        np.savez_compressed(f"{PRED_DIR}/seasonal_naive_seed{s}.npz",
                            pred=pred.astype(np.float32))
    print("saved seasonal_naive")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--models", nargs="+", default=NAMES, choices=ALL_NAMES)
    t.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    t.set_defaults(fn=cmd_train)
    n = sub.add_parser("naive")
    n.add_argument("--seeds", nargs="+", type=int, default=[0])
    n.set_defaults(fn=cmd_naive)
    a = ap.parse_args()
    a.fn(a)