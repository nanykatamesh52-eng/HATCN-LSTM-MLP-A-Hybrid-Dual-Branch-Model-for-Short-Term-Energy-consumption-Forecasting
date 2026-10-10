"""
Informer / Autoformer / TFT baselines (neuralforecast) on the SAME split and
the SAME 21,787 test windows as baselines.py. Output format is identical:
preds/<model>_seed<s>.npz  with key "pred" = (N, 24, 3) = [q0.1, q0.5, q0.9],
so `python make_tables.py` picks them up.

    pip install neuralforecast          # >= 1.6 needed (refit=False)
    python transformers_nf.py --models informer autoformer tft --seeds 0 1 2
    python make_tables.py

Protocol (mirrors the other models)
  * Normalized load (train-only min-max, same as hatcn_rebuild.py) is the target.
  * Train = first 70%, validation = next 15% (early stopping), test = last 15%.
  * Lookback 168 h, horizon 24 h, quantile loss MQLoss for q = 0.1 / 0.5 / 0.9.
  * Inputs are NOT identical to the HATCN family (library limitation, say so):
      Informer / Autoformer : future-known calendar (8 features) + weather
                              shifted by 24 h (value at s = weather at s-24, so
                              it is known at the forecast origin; no leakage).
      TFT                   : future-known calendar + observed weather history.
    The HATCN-family models receive the latest observed weather instead.
  * Per-window 'standard' scaling inside the library (scaler_type) for all three.
  * Quantile heads here are NOT non-crossing: the crossing rate in the table is
    the raw one. That is itself useful evidence for reviewer comment R1.3.

Resumable: existing preds files are skipped. Training uses early stopping on
validation (val_check_steps=100, patience=5 checks), max_steps default 2000.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd

H, L = 24, 168
PRED_DIR = "preds"
os.makedirs(PRED_DIR, exist_ok=True)
CAL = ["h_sin", "h_cos", "d_sin", "d_cos", "m_sin", "m_cos", "weekend", "holiday"]


# ---------------------------------------------------------------- data frame
def build_frame():
    from hatcn_rebuild import build_features
    df, cal = build_features()
    n = len(df)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    lo, hi = float(df["load"].iloc[:n_tr].min()), float(df["load"].iloc[:n_tr].max())
    f = pd.DataFrame({"unique_id": "load", "ds": df.index,
                      "y": ((df["load"].values - lo) / (hi - lo + 1e-9))})
    for c in CAL:
        f[c] = cal[c].values.astype(np.float32)
    weather = [c for c in df.columns if c != "load"]
    for c in weather:                                   # train-only min-max
        a, b = float(df[c].iloc[:n_tr].min()), float(df[c].iloc[:n_tr].max())
        f[c] = ((df[c].values - a) / (b - a + 1e-9)).astype(np.float32)
        f[c + "_lag24"] = f[c].shift(24).bfill().values
    return f, weather, n, n_tr, n_va, df.index.freqstr


# --------------------------------------------------------- output conversion
def cv_to_preds(cv, name, y_test):
    """neuralforecast cross-validation frame -> (N, 24, 3) aligned with y_test."""
    cols = [c for c in cv.columns if c.startswith(name)]

    def pick(keys):
        for c in cols:
            if any(k in c for k in keys):
                return c
        raise KeyError(f"no column for {keys} among {cols}")

    lo_c, med_c, hi_c = pick(["ql0.1", "lo-80"]), pick(["ql0.5", "median"]), pick(["ql0.9", "hi-80"])
    cv = cv.sort_values(["cutoff", "ds"])
    assert len(cv) == len(y_test) * H, f"rows {len(cv)} != {len(y_test)}*{H}"
    y = cv["y"].values.reshape(-1, H)
    assert np.allclose(y, y_test, atol=1e-4), "windows NOT aligned with y_test.npy"
    return np.stack([cv[c].values.reshape(-1, H) for c in (lo_c, med_c, hi_c)], axis=-1)


# --------------------------------------------------------------------- models
def make_loss():
    from neuralforecast.losses.pytorch import MQLoss
    try:
        return MQLoss(quantiles=[0.1, 0.5, 0.9])
    except TypeError:                                   # older API
        return MQLoss(level=[80])


def run_one(name, seed, frame, weather, n_va, n_te_steps, freq, max_steps, y_test):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import Informer, Autoformer, TFT

    common = dict(h=H, input_size=L, loss=make_loss(), scaler_type="standard",
                  max_steps=max_steps, val_check_steps=100,
                  early_stop_patience_steps=5, learning_rate=1e-3,
                  random_seed=seed)
    lr = 1e-4 if name == "autoformer" else 1e-3
    common["learning_rate"] = lr
    lag = [c + "_lag24" for c in weather]
    if name == "informer":
        futr, hist = CAL + lag, []
        model = Informer(futr_exog_list=futr, **common)
    elif name == "autoformer":
        futr, hist = CAL + lag, []
        model = Autoformer(futr_exog_list=futr, **common)
    else:
        futr, hist = CAL, weather
        model = TFT(futr_exog_list=futr, hist_exog_list=hist, **common)

    data = frame[["unique_id", "ds", "y"] + futr + hist]
    nf = NeuralForecast(models=[model], freq=freq)
    cv = nf.cross_validation(df=data, val_size=n_va, test_size=n_te_steps,
                             step_size=1, n_windows=None, refit=False)
    return cv_to_preds(cv.reset_index(drop=True), model.__class__.__name__, y_test)


def main():
    import neuralforecast
    print("neuralforecast", neuralforecast.__version__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["informer", "autoformer", "tft"],
                    choices=["informer", "autoformer", "tft"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max_steps", type=int, default=2000)
    a = ap.parse_args()

    y_test = np.load(f"{PRED_DIR}/y_test.npy")
    frame, weather, n, n_tr, n_va, freq = build_frame()
    n_te_steps = len(y_test) + H - 1
    assert n - (n_tr + n_va) == n_te_steps, "test length mismatch with the other models"
    for name in a.models:
        for s in a.seeds:
            out = f"{PRED_DIR}/{name}_seed{s}.npz"
            if os.path.exists(out):
                print("skip", out)
                continue
            print(f"=== {name} seed {s}")
            pred = run_one(name, s, frame, weather, n_va, n_te_steps, freq,
                           a.max_steps, y_test)
            np.savez_compressed(out, pred=pred.astype(np.float32))
            print("saved", out, pred.shape,
                  "crossing:", float(((pred[..., 0] > pred[..., 1]) |
                                       (pred[..., 1] > pred[..., 2])).mean()))


if __name__ == "__main__":
    main()