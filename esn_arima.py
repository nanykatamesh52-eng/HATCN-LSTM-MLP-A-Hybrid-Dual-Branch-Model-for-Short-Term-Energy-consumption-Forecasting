"""
ESN Ensemble and ARIMA baselines on the SAME pipeline as baselines.py
(same data, split, train-only normalization, test windows, seeds).
They write preds/esn_seed<s>.npz and preds/arima_seed0.npz in the same
format, so make_tables.py picks them up automatically.

    python esn_arima.py esn                 # 3 seeds
    python esn_arima.py arima               # needs statsmodels
    python make_tables.py

IMPORTANT - these are re-implementations written for this revision, not the
authors' original code. Say so in the paper and describe them in Section 4.

ESN Ensemble (in the spirit of Mallea et al. 2026)
  * K leaky reservoirs with different random weights, driven by the whole
    series (input = normalized load + 8 calendar features at time t).
  * ONE readout shared by all reservoirs (trained collectively): ridge
    regression on [all K reservoir states at t-1 | the same exogenous vector E
    that the neural models receive | 1] -> 24 future values.
  * Ridge strength chosen on the validation set.
  * Prediction interval: point forecast + per-step empirical residual
    quantiles (0.1 / 0.9) of the VALIDATION set. Hence its PICP is calibrated
    on validation by construction; mention this when comparing.

ARIMA
  * Non-seasonal ARIMA(p,d,q) fitted on the TRAIN part only (default (2,1,2)),
    parameters then frozen; the Kalman filter is run over the whole series so
    the forecast at each origin only uses data up to t-1.
  * Gaussian intervals from the psi-weights: mean +- 1.2816 * sigma * sqrt(sum psi^2).
"""
import os
import json
import argparse
import numpy as np

LOOKBACK, HORIZON = 168, 24
PRED_DIR = "preds"
os.makedirs(PRED_DIR, exist_ok=True)
Z90 = 1.2815515655446004   # N(0,1) 0.9-quantile


# ------------------------------------------------------------ index helpers
def window_ts(n, split=(0.70, 0.15, 0.15)):
    """Forecast-origin index t of every window, replicating make_windows()."""
    n_tr, n_va = int(n * split[0]), int(n * split[1])
    cuts = [0, n_tr, n_tr + n_va, n]
    return [np.arange(max(a, LOOKBACK), b - HORIZON + 1)
            for a, b in zip(cuts[:-1], cuts[1:])], n_tr


# ---------------------------------------------------------------------- ESN
def run_reservoirs(U, K, N, seed, rho=0.9, in_scale=0.5, leak=0.3, density=0.1):
    """Return states (T, K*N) float32. State at row t has consumed U[0..t]."""
    rng = np.random.default_rng(seed)
    d = U.shape[1]
    Win = (rng.uniform(-1, 1, (K, N, d)) * in_scale).astype(np.float32)
    W = rng.uniform(-1, 1, (K, N, N)) * (rng.random((K, N, N)) < density)
    for k in range(K):
        ev = np.max(np.abs(np.linalg.eigvals(W[k])))
        W[k] *= rho / max(ev, 1e-8)
    W = W.astype(np.float32)
    T = len(U)
    S = np.zeros((T, K * N), np.float32)
    x = np.zeros((K, N), np.float32)
    U = U.astype(np.float32)
    for t in range(T):
        pre = np.matmul(W, x[:, :, None])[:, :, 0] + Win @ U[t]
        x = (1 - leak) * x + leak * np.tanh(pre)
        S[t] = x.ravel()
    return S


def _feats(S, idx, E):
    return np.hstack([S[idx - 1], E.astype(np.float32),
                      np.ones((len(idx), 1), np.float32)])


def _gram(S, idx, E, Y, chunk=20000):
    A = B = None
    for i in range(0, len(idx), chunk):
        F = _feats(S, idx[i:i + chunk], E[i:i + chunk]).astype(np.float64)
        a, b = F.T @ F, F.T @ Y[i:i + chunk]
        A = a if A is None else A + a
        B = b if B is None else B + b
    return A, B


def _predict(S, idx, E, Wout, chunk=20000):
    return np.vstack([_feats(S, idx[i:i + chunk], E[i:i + chunk]) @ Wout
                      for i in range(0, len(idx), chunk)])


def esn_forecast(U, ts, E, Y, seed, K=10, N=200, lams=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100),
                 **res_kw):
    """ts/E/Y: lists [train, val, test]. Returns test preds (n_te, 24, 3)."""
    S = run_reservoirs(U, K, N, seed, **res_kw)
    A, B = _gram(S, ts[0], E[0], Y[0])
    best = (np.inf, None, None)
    for lam in lams:
        Wout = np.linalg.solve(A + lam * np.eye(len(A)), B)
        rmse = float(np.sqrt(np.mean((_predict(S, ts[1], E[1], Wout) - Y[1]) ** 2)))
        print(f"  seed {seed} ridge {lam:g}: val RMSE {rmse:.5f}")
        if rmse < best[0]:
            best = (rmse, lam, Wout)
    _, lam, Wout = best
    print(f"  seed {seed}: chosen ridge {lam:g}")
    res = Y[1] - _predict(S, ts[1], E[1], Wout)              # validation residuals
    q_lo = np.minimum(np.quantile(res, 0.1, axis=0), 0.0)
    q_hi = np.maximum(np.quantile(res, 0.9, axis=0), 0.0)
    med = _predict(S, ts[2], E[2], Wout)
    return np.stack([med + q_lo, med, med + q_hi], axis=-1)


# -------------------------------------------------------------------- ARIMA
def _shift(x, k):
    """out[t] = x[t-k] (zeros before the start)."""
    out = np.zeros_like(x)
    if k == 0:
        return x.copy()
    out[k:] = x[:-k]
    return out


def arima_predict(u, phi, theta, sigma2, ts_test, H=HORIZON):
    """ARIMA(p,1,q) with FROZEN parameters, no statsmodels internals.
    w_t = u_t - u_{t-1} follows ARMA(p,q):  w_t = sum phi_i w_{t-i} + e_t + sum theta_j e_{t-j}.
    Residuals e are computed recursively (causal), then h-step forecasts are
    made from EVERY origin at once. Returns (len(ts_test), H, 3)."""
    phi, theta = np.asarray(phi, float), np.asarray(theta, float)
    p, q = len(phi), len(theta)
    w = np.r_[0.0, np.diff(u)]
    T = len(w)
    e = np.zeros(T)
    for t in range(T):
        pr = 0.0
        for i in range(p):
            if t - 1 - i >= 0:
                pr += phi[i] * w[t - 1 - i]
        for j in range(q):
            if t - 1 - j >= 0:
                pr += theta[j] * e[t - 1 - j]
        e[t] = w[t] - pr

    wh = {}                                   # wh[h][t] = E[w_{t+h} | info up to t]
    for h in range(1, H + 1):
        s = np.zeros(T)
        for i in range(1, p + 1):
            s += phi[i - 1] * (_shift(w, i - h) if h - i <= 0 else wh[h - i])
        for j in range(h, q + 1):
            s += theta[j - 1] * _shift(e, j - h)
        wh[h] = s
    fc = np.zeros((T, H))
    acc = np.zeros(T)
    for h in range(1, H + 1):
        acc = acc + wh[h]
        fc[:, h - 1] = u + acc                # integrate (d = 1)

    # Gaussian interval from psi-weights of the integrated ARMA
    ar = np.polymul(np.r_[1.0, -phi], [1.0, -1.0])
    ma = np.r_[1.0, theta]
    psi = np.zeros(H)
    for k in range(H):
        v = ma[k] if k < len(ma) else 0.0
        for i in range(1, min(k, len(ar) - 1) + 1):
            v -= ar[i] * (psi[k - i] if k - i > 0 else 1.0)
        psi[k] = v
    sd = np.sqrt(sigma2 * np.cumsum(psi ** 2))
    med = fc[ts_test - 1]                     # forecast origin = last observed t-1
    print(f"  in-sample 1-step RMSE (train recursion): "
          f"{np.sqrt(np.mean(e[1000:] ** 2)):.5f}  (sqrt(sigma2) = {np.sqrt(sigma2):.5f})")
    return np.stack([med - Z90 * sd, med, med + Z90 * sd], axis=-1)


def arima_forecast(u, n_tr, ts_test, order=(2, 1, 2)):
    from statsmodels.tsa.arima.model import ARIMA
    p, d, q = order
    assert d == 1, "this implementation supports d = 1 only"
    print(f"fitting ARIMA{order} on {n_tr} train points ...")
    res = ARIMA(u[:n_tr], order=order).fit()
    print(res.summary().tables[1])
    return arima_predict(u, np.asarray(res.arparams), np.asarray(res.maparams),
                         float(res.params[-1]), ts_test)


# ------------------------------------------------------------------- driver
def _setup():
    from hatcn_rebuild import build_features, make_windows
    import hatcn_rebuild as hr
    assert (hr.LOOKBACK, hr.HORIZON) == (LOOKBACK, HORIZON)
    df, cal = build_features()
    data, lo, hi = make_windows(df, cal)
    ts, n_tr = window_ts(len(df))
    for k in range(3):                                      # alignment guards
        assert len(ts[k]) == len(data[k][0]), "window count mismatch"
    u = ((df["load"].values - lo) / (hi - lo + 1e-9)).astype(np.float64)
    assert abs(float(data[2][0][0, -1]) - u[ts[2][0] - 1]) < 1e-5, "index misaligned"
    np.save(f"{PRED_DIR}/y_test.npy", data[2][2].numpy())
    with open(f"{PRED_DIR}/meta.json", "w") as f:
        json.dump({"load_lo": lo, "load_hi": hi}, f)
    return df, cal, data, ts, n_tr, u


def cmd_esn(a):
    df, cal, data, ts, _, u = _setup()
    U = np.hstack([u[:, None], cal.values])
    E = [data[k][1].numpy() for k in range(3)]
    Y = [data[k][2].numpy().astype(np.float64) for k in range(3)]
    for s in a.seeds:
        out = f"{PRED_DIR}/esn_seed{s}.npz"
        if os.path.exists(out):
            print("skip", out)
            continue
        pred = esn_forecast(U, ts, E, Y, s, K=a.K, N=a.N)
        np.savez_compressed(out, pred=pred.astype(np.float32))


def cmd_arima(a):
    df, cal, data, ts, n_tr, u = _setup()
    pred = arima_forecast(u, n_tr, ts[2], tuple(a.order))
    np.savez_compressed(f"{PRED_DIR}/arima_seed0.npz", pred=pred.astype(np.float32))
    print("saved arima")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("esn")
    e.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    e.add_argument("--K", type=int, default=10, help="number of reservoirs")
    e.add_argument("--N", type=int, default=200, help="units per reservoir")
    e.set_defaults(fn=cmd_esn)
    r = sub.add_parser("arima")
    r.add_argument("--order", nargs=3, type=int, default=[2, 1, 2])
    r.set_defaults(fn=cmd_arima)
    args = ap.parse_args()
    args.fn(args)