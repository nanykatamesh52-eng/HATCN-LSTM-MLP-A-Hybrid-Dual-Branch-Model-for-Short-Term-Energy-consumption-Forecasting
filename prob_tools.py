"""
Tools for the HATCN-LSTM-MLP revision:
  1) NonCrossingHead  : PyTorch output head that guarantees lower <= median <= upper
  2) pinball_loss     : training loss for q = (0.1, 0.5, 0.9)
  3) evaluate_prob    : PICP, PINAW, Winkler/Interval score, pinball, crossing rate

Shapes: y_true (N, H)   preds (N, H, 3) ordered as [q0.1, q0.5, q0.9]
Run `python prob_tools.py` for a numpy-only self-test of the metrics.
"""
import numpy as np

QUANTILES = (0.1, 0.5, 0.9)
NOMINAL = QUANTILES[2] - QUANTILES[0]   # 0.8  -> the 0.1-0.9 interval is an 80% interval
ALPHA = 1.0 - NOMINAL                   # 0.2


# ---------------------------------------------------------------- 1) model side
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class NonCrossingHead(nn.Module):
        """Replace the old Dense(72)->reshape(24,3).
        Predict a median, then two strictly positive gaps:
            q50 = m
            q10 = m - softplus(a)
            q90 = m + softplus(b)
        so q10 <= q50 <= q90 holds BY CONSTRUCTION (no post-hoc sorting)."""

        def __init__(self, in_dim, horizon=24):
            super().__init__()
            self.h = horizon
            self.fc = nn.Linear(in_dim, horizon * 3)

        def forward(self, z):                      # z: (B, in_dim)
            out = self.fc(z).view(-1, self.h, 3)
            m = out[..., 0]
            lo = m - F.softplus(out[..., 1])
            hi = m + F.softplus(out[..., 2])
            return torch.stack([lo, m, hi], dim=-1)   # (B, H, 3)

    def pinball_loss(pred, y, quantiles=QUANTILES):
        """pred (B,H,3), y (B,H)"""
        q = torch.tensor(quantiles, device=pred.device).view(1, 1, -1)
        e = y.unsqueeze(-1) - pred
        return torch.maximum(q * e, (q - 1) * e).mean()
except ImportError:  # metrics below work without torch
    pass


# ------------------------------------------------------------- 2) metrics side
def crossing_rate(preds):
    """Fraction of (sample, step) pairs where quantiles are NOT ordered."""
    bad = (preds[..., 0] > preds[..., 1]) | (preds[..., 1] > preds[..., 2])
    return float(bad.mean())


def evaluate_prob(y, preds, y_train_range=None):
    """y (N,H); preds (N,H,3). Use the SAME scale for every model (e.g. normalized)."""
    lo, med, hi = preds[..., 0], preds[..., 1], preds[..., 2]
    rng = y_train_range if y_train_range else (y.max() - y.min())

    picp = float(np.mean((y >= lo) & (y <= hi)))                 # coverage
    pinaw = float(np.mean(hi - lo) / rng)                        # normalized width
    # Winkler / interval score (Gneiting & Raftery) for the (1-ALPHA) interval
    width = hi - lo
    winkler = width + (2 / ALPHA) * (lo - y) * (y < lo) + (2 / ALPHA) * (y - hi) * (y > hi)
    # mean pinball over the three quantiles
    pin = []
    for k, q in enumerate(QUANTILES):
        e = y - preds[..., k]
        pin.append(np.mean(np.maximum(q * e, (q - 1) * e)))
    return {
        "nominal_coverage": NOMINAL,
        "PICP": picp,
        "coverage_gap(PICP-nominal)": picp - NOMINAL,
        "PINAW": pinaw,
        "Winkler": float(winkler.mean()),
        "pinball_q10": float(pin[0]), "pinball_q50": float(pin[1]), "pinball_q90": float(pin[2]),
        "pinball_mean": float(np.mean(pin)),
        "crossing_rate": crossing_rate(preds),
        # point metrics on the median, for the same table
        "RMSE": float(np.sqrt(np.mean((y - med) ** 2))),
        "MAE": float(np.mean(np.abs(y - med))),
        "MSE": float(np.mean((y - med) ** 2)),
    }


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, H = 2000, 24
    y = rng.normal(0.5, 0.1, size=(N, H))
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.9)
    good = np.stack([0.5 - z * 0.1 + 0 * y, 0.5 + 0 * y, 0.5 + z * 0.1 + 0 * y], -1)
    print("well-calibrated model:")
    for k, v in evaluate_prob(y, good).items():
        print(f"  {k:28s} {v:.4f}")
    bad = good.copy()
    bad[..., [0, 2]] = bad[..., [2, 0]]    # force crossing
    print("crossing_rate when lower/upper swapped:", crossing_rate(bad))