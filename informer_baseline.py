"""
Informer-style probabilistic baseline for PJME short-term load forecasting.

Fair-comparison setup with HATCN-LSTM-MLP:
- Dataset: PJME_hourly.csv
- Input: previous 168 hours of load
- Horizon: next 24 hours
- Chronological split: 70% / 15% / 15%
- Train-only Min-Max normalization
- Future-known calendar features
- Quantiles: 0.1 / 0.5 / 0.9
- 3 random seeds
- Early stopping on validation pinball loss
- Reports:
    RMSE, MAE, MAPE
    PICP, PINAW, Winkler
    Pinball loss
    Crossing rate

NOTE:
This is an Informer-style Transformer baseline implemented with
PyTorch's TransformerEncoder. It is intentionally kept simple and
reproducible for a fair baseline comparison. It should be described
as "Informer-style Transformer" unless a canonical ProbSparse
Informer implementation is used.
"""

import copy
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas.tseries.holiday import USFederalHolidayCalendar


# ============================================================
# Configuration
# ============================================================

DATA_FILE = "PJME_hourly.csv"

LOOKBACK = 168
HORIZON = 24

QUANTILES = (0.1, 0.5, 0.9)

SEEDS = (0, 1, 2)

EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 64
LR = 1e-3

D_MODEL = 128
N_HEADS = 8
N_LAYERS = 3
FF_DIM = 256
DROPOUT = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Pinball loss
# ============================================================

def pinball_loss(pred, target):
    """
    pred:
        (B, H, 3)
    target:
        (B, H)
    """
    target = target.unsqueeze(-1)

    q = torch.tensor(
        QUANTILES,
        dtype=pred.dtype,
        device=pred.device
    ).view(1, 1, -1)

    error = target - pred

    loss = torch.maximum(
        q * error,
        (q - 1.0) * error
    )

    return loss.mean()


# ============================================================
# Non-crossing quantile head
# ============================================================

class NonCrossingHead(nn.Module):

    def __init__(self, input_dim, horizon):
        super().__init__()

        self.horizon = horizon

        self.base = nn.Linear(
            input_dim,
            horizon
        )

        self.delta1 = nn.Linear(
            input_dim,
            horizon
        )

        self.delta2 = nn.Linear(
            input_dim,
            horizon
        )

    def forward(self, x):

        q50 = self.base(x)

        q10 = q50 - F.softplus(
            self.delta1(x)
        )

        q90 = q50 + F.softplus(
            self.delta2(x)
        )

        return torch.stack(
            [q10, q50, q90],
            dim=-1
        )


# ============================================================
# Positional encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=512
    ):
        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model
        )

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            )
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


# ============================================================
# Informer-style Transformer baseline
# ============================================================

class InformerStyle(nn.Module):

    def __init__(
        self,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT
    ):
        super().__init__()

        # Historical load is the main encoder input.
        self.load_projection = nn.Linear(
            1,
            d_model
        )

        self.positional = PositionalEncoding(
            d_model,
            max_len=LOOKBACK
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # Direct multi-step probabilistic head.
        self.head = NonCrossingHead(
            d_model,
            HORIZON
        )

    def forward(self, x):

        # x: (B, 168)

        x = x.unsqueeze(-1)

        h = self.load_projection(x)

        h = self.positional(h)

        h = self.encoder(h)

        # Mean pooling over historical context.
        h = h.mean(dim=1)

        h = self.norm(h)

        h = self.dropout(h)

        return self.head(h)


# ============================================================
# Feature construction
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["Datetime"]
    )

    df = (
        df
        .drop_duplicates("Datetime")
        .set_index("Datetime")
        .sort_index()
        .asfreq("h")
        .interpolate()
    )

    if "PJME_MW" not in df.columns:
        raise ValueError(
            "PJME_hourly.csv must contain a 'PJME_MW' column."
        )

    load = df["PJME_MW"].astype(float)

    idx = load.index

    holiday_dates = (
        USFederalHolidayCalendar()
        .holidays(
            idx.min(),
            idx.max()
        )
    )

    # Future-known calendar features.
    cal = np.column_stack([
        np.sin(2 * np.pi * idx.hour / 24),
        np.cos(2 * np.pi * idx.hour / 24),

        np.sin(2 * np.pi * idx.dayofweek / 7),
        np.cos(2 * np.pi * idx.dayofweek / 7),

        np.sin(2 * np.pi * idx.month / 12),
        np.cos(2 * np.pi * idx.month / 12),

        (idx.dayofweek >= 5).astype(float),

        idx.normalize().isin(
            holiday_dates
        ).astype(float)
    ]).astype(np.float32)

    return load.values.astype(np.float32), cal, idx


# ============================================================
# Window creation
# ============================================================

def make_windows(
    load,
    cal,
    split=(0.70, 0.15, 0.15)
):

    n = len(load)

    n_train = int(
        n * split[0]
    )

    n_val = int(
        n * split[1]
    )

    cut1 = n_train
    cut2 = n_train + n_val

    # Train-only normalization.
    train_lo = float(
        load[:n_train].min()
    )

    train_hi = float(
        load[:n_train].max()
    )

    train_range = (
        train_hi - train_lo
    )

    load_norm = (
        load - train_lo
    ) / (
        train_range + 1e-9
    )

    splits = [
        (0, cut1),
        (cut1, cut2),
        (cut2, n)
    ]

    output = []

    for split_id, (a, b) in enumerate(splits):

        X = []
        E = []
        Y = []

        start_t = max(
            a,
            LOOKBACK
        )

        end_t = (
            b - HORIZON + 1
        )

        for t in range(
            start_t,
            end_t
        ):

            # Historical load.
            X.append(
                load_norm[
                    t - LOOKBACK:t
                ]
            )

            # Future-known calendar.
            E.append(
                cal[
                    t:t + HORIZON
                ]
            )

            # Future target.
            Y.append(
                load_norm[
                    t:t + HORIZON
                ]
            )

        X = torch.tensor(
            np.asarray(X),
            dtype=torch.float32
        )

        E = torch.tensor(
            np.asarray(E),
            dtype=torch.float32
        )

        Y = torch.tensor(
            np.asarray(Y),
            dtype=torch.float32
        )

        output.append(
            (X, E, Y)
        )

        print(
            f"Split {split_id}: "
            f"X={tuple(X.shape)}, "
            f"E={tuple(E.shape)}, "
            f"Y={tuple(Y.shape)}"
        )

    return (
        output,
        train_lo,
        train_hi
    )


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict(
    model,
    X
):

    model.eval()

    outputs = []

    for start in range(
        0,
        len(X),
        BATCH_SIZE
    ):

        xb = X[
            start:start + BATCH_SIZE
        ].to(DEVICE)

        pred = model(xb)

        outputs.append(
            pred.cpu()
        )

    return torch.cat(
        outputs,
        dim=0
    ).numpy()


# ============================================================
# Metrics
# ============================================================

def evaluate_prob(
    y_true,
    pred,
    train_range
):

    y_true = np.asarray(
        y_true,
        dtype=np.float64
    )

    pred = np.asarray(
        pred,
        dtype=np.float64
    )

    q10 = pred[:, :, 0]
    q50 = pred[:, :, 1]
    q90 = pred[:, :, 2]

    # Point metrics.
    error = (
        y_true - q50
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    epsilon = 1e-8

    mape = (
        np.mean(
            np.abs(
                error
            )
            /
            np.maximum(
                np.abs(y_true),
                epsilon
            )
        )
        * 100.0
    )

    # Prediction interval.
    covered = (
        (y_true >= q10)
        &
        (y_true <= q90)
    )

    picp = np.mean(
        covered
    )

    nominal = 0.80

    coverage_gap = (
        picp - nominal
    )

    pinaw = (
        np.mean(
            q90 - q10
        )
        /
        max(
            train_range,
            epsilon
        )
    )

    # Winkler score for central 80% interval.
    alpha = 0.20

    interval_width = (
        q90 - q10
    )

    lower_penalty = (
        2.0 / alpha
    ) * (
        q10 - y_true
    )

    upper_penalty = (
        2.0 / alpha
    ) * (
        y_true - q90
    )

    winkler = np.where(
        y_true < q10,
        interval_width
        + lower_penalty,
        np.where(
            y_true > q90,
            interval_width
            + upper_penalty,
            interval_width
        )
    )

    winkler = np.mean(
        winkler
    )

    # Pinball.
    pinballs = []

    for i, q in enumerate(
        QUANTILES
    ):

        e = (
            y_true
            - pred[:, :, i]
        )

        loss = np.maximum(
            q * e,
            (q - 1.0) * e
        )

        pinballs.append(
            np.mean(loss)
        )

    pinball_q10 = pinballs[0]
    pinball_q50 = pinballs[1]
    pinball_q90 = pinballs[2]

    pinball_mean = np.mean(
        pinballs
    )

    crossing_rate = np.mean(
        (
            (q10 > q50)
            |
            (q50 > q90)
        )
    )

    return {
        "nominal_coverage": nominal,
        "PICP": picp,
        "coverage_gap(PICP-nominal)": coverage_gap,
        "PINAW": pinaw,
        "Winkler": winkler,
        "pinball_q10": pinball_q10,
        "pinball_q50": pinball_q50,
        "pinball_q90": pinball_q90,
        "pinball_mean": pinball_mean,
        "crossing_rate": crossing_rate,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": np.mean(
            error ** 2
        ),
        "MAPE": mape
    }


# ============================================================
# Train one seed
# ============================================================

def train_one_seed(
    seed,
    data,
    train_lo,
    train_hi
):

    set_seed(seed)

    (
        train_data,
        val_data,
        test_data
    ) = data

    Xtr, Etr, Ytr = train_data
    Xva, Eva, Yva = val_data
    Xte, Ete, Yte = test_data

    model = InformerStyle().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    history = []

    print(
        f"\n{'=' * 70}"
    )
    print(
        f"STARTING SEED {seed}"
    )
    print(
        f"{'=' * 70}"
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        indices = torch.randperm(
            len(Xtr)
        )

        train_losses = []

        for start in range(
            0,
            len(indices),
            BATCH_SIZE
        ):

            ids = indices[
                start:start + BATCH_SIZE
            ]

            xb = Xtr[ids].to(
                DEVICE
            )

            yb = Ytr[ids].to(
                DEVICE
            )

            pred = model(
                xb
            )

            loss = pinball_loss(
                pred,
                yb
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(train_losses)
        )

        # Validation.
        model.eval()

        val_preds = []

        with torch.no_grad():

            for start in range(
                0,
                len(Xva),
                BATCH_SIZE
            ):

                xb = Xva[
                    start:start + BATCH_SIZE
                ].to(DEVICE)

                val_preds.append(
                    model(xb).cpu()
                )

        val_pred = torch.cat(
            val_preds,
            dim=0
        )

        val_loss = pinball_loss(
            val_pred,
            Yva
        ).item()

        history.append({
            "seed": seed,
            "epoch": epoch,
            "train_pinball_loss": train_loss,
            "val_pinball_loss": val_loss
        })

        print(
            f"seed {seed} | "
            f"epoch {epoch:02d} | "
            f"train {train_loss:.5f} | "
            f"val {val_loss:.5f}"
        )

        if val_loss < best_val:

            best_val = val_loss
            bad_epochs = 0

            best_state = copy.deepcopy(
                model.state_dict()
            )

        else:

            bad_epochs += 1

            if bad_epochs >= PATIENCE:

                print(
                    f"Early stopping at epoch {epoch}"
                )

                break

    if best_state is None:
        raise RuntimeError(
            "No best model state was saved."
        )

    model.load_state_dict(
        best_state
    )

    # Test.
    pred_norm = predict(
        model,
        Xte
    )

    y_norm = Yte.numpy()

    # Normalized metrics.
    metrics_norm = evaluate_prob(
        y_norm,
        pred_norm,
        train_range=1.0
    )

    # Original MW scale.
    train_range = (
        train_hi - train_lo
    )

    y_mw = (
        y_norm * train_range
        + train_lo
    )

    pred_mw = (
        pred_norm * train_range
        + train_lo
    )

    metrics_mw = evaluate_prob(
        y_mw,
        pred_mw,
        train_range=train_range
    )

    return (
        metrics_norm,
        metrics_mw,
        pd.DataFrame(history)
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    load, cal, idx = load_data()

    print("\nDataset:")
    print(
        f"records = {len(load)}"
    )
    print(
        f"start   = {idx.min()}"
    )
    print(
        f"end     = {idx.max()}"
    )

    (
        data,
        train_lo,
        train_hi
    ) = make_windows(
        load,
        cal
    )

    print(
        "\nTraining load range:"
    )
    print(
        f"min = {train_lo:.4f}"
    )
    print(
        f"max = {train_hi:.4f}"
    )

    results = []

    for seed in SEEDS:

        (
            metrics_norm,
            metrics_mw,
            history
        ) = train_one_seed(
            seed,
            data,
            train_lo,
            train_hi
        )

        history.to_csv(
            f"informer_loss_history_seed_{seed}.csv",
            index=False
        )

        results.append({
            "seed": seed,
            **{
                f"norm_{k}": v
                for k, v in metrics_norm.items()
            },
            **{
                f"MW_{k}": v
                for k, v in metrics_mw.items()
            }
        })

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "informer_seed_results.csv",
        index=False
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "FINAL INFORMER-STYLE NORMALIZED RESULTS"
    )
    print(
        "=" * 70
    )

    for key in metrics_norm.keys():

        values = results_df[
            f"norm_{key}"
        ].astype(float)

        print(
            f"{key:28s} "
            f"{values.mean():.4f} "
            f"+- "
            f"{values.std():.4f}"
        )

    print(
        "\n" + "=" * 70
    )
    print(
        "FINAL INFORMER-STYLE ORIGINAL MW RESULTS"
    )
    print(
        "=" * 70
    )

    for key in metrics_mw.keys():

        values = results_df[
            f"MW_{key}"
        ].astype(float)

        print(
            f"{key:28s} "
            f"{values.mean():.4f} "
            f"+- "
            f"{values.std():.4f}"
        )

    print(
        "\nSaved:"
    )
    print(
        "  informer_seed_results.csv"
    )

    for seed in SEEDS:
        print(
            f"  informer_loss_history_seed_{seed}.csv"
        )

    print("\nDone.")
