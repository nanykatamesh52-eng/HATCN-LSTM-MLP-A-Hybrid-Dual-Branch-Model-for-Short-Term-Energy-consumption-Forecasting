"""
Rebuild of HATCN-LSTM-MLP from the manuscript (Table 1 + Table 2),
with a non-crossing quantile head.

Needs:
    torch
    numpy
    pandas
    prob_tools.py

DATA:
    PJME_hourly.csv
        Kaggle "Hourly Energy Consumption"
        Required columns:
            Datetime
            PJME_MW

    weather.csv
        OPTIONAL
        Required columns if available:
            Datetime
            temp
            humidity

If weather.csv is missing, the model runs with calendar features only.

USAGE:
    python hatcn_rebuild.py

The script:
    - Uses 3 random seeds
    - Fits Min-Max normalization on training data only
    - Uses 168 hours of historical load
    - Forecasts the next 24 hours
    - Uses future-known calendar features
    - Uses latest observed weather when weather.csv is available
    - Uses a non-crossing quantile head
    - Saves training/validation loss histories
    - Reports normalized metrics
    - Reports original MW-scale metrics
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pandas.tseries.holiday import USFederalHolidayCalendar

from prob_tools import (
    NonCrossingHead,
    pinball_loss,
    evaluate_prob
)


# ==============================================================
# Configuration
# ==============================================================

LOOKBACK = 168
HORIZON = 24

DEV = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEV}")


# ==============================================================
# DATA
# ==============================================================

def build_features(
    path_load="PJME_hourly.csv",
    path_weather="weather.csv"
):
    """
    Load PJME energy data and optionally weather data.

    Load data:
        Datetime
        PJME_MW

    Weather data:
        Datetime
        temp
        humidity

    Calendar features:
        h_sin
        h_cos
        d_sin
        d_cos
        m_sin
        m_cos
        weekend
        holiday
    """

    # ----------------------------------------------------------
    # Load energy data
    # ----------------------------------------------------------

    df = pd.read_csv(
        path_load,
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

    # Rename load column
    df.columns = ["load"]

    # ----------------------------------------------------------
    # Optional weather data
    # ----------------------------------------------------------

    if os.path.exists(path_weather):

        print(f"Weather file found: {path_weather}")

        w = pd.read_csv(
            path_weather,
            parse_dates=["Datetime"]
        )

        w = (
            w
            .drop_duplicates("Datetime")
            .set_index("Datetime")
            .sort_index()
            .asfreq("h")
        )

        # Keep only expected columns
        w = w[["temp", "humidity"]]

        # Join with energy data
        df = df.join(w)

        # Interpolate missing weather values
        df = df.interpolate().bfill().ffill()

    else:

        print(
            "WARNING: weather.csv not found. "
            "The model will use calendar features only."
        )

    # ----------------------------------------------------------
    # Calendar features
    # ----------------------------------------------------------

    idx = df.index

    holiday_dates = (
        USFederalHolidayCalendar()
        .holidays(
            idx.min(),
            idx.max()
        )
    )

    cal = pd.DataFrame(
        {
            "h_sin": np.sin(
                2 * np.pi * idx.hour / 24
            ),

            "h_cos": np.cos(
                2 * np.pi * idx.hour / 24
            ),

            "d_sin": np.sin(
                2 * np.pi * idx.dayofweek / 7
            ),

            "d_cos": np.cos(
                2 * np.pi * idx.dayofweek / 7
            ),

            "m_sin": np.sin(
                2 * np.pi * idx.month / 12
            ),

            "m_cos": np.cos(
                2 * np.pi * idx.month / 12
            ),

            "weekend": (
                idx.dayofweek >= 5
            ).astype(float),

            "holiday": (
                idx.normalize().isin(holiday_dates)
            ).astype(float),
        },
        index=idx
    )

    return df, cal


# ==============================================================
# WINDOW CREATION
# ==============================================================

def make_windows(
    df,
    cal,
    split=(0.70, 0.15, 0.15)
):
    """
    Create chronological train/validation/test windows.

    IMPORTANT:
        Min-Max normalization parameters are fitted on
        TRAINING DATA ONLY.

    This prevents information leakage from validation/test data.

    Each sample contains:

        X:
            Previous 168 hours of normalized load.

        E:
            Future calendar features for the next 24 hours
            + latest observed weather.

        Y:
            Next 24 hours of normalized load.

    At the beginning of validation/test, historical context
    can come from the immediately preceding split.
    """

    n = len(df)

    n_tr = int(n * split[0])
    n_va = int(n * split[1])

    cuts = [
        0,
        n_tr,
        n_tr + n_va,
        n
    ]

    # ----------------------------------------------------------
    # TRAIN-ONLY MIN-MAX NORMALIZATION
    # ----------------------------------------------------------

    lo = df.iloc[:n_tr].min()
    hi = df.iloc[:n_tr].max()

    dfn = (
        df - lo
    ) / (
        hi - lo + 1e-9
    )

    # ----------------------------------------------------------
    # Normalized load
    # ----------------------------------------------------------

    load = (
        dfn["load"]
        .values
        .astype(np.float32)
    )

    # ----------------------------------------------------------
    # Weather
    # ----------------------------------------------------------

    weather_columns = [
        c for c in dfn.columns
        if c != "load"
    ]

    if weather_columns:

        weather = (
            dfn[weather_columns]
            .values
            .astype(np.float32)
        )

    else:

        weather = np.empty(
            (len(df), 0),
            dtype=np.float32
        )

    # ----------------------------------------------------------
    # Calendar
    # ----------------------------------------------------------

    calv = cal.values.astype(
        np.float32
    )

    out = []

    # ==========================================================
    # Create windows for train / validation / test
    # ==========================================================

    for split_id, (a, b) in enumerate(
        zip(
            cuts[:-1],
            cuts[1:]
        )
    ):

        X = []
        E = []
        Y = []

        # ------------------------------------------------------
        # IMPORTANT:
        #
        # max(a, LOOKBACK)
        #
        # allows validation/test samples at the beginning
        # of each split to use historical observations from
        # the previous split.
        #
        # No future validation/test targets are used.
        # ------------------------------------------------------

        start_t = max(
            a,
            LOOKBACK
        )

        end_t = b - HORIZON + 1

        for t in range(
            start_t,
            end_t
        ):

            # --------------------------------------------------
            # Historical load
            # --------------------------------------------------

            X.append(
                load[
                    t - LOOKBACK:t
                ]
            )

            # --------------------------------------------------
            # Future-known calendar
            # --------------------------------------------------

            fut_cal = (
                calv[
                    t:t + HORIZON
                ]
                .reshape(-1)
            )

            # --------------------------------------------------
            # Latest observed weather
            #
            # This is NOT future weather.
            #
            # It represents the latest weather observation
            # available at the forecast origin.
            # --------------------------------------------------

            if weather.shape[1]:

                w_now = weather[
                    t - 1
                ]

            else:

                w_now = np.zeros(
                    0,
                    dtype=np.float32
                )

            # --------------------------------------------------
            # MLP branch input
            # --------------------------------------------------

            E.append(
                np.concatenate(
                    [
                        fut_cal,
                        w_now
                    ]
                )
            )

            # --------------------------------------------------
            # Target
            # --------------------------------------------------

            Y.append(
                load[
                    t:t + HORIZON
                ]
            )

        # ------------------------------------------------------
        # Convert to tensors
        # ------------------------------------------------------

        X_tensor = torch.tensor(
            np.array(X),
            dtype=torch.float32
        )

        E_tensor = torch.tensor(
            np.array(E),
            dtype=torch.float32
        )

        Y_tensor = torch.tensor(
            np.array(Y),
            dtype=torch.float32
        )

        out.append(
            (
                X_tensor,
                E_tensor,
                Y_tensor
            )
        )

        print(
            f"Split {split_id}: "
            f"X={X_tensor.shape}, "
            f"E={E_tensor.shape}, "
            f"Y={Y_tensor.shape}"
        )

    # ----------------------------------------------------------
    # Return training normalization parameters
    # ----------------------------------------------------------

    load_lo = float(
        lo["load"]
    )

    load_hi = float(
        hi["load"]
    )

    return (
        out,
        load_lo,
        load_hi
    )


# ==============================================================
# TCN BLOCK
# ==============================================================

class TCNBlock(nn.Module):

    def __init__(
        self,
        cin,
        cout,
        k,
        d,
        p=0.2
    ):

        super().__init__()

        # Left padding only
        # => causal convolution

        self.pad = (
            k - 1
        ) * d

        self.conv = nn.Conv1d(
            cin,
            cout,
            k,
            dilation=d
        )

        self.drop = nn.Dropout(
            p
        )

        self.res = (
            nn.Conv1d(
                cin,
                cout,
                1
            )
            if cin != cout
            else nn.Identity()
        )

    def forward(self, x):

        # x:
        # (B, C, T)

        y = self.conv(
            F.pad(
                x,
                (
                    self.pad,
                    0
                )
            )
        )

        y = F.relu(y)

        y = self.drop(y)

        return (
            y
            + self.res(x)
        )


# ==============================================================
# HATCN-LSTM-MLP
# ==============================================================

class HATCN_LSTM_MLP(nn.Module):

    def __init__(
        self,
        exog_dim,
        ch=64,
        lstm=128
    ):

        super().__init__()

        # ------------------------------------------------------
        # TCN
        # ------------------------------------------------------

        self.tcn = nn.Sequential(

            TCNBlock(
                1,
                ch,
                3,
                1
            ),

            TCNBlock(
                ch,
                ch,
                3,
                2
            ),

            TCNBlock(
                ch,
                ch,
                3,
                4
            )
        )

        # ------------------------------------------------------
        # LSTM
        # ------------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=ch,
            hidden_size=lstm,
            batch_first=True
        )

        # ------------------------------------------------------
        # Attention
        # ------------------------------------------------------

        self.att_W = nn.Linear(
            lstm,
            lstm
        )

        self.att_v = nn.Linear(
            lstm,
            1,
            bias=False
        )

        # ------------------------------------------------------
        # MLP exogenous branch
        # ------------------------------------------------------

        self.mlp = nn.Sequential(

            nn.Linear(
                exog_dim,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU()
        )

        # ------------------------------------------------------
        # Fusion
        # ------------------------------------------------------

        self.fuse = nn.Sequential(

            nn.Linear(
                lstm + 32,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            )
        )

        # ------------------------------------------------------
        # Non-crossing probabilistic head
        # ------------------------------------------------------

        self.head = NonCrossingHead(
            128,
            HORIZON
        )

    def forward(
        self,
        x,
        e
    ):

        # ------------------------------------------------------
        # x:
        # (B, 168)
        #
        # e:
        # (B, exog_dim)
        # ------------------------------------------------------

        # ------------------------------------------------------
        # TCN
        # ------------------------------------------------------

        h = self.tcn(
            x.unsqueeze(1)
        )

        # (B, 64, 168)
        #
        # -> (B, 168, 64)

        h = h.transpose(
            1,
            2
        )

        # ------------------------------------------------------
        # LSTM
        # ------------------------------------------------------

        h, _ = self.lstm(h)

        # (B, 168, 128)

        # ------------------------------------------------------
        # Attention
        # ------------------------------------------------------

        a = self.att_v(
            torch.tanh(
                self.att_W(h)
            )
        )

        a = torch.softmax(
            a,
            dim=1
        )

        # ------------------------------------------------------
        # Context vector
        # ------------------------------------------------------

        Hc = (
            a * h
        ).sum(
            dim=1
        )

        # ------------------------------------------------------
        # Exogenous branch
        # ------------------------------------------------------

        exog = self.mlp(e)

        # ------------------------------------------------------
        # Fusion
        # ------------------------------------------------------

        z = torch.cat(
            [
                Hc,
                exog
            ],
            dim=1
        )

        z = self.fuse(z)

        # ------------------------------------------------------
        # Quantile output
        # ------------------------------------------------------

        return self.head(z)


# ==============================================================
# MAPE
# ==============================================================

def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    """
    Mean Absolute Percentage Error (MAPE), returned as a percentage.

    Computed on the original MW scale using the median (q50)
    forecast, with epsilon protection against division by zero.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    denominator = np.maximum(np.abs(y_true), epsilon)

    return float(
        np.mean(
            np.abs((y_true - y_pred) / denominator)
        ) * 100.0
    )


# ==============================================================
# PREDICTION
# ==============================================================

def predict(
    model,
    X,
    E,
    bs=512
):

    model.eval()

    out = []

    with torch.no_grad():

        for i in range(
            0,
            len(X),
            bs
        ):

            xb = X[
                i:i + bs
            ].to(DEV)

            eb = E[
                i:i + bs
            ].to(DEV)

            pred = model(
                xb,
                eb
            )

            out.append(
                pred.cpu()
            )

    return torch.cat(
        out
    ).numpy()


# ==============================================================
# TRAINING
# ==============================================================

def run(
    seed,
    data,
    load_lo,
    load_hi,
    epochs=60,
    patience=10,
    bs=64
):

    # ----------------------------------------------------------
    # Reproducibility
    # ----------------------------------------------------------

    torch.manual_seed(
        seed
    )

    np.random.seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    # ----------------------------------------------------------
    # Data
    # ----------------------------------------------------------

    (
        Xtr,
        Etr,
        Ytr
    ), (
        Xva,
        Eva,
        Yva
    ), (
        Xte,
        Ete,
        Yte
    ) = data

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    model = HATCN_LSTM_MLP(
        Etr.shape[1]
    ).to(DEV)

    # ----------------------------------------------------------
    # Optimizer
    # ----------------------------------------------------------

    opt = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    # ----------------------------------------------------------
    # Early stopping
    # ----------------------------------------------------------

    best = float("inf")

    bad = 0

    state = None

    # ----------------------------------------------------------
    # Training history
    # ----------------------------------------------------------

    history = []

    # ==========================================================
    # Epoch loop
    # ==========================================================

    for ep in range(
        epochs
    ):

        model.train()

        perm = torch.randperm(
            len(Xtr)
        )

        train_losses = []

        # ------------------------------------------------------
        # Mini-batches
        # ------------------------------------------------------

        for i in range(
            0,
            len(perm),
            bs
        ):

            j = perm[
                i:i + bs
            ]

            xb = Xtr[
                j
            ].to(DEV)

            eb = Etr[
                j
            ].to(DEV)

            yb = Ytr[
                j
            ].to(DEV)

            # --------------------------------------------------
            # Forward
            # --------------------------------------------------

            pred = model(
                xb,
                eb
            )

            # --------------------------------------------------
            # Pinball loss
            # --------------------------------------------------

            loss = pinball_loss(
                pred,
                yb
            )

            # --------------------------------------------------
            # Backpropagation
            # --------------------------------------------------

            opt.zero_grad()

            loss.backward()

            opt.step()

            train_losses.append(
                loss.item()
            )

        # ------------------------------------------------------
        # Training loss
        # ------------------------------------------------------

        train_loss = float(
            np.mean(
                train_losses
            )
        )

        # ------------------------------------------------------
        # Validation prediction
        # ------------------------------------------------------

        pred_val = predict(
            model,
            Xva,
            Eva
        )

        # ------------------------------------------------------
        # Validation pinball loss
        # ------------------------------------------------------

        val_loss = pinball_loss(
            torch.tensor(
                pred_val,
                dtype=torch.float32
            ),
            Yva
        ).item()

        # ------------------------------------------------------
        # Save loss history
        # ------------------------------------------------------

        history.append(
            {
                "seed": seed,
                "epoch": ep + 1,
                "train_pinball_loss": train_loss,
                "val_pinball_loss": float(
                    val_loss
                )
            }
        )

        # ------------------------------------------------------
        # Print
        # ------------------------------------------------------

        print(
            f"seed {seed} "
            f"ep {ep + 1:02d} "
            f"train {train_loss:.5f} "
            f"val {val_loss:.5f}"
        )

        # ------------------------------------------------------
        # Early stopping
        # ------------------------------------------------------

        if val_loss < best:

            best = val_loss

            bad = 0

            state = {
                k: v.detach().clone()
                for k, v
                in model.state_dict().items()
            }

        else:

            bad += 1

            if bad >= patience:

                print(
                    f"Early stopping at epoch "
                    f"{ep + 1}"
                )

                break

    # ==========================================================
    # Restore best model
    # ==========================================================

    if state is None:

        raise RuntimeError(
            "No valid model state was saved."
        )

    model.load_state_dict(
        state
    )

    # ==========================================================
    # TEST PREDICTIONS
    # ==========================================================

    pred_norm = predict(
        model,
        Xte,
        Ete
    )

    y_norm = Yte.numpy()
    np.savez(
        f"test_pred_hatcn_seed{seed}.npz",
        y_norm=y_norm,            # القيم الفعلية (N, 24) مقيسة
        pred_norm=pred_norm,      # التنبؤات (N, 24, 3) = q0.1, q0.5, q0.9
        lo=load_lo, hi=load_hi    # عشان أرجّعها لـ MW
    )

    # ==========================================================
    # NORMALIZED METRICS
    # ==========================================================

    metrics_norm = evaluate_prob(
        y_norm,
        pred_norm,
        y_train_range=1.0
    )

    # ==========================================================
    # INVERSE TRANSFORMATION
    # ==========================================================

    load_range = (
        load_hi
        - load_lo
    )

    if load_range <= 0:

        raise ValueError(
            "Invalid training load range."
        )

    y_mw = (
        y_norm
        * load_range
        + load_lo
    )

    pred_mw = (
        pred_norm
        * load_range
        + load_lo
    )

    # ==========================================================
    # ORIGINAL MW METRICS
    # ==========================================================

    metrics_mw = evaluate_prob(
        y_mw,
        pred_mw,
        y_train_range=load_range
    )

    # --------------------------------------------------------------
    # MAPE on original MW scale
    # --------------------------------------------------------------
    # Use q50 (median) as the point forecast.
    mape_mw = mean_absolute_percentage_error(
        y_mw,
        pred_mw[:, :, 1]
    )

    metrics_mw["MAPE"] = mape_mw

    # ==========================================================
    # Return
    # ==========================================================

    return (
        metrics_norm,
        metrics_mw,
        pd.DataFrame(history)
    )


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------

    df, cal = build_features()

    print("\nData shape:")
    print(df.shape)

    print("\nCalendar features:")
    print(cal.shape)

    # ----------------------------------------------------------
    # Create windows
    # ----------------------------------------------------------

    data, load_lo, load_hi = make_windows(
        df,
        cal
    )

    print("\nTraining load range:")
    print(
        f"min = {load_lo:.4f}"
    )

    print(
        f"max = {load_hi:.4f}"
    )

    # ----------------------------------------------------------
    # Results container
    # ----------------------------------------------------------

    results = []

    # ==========================================================
    # Run 3 seeds
    # ==========================================================

    for s in (
        0,
        1,
        2
    ):

        print("\n")
        print("=" * 70)
        print(
            f"STARTING SEED {s}"
        )
        print("=" * 70)

        (
            metrics_norm,
            metrics_mw,
            history
        ) = run(
            seed=s,
            data=data,
            load_lo=load_lo,
            load_hi=load_hi
        )

        # ------------------------------------------------------
        # Save loss history
        # ------------------------------------------------------

        history_file = (
            f"loss_history_seed_{s}.csv"
        )

        history.to_csv(
            history_file,
            index=False
        )

        print(
            f"\nSaved loss history: "
            f"{history_file}"
        )

        # ------------------------------------------------------
        # Store results
        # ------------------------------------------------------

        results.append(
            {
                "seed": s,
                "normalized": metrics_norm,
                "MW": metrics_mw
            }
        )

    # ==========================================================
    # FINAL NORMALIZED RESULTS
    # ==========================================================

    print("\n")
    print("=" * 70)
    print(
        "FINAL NORMALIZED RESULTS"
    )
    print("=" * 70)

    for k in results[0]["normalized"]:

        values = np.array(
            [
                r["normalized"][k]
                for r in results
            ],
            dtype=float
        )

        print(
            f"{k:28s} "
            f"{values.mean():.4f} "
            f"+- "
            f"{values.std():.4f}"
        )

    # ==========================================================
    # FINAL ORIGINAL MW RESULTS
    # ==========================================================

    print("\n")
    print("=" * 70)
    print(
        "FINAL ORIGINAL MW RESULTS"
    )
    print("=" * 70)
    print("MAPE is reported as a percentage using the q50 (median) forecast.")

    for k in results[0]["MW"]:

        values = np.array(
            [
                r["MW"][k]
                for r in results
            ],
            dtype=float
        )

        print(
            f"{k:28s} "
            f"{values.mean():.4f} "
            f"+- "
            f"{values.std():.4f}"
        )

    # ==========================================================
    # FINAL SUMMARY
    # ==========================================================

    print("\n")
    print("=" * 70)
    print(
        "RUN COMPLETED"
    )
    print("=" * 70)

    print(
        "\nGenerated files:"
    )

    for s in (
        0,
        1,
        2
    ):

        print(
            f"  loss_history_seed_{s}.csv"
        )

    print(
        "\nDone."
    )