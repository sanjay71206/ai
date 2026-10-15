"""LSTM price forecasting. TensorFlow is imported lazily so the API boots without it."""
import json
import os
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from backend.services.data_service import ApiError

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved")
_LOCK = threading.Lock()
_MODELS: dict = {}


def _paths(t):
    return os.path.join(MODEL_DIR, f"{t}.keras"), os.path.join(MODEL_DIR, f"{t}.json")


def _tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        raise ApiError(503, "TensorFlow is not installed. Use Python 3.10-3.12 and run: pip install -r backend/requirements.txt")


def make_windows(s: np.ndarray, lb: int):
    X = np.stack([s[i - lb:i] for i in range(lb, len(s))])[..., None]
    return X, s[lb:]


def _metrics(actual, pred, prev):
    return {"mae": float(np.mean(np.abs(actual - pred))),
            "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
            "dir": float(np.mean(np.sign(actual - prev) == np.sign(pred - prev)) * 100)}


def train_model(ticker: str, df: pd.DataFrame, lookback: int = 20, epochs: int = 20) -> dict:
    close = df["Close"].to_numpy(float)
    n = len(close)
    if n < lookback + 100:
        raise ApiError(422, f"Not enough history: {n} rows, need at least {lookback + 100}.")
    tf = _tf()
    with _LOCK:
        tf.keras.utils.set_random_seed(42)
        tr_end, va_end = int(0.7 * n), int(0.85 * n)          # chronological 70/15/15
        sc = MinMaxScaler().fit(close[:tr_end].reshape(-1, 1))  # fit on train only: no leakage
        s = sc.transform(close.reshape(-1, 1)).ravel()
        X, y = make_windows(s, lookback)
        idx = np.arange(lookback, n)                            # index of each target
        tr, va, te = idx < tr_end, (idx >= tr_end) & (idx < va_end), idx >= va_end
        L = tf.keras.layers
        model = tf.keras.Sequential([L.Input((lookback, 1)), L.LSTM(32), L.Dropout(0.1), L.Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        h = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=epochs, batch_size=32,
                      shuffle=False, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        pred = sc.inverse_transform(model.predict(X[te], verbose=0)).ravel()
        actual, prev = close[idx[te]], close[idx[te] - 1]
        meta = {
            "ticker": ticker,
            "metrics": {"lstm": _metrics(actual, pred, prev), "baseline": _metrics(actual, prev, prev)},
            "test_dates": [d.strftime("%Y-%m-%d") for d in df.index[idx[te]]],
            "test_pred": [round(float(v), 4) for v in pred],
            "history": {"loss": [float(v) for v in h.history["loss"]], "val_loss": [float(v) for v in h.history["val_loss"]]},
            "config": {"lookback": lookback, "epochs_requested": epochs, "units": 32, "split": "70/15/15 chronological",
                       "baseline": "previous close"},
            "scaler": {"min": float(sc.data_min_[0]), "max": float(sc.data_max_[0])},
            "n_rows": n, "data_start": df.index[0].strftime("%Y-%m-%d"), "data_end": df.index[-1].strftime("%Y-%m-%d"),
            "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        os.makedirs(MODEL_DIR, exist_ok=True)
        mp, jp = _paths(ticker)
        model.save(mp)
        with open(jp, "w") as f:
            json.dump(meta, f)
        _MODELS[ticker] = model
    return meta


def read_metrics(ticker: str) -> dict:
    _, jp = _paths(ticker)
    if not os.path.exists(jp):
        raise ApiError(404, f"No evaluation for {ticker}. Train a model first.")
    with open(jp) as f:
        return json.load(f)


def forecast(ticker: str, df: pd.DataFrame, horizon: int) -> dict:
    meta = read_metrics(ticker)
    tf = _tf()
    with _LOCK:
        model = _MODELS.get(ticker) or tf.keras.models.load_model(_paths(ticker)[0])
        _MODELS[ticker] = model
        lb, lo, hi = meta["config"]["lookback"], meta["scaler"]["min"], meta["scaler"]["max"]
        span = (hi - lo) or 1.0
        w = list(((df["Close"].to_numpy(float) - lo) / span)[-lb:])
        out = []
        for _ in range(horizon):  # recursive multi-step forecast
            y = float(model(np.array(w[-lb:])[None, :, None], training=False).numpy()[0, 0])
            out.append(y)
            w.append(y)
    dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return {"ticker": ticker, "label": "Model estimate - not a guarantee",
            "last_observed": df.index[-1].strftime("%Y-%m-%d"),
            "forecast": [round(v * span + lo, 4) for v in out],
            "forecast_dates": [d.strftime("%Y-%m-%d") for d in dates]}
