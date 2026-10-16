"""Feature/target preparation for training and inference.

The leakage fix lives here: the scaler is fit on the train split only and the
same fitted scaler is reused for validation and inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


@dataclass
class TrainSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    scaler: MinMaxScaler


def build_sequences(
    feature_values: np.ndarray,     # (n_rows, n_features), raw (unscaled)
    close: np.ndarray,              # (n_rows,) aligned with feature_values
    lookback: int,
    future_steps: int,
    test_ratio: float = 0.2,
) -> TrainSplit:
    """Chronological 80/20 split; the scaler sees training rows ONLY."""
    if len(feature_values) < lookback + future_steps + 10:
        raise ValueError(
            f"not enough data: {len(feature_values)} rows < lookback {lookback} + "
            f"future {future_steps} + margin"
        )

    # target: close[t + future_steps] > close[t]
    target = (np.roll(close, -future_steps) > close).astype(np.float32)
    usable = len(close) - future_steps           # rows with a known future

    split_row = int(usable * (1.0 - test_ratio))
    split_row = max(split_row, lookback)         # at least one full window

    scaler = MinMaxScaler()
    scaler.fit(feature_values[:split_row])       # <-- train rows only (no leakage)
    scaled = scaler.transform(feature_values)

    def make_windows(start: int, end: int):
        X, y = [], []
        for i in range(start, end):
            if i + future_steps >= len(close):
                break
            X.append(scaled[i - lookback:i])
            y.append(target[i - 1])
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    X_train, y_train = make_windows(lookback, split_row)
    X_val, y_val = make_windows(split_row, usable)
    return TrainSplit(X_train, y_train, X_val, y_val, scaler)


def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    joblib.dump(scaler, path)


def load_scaler(path: str) -> MinMaxScaler:
    return joblib.load(path)
