"""PyTorch LSTM/GRU direction classifier.

Port of the v0.0.1 Keras architecture (128 -> 64 recurrent layers + dense head,
sigmoid output) with the bugs fixed:
- the feature scaler is fit on the TRAIN split only (v0.0.1 fit_transformed the
  whole dataset before splitting = look-ahead leakage);
- model artifacts are keyed by symbol+timeframe+feature-hash so stale models
  can't be loaded after a config change;
- deterministic seeding for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:            # inference-only boxes may skip torch (ONNX instead)
    torch = None

from wtb.ml.base import BaseModel


def set_seed(seed: int) -> None:
    if torch is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_hash(features: list[str]) -> str:
    return hashlib.sha1(json.dumps(features).encode()).hexdigest()[:8]


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is not installed — pip install torch")


def _make_layer(cls, in_features: int, hidden: int, dropout: float, last: bool):
    """Factory: the class body needs torch at definition time, so build locally."""
    class _RNNLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = cls(in_features, hidden, batch_first=True)
            self.drop = torch.nn.Dropout(dropout)
            self.last = last

        def forward(self, x):
            out, _ = self.rnn(x)
            out = out[:, -1, :] if self.last else out
            return self.drop(out)

    return _RNNLayer()


class RNNDirectionModel(BaseModel):
    def __init__(self, model_type: str, lookback: int, n_features: int,
                 hidden_units: tuple[int, int] = (128, 64), dropout: float = 0.3,
                 lr: float = 1e-3) -> None:
        _require_torch()
        self.name = model_type
        self.lookback = lookback
        self.n_features = n_features
        self.lr = lr
        rnn_cls = torch.nn.LSTM if model_type == "lstm" else torch.nn.GRU
        layer1 = _make_layer(rnn_cls, n_features, hidden_units[0], dropout, last=False)
        layer2 = _make_layer(rnn_cls, hidden_units[0], hidden_units[1], dropout, last=True)
        self.net = torch.nn.Sequential(
            layer1, layer2,
            torch.nn.Linear(hidden_units[1], 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid(),
        )
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    # ------------------------------------------------------------- training #
    def fit(
        self,
        X: np.ndarray,           # (samples, lookback, features) — already scaled
        y: np.ndarray,           # (samples,) 0/1
        epochs: int = 20,
        batch_size: int = 32,
        verbose: bool = False,
        seed: int = 42,
    ) -> dict:
        _require_torch()
        set_seed(seed)
        self.net.train()
        n = len(X)
        history = {"loss": []}
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss, batches = 0.0, 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = torch.tensor(X[idx], dtype=torch.float32)
                yb = torch.tensor(y[idx], dtype=torch.float32).unsqueeze(1)
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(xb), yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.detach())
                batches += 1
            history["loss"].append(epoch_loss / max(1, batches))
            if verbose:
                print(f"epoch {epoch + 1}/{epochs} loss={history['loss'][-1]:.4f}")
        return history

    def predict_proba(self, features: np.ndarray) -> float:
        _require_torch()
        x = torch.tensor(features[np.newaxis, ...], dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            return float(self.net(x).item())

    # ----------------------------------------------------------- persistence #
    def save(self, path: str) -> None:
        _require_torch()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "meta": {
                "name": self.name, "lookback": self.lookback,
                "n_features": self.n_features,
                "hidden_units": [self.net[0].rnn.hidden_size, self.net[1].rnn.hidden_size],
                "dropout": self.net[0].drop.p,
            },
        }, path)

    @classmethod
    def load(cls, path: str) -> "RNNDirectionModel":
        _require_torch()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        meta = payload["meta"]
        model = cls(
            meta["name"], meta["lookback"], meta["n_features"],
            tuple(meta["hidden_units"]), meta["dropout"],
        )
        model.net.load_state_dict(payload["state_dict"])
        model.net.eval()
        return model
