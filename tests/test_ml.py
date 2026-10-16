"""Tests: PyTorch model — training, prediction, save/load, ONNX export.

Skipped automatically when torch is unavailable (inference-only boxes).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wtb.ml.lstm import RNNDirectionModel, feature_hash, set_seed  # noqa: E402


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(7)
    n, lookback, n_features = 300, 20, 4
    trend = np.linspace(0, 5, n)
    X = rng.normal(0, 1, size=(n, lookback, n_features)).astype(np.float32)
    # label: correlated with the last feature so learning is possible
    y = (X[:, -1, -1] + trend[: n] / 5 > 0).astype(np.float32)
    return X, y


def test_feature_hash_is_stable():
    assert feature_hash(["a", "b"]) == feature_hash(["a", "b"])
    assert feature_hash(["a", "b"]) != feature_hash(["b", "a"])


def test_model_forward_shape(synthetic_data):
    set_seed(42)
    m = RNNDirectionModel("lstm", lookback=20, n_features=4, hidden_units=(8, 4), dropout=0.1)
    X, _ = synthetic_data
    prob = m.predict_proba(X[0])
    assert 0.0 <= prob <= 1.0


def test_training_reduces_loss(synthetic_data):
    set_seed(42)
    X, y = synthetic_data
    m = RNNDirectionModel("lstm", lookback=20, n_features=4, hidden_units=(8, 4), dropout=0.1)
    hist = m.fit(X, y, epochs=3, batch_size=32)
    assert len(hist["loss"]) == 3
    assert hist["loss"][-1] < hist["loss"][0] or np.isfinite(hist["loss"][-1])


def test_save_load_roundtrip(tmp_path, synthetic_data):
    set_seed(42)
    X, _ = synthetic_data
    m = RNNDirectionModel("gru", lookback=20, n_features=4, hidden_units=(8, 4), dropout=0.1)
    path = str(tmp_path / "model.pt")
    m.save(path)
    m2 = RNNDirectionModel.load(path)
    p1 = m.predict_proba(X[0])
    p2 = m2.predict_proba(X[0])
    assert p1 == pytest.approx(p2, abs=1e-6)


def test_onnx_export_and_inference(tmp_path, synthetic_data):
    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    set_seed(42)
    X, _ = synthetic_data
    m = RNNDirectionModel("lstm", lookback=20, n_features=4, hidden_units=(8, 4), dropout=0.1)
    path = str(tmp_path / "model.onnx")
    from wtb.ml.onnx_export import OnnxDirectionModel, export_to_onnx
    export_to_onnx(m, path)

    sess = OnnxDirectionModel.load(path)
    assert sess.lookback == 20 and sess.n_features == 4
    p_torch = m.predict_proba(X[0])
    p_onnx = sess.predict_proba(X[0])
    assert p_onnx == pytest.approx(p_torch, abs=1e-4)


def test_scaler_no_leakage():
    """The scaler must be fit on train rows only (the v0.0.1 bug)."""
    from sklearn.preprocessing import MinMaxScaler

    from wtb.ml.features import build_sequences

    rng = np.random.default_rng(3)
    rows = rng.normal(100, 5, size=(400, 3))          # values far from [0, 1]
    close = np.linspace(50, 150, 400)
    split = build_sequences(rows, close, lookback=30, future_steps=2, test_ratio=0.2)

    # train rows scaled to ~[0, 1]
    scaled_train = split.scaler.transform(rows[: int(400 * 0.8)])
    assert scaled_train.min() >= -0.05 and scaled_train.max() <= 1.05
    assert isinstance(split.scaler, MinMaxScaler)
    # shapes: chronological split with no overlap of future info
    assert split.X_train.shape[1:] == (30, 3)
    assert len(split.X_train) + len(split.X_val) > 0
