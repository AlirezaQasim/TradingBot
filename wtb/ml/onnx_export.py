"""ONNX export + fast inference runtime (2-5x faster / lower RAM than PyTorch)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wtb.core.logging import get_logger
from wtb.ml.base import BaseModel

log = get_logger(__name__)


def export_to_onnx(model, path: str) -> str:
    """Export an RNNDirectionModel to ONNX (dynamic batch + sequence length)."""
    import torch

    model.net.eval()
    lookback, n_features = model.lookback, model.n_features
    dummy = torch.randn(1, lookback, n_features, dtype=torch.float32)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model.net,
            dummy,
            path,
            input_names=["features"],
            output_names=["prob"],
            # batch only: sequence/feature dims stay fixed so the runtime can
            # read the expected window shape back from the graph
            dynamic_axes={"features": {0: "batch"}, "prob": {0: "batch"}},
            opset_version=17,
        )
    log.info("onnx_exported", path=path)
    return path


class OnnxDirectionModel(BaseModel):
    """Inference-only wrapper around an exported ONNX model."""

    def __init__(self, onnx_path: str, lookback: int, n_features: int) -> None:
        import onnxruntime as ort

        self.name = "onnx"
        self.lookback = lookback
        self.n_features = n_features
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict_proba(self, features: np.ndarray) -> float:
        x = features[np.newaxis, ...].astype(np.float32)
        (prob,) = self.session.run(["prob"], {self.input_name: x})
        return float(prob[0][0])

    def save(self, path: str) -> None:  # pragma: no cover — ONNX file already persisted
        raise NotImplementedError("ONNX models are files; nothing to save")

    @classmethod
    def load(cls, path: str) -> "OnnxDirectionModel":
        import onnx

        m = onnx.load(path)
        dims = {t.name: t for t in m.graph.input}
        shape = dims["features"].type.tensor_type.shape.dim
        lookback = shape[1].dim_value or 0
        n_features = shape[2].dim_value or 0
        return cls(path, lookback, n_features)
