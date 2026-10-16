"""Model service: training, incremental learning, prediction, artifact registry.

Artifacts are keyed by symbol+timeframe+feature-hash, so changing the config
can never load a stale incompatible model (a v0.0.1 bug).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from wtb.core.config import AIConfig
from wtb.core.logging import get_logger
from wtb.data.collector import Collector
from wtb.ml.base import BaseModel
from wtb.ml.features import TrainSplit, build_sequences, load_scaler, save_scaler
from wtb.ml.lstm import RNNDirectionModel, feature_hash

log = get_logger(__name__)


class ModelService:
    def __init__(self, cfg: AIConfig, symbol: str, timeframe: str, model_dir: Path) -> None:
        self.cfg = cfg
        self.symbol = symbol.upper()
        self.timeframe = timeframe.lower()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.key = f"{self.symbol}_{self.timeframe}_{cfg.model_type}_{feature_hash(cfg.features)}"
        self.model_path = self.model_dir / f"{self.key}.pt"
        self.onnx_path = self.model_dir / f"{self.key}.onnx"
        self.scaler_path = self.model_dir / f"{self.key}_scaler.pkl"
        self.meta_path = self.model_dir / f"{self.key}_meta.json"

        self.model: Optional[BaseModel] = None
        self._scaler = None
        self.n_features = len(cfg.features)
        self._training_count = 0

    # ------------------------------------------------------------------ load #
    def load_or_create(self) -> bool:
        """Load persisted artifacts. Returns True when a usable model is present."""
        try:
            if self.onnx_path.exists() and self.cfg.prefer_onnx:
                from wtb.ml.onnx_export import OnnxDirectionModel
                self.model = OnnxDirectionModel.load(str(self.onnx_path))
            elif self.model_path.exists():
                self.model = RNNDirectionModel.load(str(self.model_path))
            else:
                return False
            self._scaler = load_scaler(str(self.scaler_path))
            log.info("model_loaded", artifact=self.model_path.name)
            return True
        except Exception as exc:
            log.warning("model_load_failed", error=str(exc))
            self.model = None
            self._scaler = None
            return False

    # ----------------------------------------------------------------- train #
    def train(self, collector: Collector, epochs: int | None = None) -> dict:
        """Full training on the collector's data (chronological, leakage-free)."""
        import polars as pl

        df = collector.feature_frame(self.cfg.features)
        df = df.drop_nulls()
        if df.height < self.cfg.min_training_samples:
            raise ValueError(
                f"insufficient training rows: {df.height} < {self.cfg.min_training_samples}"
            )

        values = df.select(self.cfg.features).to_numpy()
        close = df["close"].to_numpy()
        split: TrainSplit = build_sequences(
            values, close, self.cfg.lookback, self.cfg.future_steps,
            test_ratio=self.cfg.test_ratio,
        )

        model = RNNDirectionModel(
            self.cfg.model_type, self.cfg.lookback, self.n_features,
            tuple(self.cfg.hidden_units), self.cfg.dropout, self.cfg.lr,
        )
        history = model.fit(
            split.X_train, split.y_train,
            epochs=epochs or self.cfg.epochs, batch_size=self.cfg.batch_size,
            seed=self.cfg.seed,
        )

        # validation accuracy
        acc = None
        if len(split.X_val) > 0:
            preds = np.array([
                model.predict_proba(x) > 0.5 for x in split.X_val
            ])
            acc = float((preds == split.y_val).mean())

        model.save(str(self.model_path))
        save_scaler(split.scaler, str(self.scaler_path))
        self._scaler = split.scaler
        self.model = model
        self._training_count += 1

        self._write_meta(history, acc)
        self._maybe_export_onnx(model)
        self._log_mlflow(history, acc)

        log.info("model_trained", artifact=self.key, epochs=len(history["loss"]),
                 val_acc=acc, train_rows=len(split.X_train))
        return {"loss": history["loss"][-1], "val_acc": acc, "rows": len(split.X_train)}

    def incremental_learn(self, collector: Collector) -> dict | None:
        """Fine-tune on the most recent window (v0.0.1 behaviour, counter fixed)."""
        if not self.cfg.online_learning or self.model is None or self._scaler is None:
            return None
        try:
            import torch

            df = collector.feature_frame(self.cfg.features).drop_nulls()
            rows = df.select(self.cfg.features).to_numpy()
            close = df["close"].to_numpy()
            if len(rows) < self.cfg.online_batch_size:
                return None
            tail = rows[-self.cfg.online_batch_size:]
            tail_close = close[-self.cfg.online_batch_size:]
            scaled = self._scaler.transform(tail)
            target = (np.roll(tail_close, -self.cfg.future_steps) > tail_close).astype(np.float32)

            X, y = [], []
            for i in range(self.cfg.lookback, len(scaled)):
                if i + self.cfg.future_steps >= len(tail_close):
                    break
                X.append(scaled[i - self.cfg.lookback:i])
                y.append(target[i - 1])
            if not X:
                return None

            model = self.model if isinstance(self.model, RNNDirectionModel) else None
            if model is None:
                return None   # ONNX models are inference-only; full retrain replaces them
            model.net.train()
            history = model.fit(
                np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32),
                epochs=self.cfg.online_epochs, batch_size=32,
            )
            model.save(str(self.model_path))
            self._training_count += 1
            self._maybe_export_onnx(model)
            log.info("incremental_learn_done", loss=history["loss"][-1])
            return {"loss": history["loss"][-1]}
        except Exception as exc:
            log.warning("incremental_learn_failed", error=str(exc))
            return None

    # -------------------------------------------------------------- predict #
    def predict(self, raw_features: np.ndarray) -> Optional[float]:
        """Probability(price goes up). ``raw_features``: (lookback, n_features)."""
        if self.model is None or self._scaler is None:
            return None
        if raw_features.shape[0] < self.cfg.lookback:
            return None
        window = raw_features[-self.cfg.lookback:]
        scaled = self._scaler.transform(window)
        return self.model.predict_proba(scaled.astype(np.float32))

    # -------------------------------------------------------------- helpers #
    def _maybe_export_onnx(self, model: RNNDirectionModel) -> None:
        if not self.cfg.prefer_onnx:
            return
        try:
            from wtb.ml.onnx_export import export_to_onnx
            export_to_onnx(model, str(self.onnx_path))
            from wtb.ml.onnx_export import OnnxDirectionModel
            self.model = OnnxDirectionModel.load(str(self.onnx_path))
        except Exception as exc:
            log.warning("onnx_export_failed", error=str(exc))

    def _write_meta(self, history: dict, val_acc: float | None) -> None:
        self.meta_path.write_text(json.dumps({
            "key": self.key, "symbol": self.symbol, "timeframe": self.timeframe,
            "features": self.cfg.features, "lookback": self.cfg.lookback,
            "future_steps": self.cfg.future_steps, "trainings": self._training_count,
            "last_val_acc": val_acc, "last_loss": history["loss"][-1] if history["loss"] else None,
        }, indent=2))

    def _log_mlflow(self, history: dict, val_acc: float | None) -> None:
        try:
            import mlflow

            mlflow.set_tracking_uri(self._tracking_uri())
            mlflow.set_experiment("webtraderbot")
            with mlflow.start_run(run_name=self.key):
                mlflow.log_params({
                    "model_type": self.cfg.model_type, "lookback": self.cfg.lookback,
                    "features": ",".join(self.cfg.features),
                    "future_steps": self.cfg.future_steps,
                })
                if history["loss"]:
                    mlflow.log_metric("final_loss", history["loss"][-1])
                if val_acc is not None:
                    mlflow.log_metric("val_acc", val_acc)
                mlflow.log_artifact(str(self.model_path))
        except Exception as exc:
            log.warning("mlflow_log_failed", error=str(exc))

    def _tracking_uri(self) -> str:
        from wtb.core.config import get_settings
        return get_settings().mlflow.tracking_uri
