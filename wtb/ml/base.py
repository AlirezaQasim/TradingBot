"""Model interface shared by training and inference."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> float:
        """Probability that the price N steps ahead is higher than now.
        ``features`` shape: (lookback, n_features)."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel": ...
