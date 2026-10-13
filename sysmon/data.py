"""Loading, scaling and windowing of metric time series."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import FEATURES, LOG_FEATURES


def load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (values[T, F], labels[T] or None). A ``label`` column is optional."""
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    missing = [c for c in FEATURES if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    values = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float32)
    labels = np.array([int(r["label"]) for r in rows]) if "label" in rows[0] else None
    return values, labels


@dataclass
class Scaler:
    """log1p on throughput columns, then per-feature z-score."""

    mean: np.ndarray
    std: np.ndarray

    _log_mask = np.array([f in LOG_FEATURES for f in FEATURES])

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        x = cls._pre(x)
        std = x.std(axis=0)
        return cls(mean=x.mean(axis=0), std=np.where(std < 1e-6, 1.0, std))

    @classmethod
    def _pre(cls, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=True)
        x[..., cls._log_mask] = np.log1p(np.clip(x[..., cls._log_mask], 0, None))
        return x

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((self._pre(x) - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "Scaler":
        return cls(mean=np.array(d["mean"], dtype=np.float32), std=np.array(d["std"], dtype=np.float32))


class WindowDataset(Dataset):
    """Sliding windows of length ``window`` over a scaled [T, F] array."""

    def __init__(self, x: np.ndarray, window: int, stride: int = 1) -> None:
        if len(x) < window:
            raise ValueError(f"series length {len(x)} is shorter than window {window}")
        self.x = torch.from_numpy(x)
        self.window = window
        self.starts = range(0, len(x) - window + 1, stride)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> torch.Tensor:
        s = self.starts[i]
        return self.x[s : s + self.window]
