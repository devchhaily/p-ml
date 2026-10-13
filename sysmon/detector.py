"""Inference: score a full series offline, or one sample at a time online."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import Scaler
from .features import FEATURES
from .model import LSTMAutoencoder


@dataclass
class Verdict:
    score: float
    threshold: float
    is_anomaly: bool       # this window is above threshold
    alert: bool            # debounced: above threshold for `min_consecutive` windows in a row
    top_features: list[tuple[str, float]]  # biggest contributors to the error


class AnomalyDetector:
    def __init__(self, checkpoint: str | Path, threshold: float | None = None, min_consecutive: int = 3,
                 device: str = "cpu") -> None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        if ckpt["features"] != FEATURES:
            raise ValueError("checkpoint was trained on a different feature set")
        self.model = LSTMAutoencoder(**ckpt["model_config"]).to(device).eval()
        self.model.load_state_dict(ckpt["model_state"])
        self.scaler = Scaler.from_state_dict(ckpt["scaler"])
        self.window: int = ckpt["window"]
        self.threshold: float = threshold if threshold is not None else ckpt["threshold"]
        self.min_consecutive = min_consecutive
        self.device = device
        self._buf: deque[np.ndarray] = deque(maxlen=self.window)
        self._streak = 0

    def _errors(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(windows).to(self.device)
        return self.model.feature_errors(x).cpu().numpy()

    def score_series(self, values: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Score every timestep of a raw [T, F] series. The first window-1 steps get NaN."""
        x = self.scaler.transform(values)
        windows = np.lib.stride_tricks.sliding_window_view(x, self.window, axis=0).transpose(0, 2, 1)
        scores = np.concatenate([
            self._errors(np.ascontiguousarray(windows[i : i + batch_size])).mean(axis=1)
            for i in range(0, len(windows), batch_size)
        ])
        return np.concatenate([np.full(self.window - 1, np.nan), scores])

    def update(self, sample: dict[str, float]) -> Verdict | None:
        """Feed one live sample. Returns None until the window is full."""
        self._buf.append(np.array([sample[f] for f in FEATURES], dtype=np.float32))
        if len(self._buf) < self.window:
            return None
        window = self.scaler.transform(np.stack(self._buf))[None]
        errs = self._errors(window)[0]
        score = float(errs.mean())
        is_anom = score > self.threshold
        self._streak = self._streak + 1 if is_anom else 0
        order = np.argsort(errs)[::-1][:3]
        share = errs / max(errs.sum(), 1e-12)
        return Verdict(
            score=score,
            threshold=self.threshold,
            is_anomaly=is_anom,
            alert=self._streak >= self.min_consecutive,
            top_features=[(FEATURES[i], float(share[i])) for i in order],
        )


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float, window: int) -> dict[str, float]:
    """Precision / recall / F1 over windows, ignoring warm-up steps without a score.

    The score at step t describes the window ending at t, so that window counts
    as anomalous if *any* step inside it is labelled anomalous. Otherwise the
    window-1 steps after every incident would be counted as false positives.
    """
    window_labels = np.convolve(labels, np.ones(window, dtype=np.int64))[: len(labels)] > 0
    mask = ~np.isnan(scores)
    pred = scores[mask] > threshold
    true = window_labels[mask]
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
