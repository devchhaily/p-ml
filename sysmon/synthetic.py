"""Synthetic host metrics with labelled anomalies.

Useful to develop and sanity-check the pipeline without collecting hours of
real data first. Normal behaviour is a daily-ish cycle with correlated
metrics; anomalies are injected on top and marked with label=1.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .features import FEATURES

ANOMALY_KINDS = ("cpu_spike", "memory_leak", "disk_burst", "net_flood", "fork_bomb")


def generate(n: int, anomaly_rate: float = 0.0, period: int = 720, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cycle = 0.5 + 0.5 * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))  # 0..1 "busyness"
    noise = lambda scale: rng.normal(0, scale, n)

    cpu = 15 + 40 * cycle + noise(4)
    data = {
        "cpu_percent": cpu,
        "mem_percent": 45 + 15 * cycle + noise(1.0),
        "swap_percent": 5 + noise(0.3),
        "load_per_cpu": cpu / 100 + noise(0.04),
        "num_procs": 250 + 30 * cycle + noise(3),
        "disk_read_mbs": rng.gamma(2.0, 1.5 + 3 * cycle),
        "disk_write_mbs": rng.gamma(2.0, 1.0 + 2 * cycle),
        "net_sent_mbs": rng.gamma(2.0, 0.5 + 1.5 * cycle),
        "net_recv_mbs": rng.gamma(2.0, 1.0 + 3 * cycle),
    }
    labels = np.zeros(n, dtype=np.int64)

    n_events = int(n * anomaly_rate / 30)  # events last ~30 steps on average
    for _ in range(n_events):
        length = int(rng.integers(10, 50))
        start = int(rng.integers(0, max(n - length, 1)))
        sl = slice(start, start + length)
        kind = rng.choice(ANOMALY_KINDS)
        if kind == "cpu_spike":
            data["cpu_percent"][sl] = rng.uniform(92, 100, length)
            data["load_per_cpu"][sl] += rng.uniform(1.5, 3.0)
        elif kind == "memory_leak":
            data["mem_percent"][sl] += np.linspace(0, 40, length)
            data["swap_percent"][sl] += np.linspace(0, 30, length)
        elif kind == "disk_burst":
            data["disk_write_mbs"][sl] = rng.uniform(150, 400, length)
            data["load_per_cpu"][sl] += 0.8
        elif kind == "net_flood":
            data["net_recv_mbs"][sl] = rng.uniform(80, 120, length)
            data["net_sent_mbs"][sl] = rng.uniform(40, 80, length)
        elif kind == "fork_bomb":
            data["num_procs"][sl] += np.linspace(0, 800, length)
            data["cpu_percent"][sl] = np.maximum(data["cpu_percent"][sl], 85)
        labels[sl] = 1

    x = np.stack([data[f] for f in FEATURES], axis=1)
    lo = np.zeros(len(FEATURES))
    hi = np.full(len(FEATURES), np.inf)
    for i, f in enumerate(FEATURES):
        if f.endswith("percent"):
            hi[i] = 100
    return np.clip(x, lo, hi).astype(np.float32), labels


def write_csv(path: str | Path, x: np.ndarray, labels: np.ndarray, interval: float = 5.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", *FEATURES, "label"])
        for i, (row, y) in enumerate(zip(x, labels)):
            w.writerow([i * interval, *(f"{v:.4f}" for v in row), int(y)])
