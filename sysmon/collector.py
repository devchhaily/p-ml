"""Sample live host metrics with psutil.

Counters such as disk/network bytes are cumulative, so the collector keeps the
previous reading and reports a per-second rate.
"""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Iterator
from pathlib import Path

import psutil

from .features import FEATURES

_MB = 1024 * 1024


class MetricCollector:
    def __init__(self) -> None:
        self._cpus = psutil.cpu_count() or 1
        self._last_t = time.monotonic()
        self._last_disk = psutil.disk_io_counters()  # may be None in containers
        self._last_net = psutil.net_io_counters()
        psutil.cpu_percent(interval=None)  # prime: first call always returns 0.0

    def sample(self) -> dict[str, float]:
        now = time.monotonic()
        dt = max(now - self._last_t, 1e-6)
        disk = psutil.disk_io_counters()
        net = psutil.net_io_counters()

        def rate(cur, prev, attr: str) -> float:
            if cur is None or prev is None:
                return 0.0
            return max(getattr(cur, attr) - getattr(prev, attr), 0) / dt / _MB

        row = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "mem_percent": psutil.virtual_memory().percent,
            "swap_percent": psutil.swap_memory().percent,
            "load_per_cpu": os.getloadavg()[0] / self._cpus if hasattr(os, "getloadavg") else 0.0,
            "num_procs": float(len(psutil.pids())),
            "disk_read_mbs": rate(disk, self._last_disk, "read_bytes"),
            "disk_write_mbs": rate(disk, self._last_disk, "write_bytes"),
            "net_sent_mbs": rate(net, self._last_net, "bytes_sent"),
            "net_recv_mbs": rate(net, self._last_net, "bytes_recv"),
        }
        self._last_t, self._last_disk, self._last_net = now, disk, net
        return row

    def stream(self, interval: float) -> Iterator[dict[str, float]]:
        while True:
            time.sleep(interval)
            yield self.sample()


def collect_to_csv(path: str | Path, interval: float, duration: float | None) -> int:
    """Append samples to ``path`` until ``duration`` seconds pass (or Ctrl-C). Returns rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    collector = MetricCollector()
    deadline = None if duration is None else time.monotonic() + duration
    written = 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", *FEATURES])
        if new_file:
            writer.writeheader()
        try:
            for row in collector.stream(interval):
                writer.writerow({"timestamp": time.time(), **row})
                f.flush()
                written += 1
                if deadline is not None and time.monotonic() >= deadline:
                    break
        except KeyboardInterrupt:
            pass
    return written
