# p-ml — system monitoring with PyTorch

Anomaly detection on host metrics (CPU, memory, swap, load, processes, disk and
network throughput), built from scratch with PyTorch. It doesn't use an
anomaly-detection library.

## How it works

```
psutil ──► collector ──► CSV ──► scaler (log1p + z-score) ──► sliding windows [T=30, F=9]
                                                                     │
                                       LSTM encoder ─► latent z ─► LSTM decoder
                                                                     │
                              reconstruction error per feature ◄─────┘
                                     │
               score > threshold (p99.5 of validation error)? ─► ALERT + top features
```

The model is an **LSTM autoencoder**. It is trained only on *normal*
behaviour, so it learns to rebuild ordinary windows well and unusual ones badly.
A window's reconstruction error is its anomaly score. The per-feature error
shows *which* metric is responsible: `swap_percent 44%, num_procs 43%`.

You don't need labels. The threshold is set from the error distribution on a
held-out, chronologically later slice of normal data. Live alerts are
debounced: by default an alert needs 3 anomalous windows in a row.

## Layout

| File | Purpose |
|---|---|
| `sysmon/features.py` | The metric list, the single source of truth for column order |
| `sysmon/collector.py` | psutil sampler that turns cumulative counters into per-second rates |
| `sysmon/data.py` | CSV loading, `Scaler`, `WindowDataset` |
| `sysmon/model.py` | `LSTMAutoencoder` |
| `sysmon/train.py` | Training loop, early stopping, threshold calibration, checkpointing |
| `sysmon/detector.py` | Offline `score_series`, online `update`, and `evaluate` |
| `sysmon/synthetic.py` | Synthetic metrics with labelled anomalies (CPU spike, memory leak, disk burst, network flood, fork bomb) |
| `sysmon/__main__.py` | CLI |

## Quick start

```bash
pip install -r requirements-dev.txt

# 1. Try the full pipeline on synthetic data (about 30 s on a CPU)
python -m sysmon synth --out data/train.csv --n 8000 --seed 1
python -m sysmon synth --out data/test.csv  --n 4000 --anomaly-rate 0.08 --seed 2
python -m sysmon train --data data/train.csv --epochs 15
python -m sysmon evaluate --data data/test.csv
#   precision ~0.99, recall ~0.97, F1 ~0.98

# 2. Use it on a real machine
python -m sysmon collect --interval 5 --duration 86400   # record about a day of normal use
python -m sysmon train --data data/metrics.csv
python -m sysmon monitor --interval 5 --log alerts.jsonl
```

Example `monitor` output:

```
15:02:36 ok    score=0.3120  cpu=12% mem=41%
15:02:41 high  score=0.9410  cpu=97% mem=42%  <- cpu_percent 61%, load_per_cpu 30%, num_procs 4%
15:02:51 ALERT score=1.2203  cpu=99% mem=42%  <- cpu_percent 58%, load_per_cpu 33%, num_procs 5%
```

## Tips

* **Train on normal data.** The training data should cover your machine's
  normal cycles, such as nightly backups and busy hours. Otherwise those
  cycles will be flagged. If your CSV has a `label` column, rows with
  `label=1` are dropped before training.
* **Keep `--interval` the same** in `collect` and `monitor`. The model learns
  temporal patterns at that sample rate.
* **Tune the threshold:** use `--threshold-percentile` when training, or
  `--threshold` / `--min-consecutive` when monitoring.
* All `TrainConfig` fields are CLI flags: `--window`, `--hidden`, `--latent`,
  `--layers`, `--epochs`, `--lr`, and so on.
* `evaluate` scores windows. A window counts as anomalous if any step in it
  is labelled anomalous.

## Tests

```bash
python -m pytest -q
```
