"""CLI: python -m sysmon {collect,synth,train,evaluate,monitor}"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time

from .train import TrainConfig


def cmd_collect(a: argparse.Namespace) -> None:
    from .collector import collect_to_csv

    print(f"collecting every {a.interval}s -> {a.out} (Ctrl-C to stop)")
    n = collect_to_csv(a.out, a.interval, a.duration)
    print(f"wrote {n} rows")


def cmd_synth(a: argparse.Namespace) -> None:
    from .synthetic import generate, write_csv

    x, y = generate(a.n, a.anomaly_rate, seed=a.seed)
    write_csv(a.out, x, y)
    print(f"wrote {len(x)} rows ({y.mean():.1%} anomalous) -> {a.out}")


def cmd_train(a: argparse.Namespace) -> None:
    from .data import load_csv
    from .train import train

    values, labels = load_csv(a.data)
    if labels is not None and labels.any():
        print(f"dropping {int(labels.sum())} labelled-anomalous rows from training data")
        values = values[labels == 0]
    overrides = {f.name: getattr(a, f.name) for f in dataclasses.fields(TrainConfig) if getattr(a, f.name, None) is not None}
    train(values, TrainConfig(**overrides), a.out)


def cmd_evaluate(a: argparse.Namespace) -> None:
    from .data import load_csv
    from .detector import AnomalyDetector, evaluate

    values, labels = load_csv(a.data)
    if labels is None:
        sys.exit("evaluate needs a CSV with a 'label' column (see `synth`)")
    det = AnomalyDetector(a.model, threshold=a.threshold)
    metrics = evaluate(det.score_series(values), labels, det.threshold, det.window)
    print(json.dumps({"threshold": det.threshold, **metrics}, indent=2))


def cmd_monitor(a: argparse.Namespace) -> None:
    from .collector import MetricCollector
    from .detector import AnomalyDetector

    det = AnomalyDetector(a.model, threshold=a.threshold, min_consecutive=a.min_consecutive)
    log = open(a.log, "a") if a.log else None
    print(f"monitoring every {a.interval}s, warm-up {det.window} samples, threshold {det.threshold:.4f}")
    try:
        for sample in MetricCollector().stream(a.interval):
            v = det.update(sample)
            if v is None:
                continue
            ts = time.strftime("%H:%M:%S")
            status = "ALERT" if v.alert else ("high " if v.is_anomaly else "ok   ")
            top = ", ".join(f"{name} {share:.0%}" for name, share in v.top_features)
            print(f"{ts} {status} score={v.score:.4f}  cpu={sample['cpu_percent']:.0f}% mem={sample['mem_percent']:.0f}%"
                  + (f"  <- {top}" if v.is_anomaly else ""))
            if log:
                log.write(json.dumps({"ts": time.time(), **dataclasses.asdict(v), "sample": sample}) + "\n")
                log.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if log:
            log.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sysmon", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("collect", help="record live metrics to CSV")
    s.add_argument("--out", default="data/metrics.csv")
    s.add_argument("--interval", type=float, default=5.0, help="seconds between samples")
    s.add_argument("--duration", type=float, help="seconds to run (default: until Ctrl-C)")
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("synth", help="generate synthetic labelled metrics")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=10_000)
    s.add_argument("--anomaly-rate", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("train", help="train the autoencoder on normal data")
    s.add_argument("--data", required=True)
    s.add_argument("--out", default="models/sysmon.pt")
    for f in dataclasses.fields(TrainConfig):
        s.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), help=f"default {f.default}")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="precision/recall/F1 on a labelled CSV")
    s.add_argument("--data", required=True)
    s.add_argument("--model", default="models/sysmon.pt")
    s.add_argument("--threshold", type=float, help="override the calibrated threshold")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("monitor", help="score live metrics and raise alerts")
    s.add_argument("--model", default="models/sysmon.pt")
    s.add_argument("--interval", type=float, default=5.0, help="must match the interval used in `collect`")
    s.add_argument("--threshold", type=float)
    s.add_argument("--min-consecutive", type=int, default=3, help="anomalous windows in a row before ALERT")
    s.add_argument("--log", help="append JSONL verdicts to this file")
    s.set_defaults(fn=cmd_monitor)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
