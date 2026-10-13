"""Training loop + threshold calibration. Produces a single self-contained checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Scaler, WindowDataset
from .features import FEATURES
from .model import LSTMAutoencoder


@dataclass
class TrainConfig:
    window: int = 30
    hidden: int = 64
    latent: int = 16
    layers: int = 1
    dropout: float = 0.0
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    val_fraction: float = 0.2
    patience: int = 5
    threshold_percentile: float = 99.5
    seed: int = 0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def score_windows(model: LSTMAutoencoder, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Per-window, per-feature reconstruction error: [N, F]."""
    model.eval()
    return torch.cat([model.feature_errors(b.to(device)).cpu() for b in loader]).numpy()


def train(values: np.ndarray, cfg: TrainConfig, out_path: str | Path, log=print) -> dict:
    """Train on ``values`` [T, F] assumed to be (mostly) normal behaviour."""
    torch.manual_seed(cfg.seed)
    device = pick_device()

    # Split chronologically: validation is the most recent slice, so we never
    # train on the future and the threshold reflects unseen data.
    split = int(len(values) * (1 - cfg.val_fraction))
    train_raw, val_raw = values[:split], values[split:]
    scaler = Scaler.fit(train_raw)  # fit on train only to avoid leakage

    train_ds = WindowDataset(scaler.transform(train_raw), cfg.window)
    val_ds = WindowDataset(scaler.transform(val_raw), cfg.window)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = LSTMAutoencoder(len(FEATURES), cfg.hidden, cfg.latent, cfg.layers, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = torch.nn.MSELoss()

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_dl:
            batch = batch.to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(batch)
        train_loss = total / len(train_ds)
        val_loss = float(score_windows(model, val_dl, device).mean())
        sched.step(val_loss)
        log(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                log(f"early stop (no improvement for {cfg.patience} epochs)")
                break

    model.load_state_dict(best_state)
    val_scores = score_windows(model, val_dl, device).mean(axis=1)
    threshold = float(np.percentile(val_scores, cfg.threshold_percentile))
    log(f"threshold (p{cfg.threshold_percentile} of val error) = {threshold:.4f}")

    ckpt = {
        "model_config": model.config,
        "model_state": model.state_dict(),
        "scaler": scaler.state_dict(),
        "features": FEATURES,
        "window": cfg.window,
        "threshold": threshold,
        "train_config": vars(cfg),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    log(f"saved {out_path}")
    return ckpt
