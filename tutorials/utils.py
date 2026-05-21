"""Shared utilities for workshop notebooks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class StreamflowDataset(Dataset):
    """Sliding-window dataset for sequence-to-sequence streamflow prediction.

    Each sample:
        x : (seq_len, n_features) — normalized climate forcings
        y : (seq_len,) — normalized streamflow (NaNs preserved for masking)

    Parameters
    ----------
    x : (time, basins, features) normalized forcings
    y : (time, basins, 1) normalized streamflow
    seq_len : window length in days
    stride : days between consecutive windows (larger -> fewer, faster samples)
    """

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 seq_len: int = 365, stride: int = 1):
        self.seq_len = seq_len
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        n_time, n_basins, _ = x.shape
        for basin in range(n_basins):
            for t in range(0, n_time - seq_len, stride):
                self.samples.append((
                    x[t:t + seq_len, basin, :].astype(np.float32),
                    y[t:t + seq_len, basin, 0].astype(np.float32),
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss that ignores NaN positions in the target tensor."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return ((pred[mask] - target[mask]) ** 2).mean()


def nse_score(pred: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency on raw (denormalized) streamflow arrays."""
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    if mask.sum() < 2:
        return float("nan")
    p, o = pred[mask], obs[mask]
    denom = np.sum((o - np.mean(o)) ** 2)
    return float(1 - np.sum((p - o) ** 2) / denom) if denom > 0 else float("nan")


def count_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    verbose: bool = True,
    device: torch.device | None = None,
) -> tuple[list[float], list[float]]:
    """Train model and return (train_losses, val_losses) per epoch.

    Restores the best (lowest validation loss) weights at the end.
    """
    if device is None:
        device = next(model.parameters()).device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = masked_mse_loss(model(x_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                val_loss += masked_mse_loss(model(x_b), y_b).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 5 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

    model.load_state_dict(best_state)
    return train_losses, val_losses


def train_model_with_attrs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    verbose: bool = True,
    device: torch.device | None = None,
) -> tuple[list[float], list[float]]:
    """Same as train_model but expects (x_dynamic, x_static, y) batches."""
    if device is None:
        device = next(model.parameters()).device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        for x_dyn, x_stat, y_b in train_loader:
            x_dyn = x_dyn.to(device)
            x_stat = x_stat.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad()
            loss = masked_mse_loss(model(x_dyn, x_stat), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_dyn, x_stat, y_b in val_loader:
                x_dyn = x_dyn.to(device)
                x_stat = x_stat.to(device)
                y_b = y_b.to(device)
                val_loss += masked_mse_loss(model(x_dyn, x_stat), y_b).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 5 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

    model.load_state_dict(best_state)
    return train_losses, val_losses


def predict_full_timeseries(
    model: nn.Module,
    x_norm: np.ndarray,
    seq_len: int = 365,
    device: torch.device | None = None,
) -> np.ndarray:
    """Sliding-window inference for a standard LstmModel.

    For each day t >= seq_len, feeds x[t-seq_len:t] into the model and records
    the prediction at day t (last output of the sequence).

    Returns an array of shape (time, n_basins).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    n_time, n_basins, _ = x_norm.shape
    preds = np.full((n_time, n_basins), np.nan, dtype=np.float32)

    with torch.no_grad():
        for t in range(seq_len, n_time):
            window = x_norm[t - seq_len:t, :, :].transpose(1, 0, 2)
            x_t = torch.from_numpy(window).float().to(device)
            out = model(x_t)
            preds[t] = out[:, -1].cpu().numpy()

    return preds


def predict_ts_with_attrs(
    model: nn.Module,
    x_norm: np.ndarray,
    attrs: np.ndarray,
    seq_len: int = 365,
    device: torch.device | None = None,
) -> np.ndarray:
    """Sliding-window inference for a model that takes static attributes.

    Returns an array of shape (time, n_basins).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    n_time, n_basins, _ = x_norm.shape
    preds = np.full((n_time, n_basins), np.nan, dtype=np.float32)
    attrs_t = torch.from_numpy(attrs.astype(np.float32)).to(device)

    with torch.no_grad():
        for t in range(seq_len, n_time):
            window = x_norm[t - seq_len:t, :, :].transpose(1, 0, 2)
            x_t = torch.from_numpy(window).float().to(device)
            out = model(x_t, attrs_t)
            preds[t] = out[:, -1].cpu().numpy()

    return preds
