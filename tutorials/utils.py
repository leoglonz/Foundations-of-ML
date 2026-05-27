"""Utility functions.

CIROH Developer's Conference 2026 | Foundations of Machine Learning.

Contents
--------
count_params            Count trainable parameters in an nn.Module
nse_score               Nash-Sutcliffe Efficiency (numpy, NaN-safe)
masked_mse_loss         MSE loss that skips NaN positions in the target
train                   Full training loop with optional checkpoint save/load
predict_full_timeseries Sliding-window inference over a full time series
StreamflowDataset       Sliding-window PyTorch Dataset (dynamic forcings only)
"""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------------------------------------------------------------------
# Device — auto-detected once at import time
# ------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------------------------------------------------------
# 1. count_params
# ------------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters in *model*.

    Parameters
    ----------
    model
        The PyTorch module to inspect.

    Returns
    -------
    int
        Total count of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------------------
# 2. nse_score
# ------------------------------------------------------------------------------

def nse_score(pred: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency on raw (denormalized) streamflow arrays.

    NSE = 1 - SS_res / SS_tot

    Returns NaN when there are fewer than 2 valid paired observations.

    Parameters
    ----------
    pred
        Predicted streamflow (ft3/s), any shape.
    obs
        Observed streamflow (ft3/s), same shape as pred.

    Returns
    -------
    float
        NSE value in (-inf, 1], or NaN if insufficient valid data.
    """
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    if mask.sum() < 2:
        return float('nan')
    p, o = pred[mask], obs[mask]
    denom = np.sum((o - np.mean(o)) ** 2)
    return float(1 - np.sum((p - o) ** 2) / denom) if denom > 0 else float('nan')


# ------------------------------------------------------------------------------
# 3. masked_mse_loss
# ------------------------------------------------------------------------------

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss that ignores NaN positions in *target*.

    Streamflow records contain gaps (sensor outages, ice, etc.).  Standard
    MSE treats NaN as a valid value and produces NaN gradients.  This version
    masks those positions out before computing the mean.

    Parameters
    ----------
    pred
        Model predictions tensor.
    target
        Ground-truth tensor, possibly containing NaN values.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor; differentiably zero when all targets are NaN.
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0  # differentiable zero
    return ((pred[mask] - target[mask]) ** 2).mean()


# ------------------------------------------------------------------------------
# 4. train
# ------------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    verbose: bool = True,
    weights_path: str | Path | None = None,
    train_from_scratch: bool = True,
    save_weights: bool = False,
) -> tuple[list[float], list[float]]:
    """Train *model* and return per-epoch train and validation losses.

      - Adam optimizer with ReduceLROnPlateau scheduler (halves LR after
        5 epochs of no val improvement)
      - Gradient clipping (max_norm=1.0) to prevent exploding gradients
        in the LSTM
      - Best-weight checkpointing: restores the lowest-val-loss weights
        at the end of training

    Works with DataLoaders that yield either (x, y) pairs or
    (x_dynamic, x_static, y) triples — the batch is unpacked
    automatically and all input tensors are forwarded to the model.

    When train_from_scratch is False and a checkpoint exists at weights_path,
    training is skipped entirely and the saved weights are loaded instead.

    Parameters
    ----------
    model
        Accepts one or more input tensors in its forward method.
    train_loader
        DataLoader yielding (*inputs, y) batches.
    val_loader
        DataLoader yielding (*inputs, y) batches.
    n_epochs
        Number of full passes over the training set (also sets the length of
        dummy loss lists when loading pre-saved weights).
    lr
        Initial learning rate.
    verbose
        If True, print a summary line every 5 epochs.
    weights_path
        Where to save or load the .pt checkpoint. If None, saving and
        loading are both skipped silently.
    train_from_scratch
        If True, train normally; if False, load from weights_path.
    save_weights
        If True (and train_from_scratch is True), save after training.

    Returns
    -------
    train_losses
        Per-epoch training MSE loss values.
    val_losses
        Per-epoch validation MSE loss values.
    """
    weights_path = Path(weights_path) if weights_path is not None else None

    if not train_from_scratch:
        if weights_path is not None and weights_path.exists():
            checkpoint = torch.load(weights_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            train_losses = checkpoint.get('train_losses', [0.0] * n_epochs)
            val_losses   = checkpoint.get('val_losses',   [0.0] * n_epochs)
            print(f"Loaded weights from {weights_path.name}  "
                  f"(best val loss: {min(val_losses):.4f})")
            return train_losses, val_losses
        else:
            print(
                f"Warning: train_from_scratch=False but no weights found at "
                f"'{weights_path}'. Training from scratch instead."
            )

    if verbose:
        print(f"Training for {n_epochs} epochs ...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    train_losses, val_losses = [], []
    best_val = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        # ---- training pass ------------------------------------------------
        model.train()
        running = 0.0
        for batch in train_loader:
            *x_parts, y_b = batch
            x_parts = [x.to(device) for x in x_parts]
            y_b = y_b.to(device)
            optimizer.zero_grad()
            loss = masked_mse_loss(model(*x_parts), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        # ---- validation pass ----------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                *x_parts, y_b = batch
                x_parts = [x.to(device) for x in x_parts]
                y_b = y_b.to(device)
                val_loss += masked_mse_loss(model(*x_parts), y_b).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 5 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(
                f"  Epoch {epoch+1:3d}/{n_epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}"
            )

    model.load_state_dict(best_state)

    if save_weights and weights_path is not None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'model_state':  model.state_dict(),
                'train_losses': train_losses,
                'val_losses':   val_losses,
            },
            weights_path,
        )
        print(f"Weights saved to {weights_path}")

    return train_losses, val_losses


# ------------------------------------------------------------------------------
# 8. predict_full_timeseries
# ------------------------------------------------------------------------------

def predict(
    model: nn.Module,
    x_norm: np.ndarray,
    seq_len: int = 365,
    attrs_norm: np.ndarray | None = None,
) -> np.ndarray:
    """Sliding-window inference over a complete time series.

    For each day t >= seq_len, feeds the window x[t-seq_len : t] into the
    model and records the prediction at day t (the last step of the output
    sequence).  Days 0 ... seq_len-1 are left as NaN (no full context window
    available yet).

    Parameters
    ----------
    model
        Trained module. Called as model(x_t) when attrs_norm is None,
        or model(x_t, attrs_t) when attrs_norm is provided.
    x_norm
        Normalised forcing array of shape (time, basins, features).
    seq_len
        Context window length in days.
    attrs_norm
        Optional normalised static attribute array of shape (basins, n_attrs).
        Pass this for attribute-conditioned models.

    Returns
    -------
    np.ndarray
        Float32 array of shape (time, basins) in normalised space. The
        first seq_len rows are NaN.
    """
    model.eval()
    n_time, n_basins, _ = x_norm.shape
    preds = np.full((n_time, n_basins), np.nan, dtype=np.float32)
    attrs_t = torch.from_numpy(attrs_norm).float().to(device) if attrs_norm is not None else None

    with torch.no_grad():
        for t in range(seq_len, n_time):
            window = x_norm[t - seq_len:t, :, :].transpose(1, 0, 2)  # (basins, seq, feat)
            x_t = torch.from_numpy(window).float().to(device)
            out = model(x_t) if attrs_t is None else model(x_t, attrs_t)
            preds[t] = out[:, -1].cpu().numpy()

    return preds


# ------------------------------------------------------------------------------
# 10. StreamflowDataset
# ------------------------------------------------------------------------------

class StreamflowDataset(Dataset):
    """Sliding-window PyTorch Dataset for sequence-to-sequence prediction.

    Conceptually, each training sample is a moving window:
        x[t : t + seq_len]  ->  y[t : t + seq_len]
    repeated for every basin and every valid starting time.

    When attrs is provided each sample also includes the basin's static
    attribute vector, and __getitem__ returns (x, attrs, y) triples instead
    of (x, y) pairs.

    Parameters
    ----------
    x
        Forcing array of shape (time, basins, features).
    y
        Target array of shape (time, basins, 1).
    seq_len
        Window length in days.
    stride
        Days between consecutive windows.
    attrs
        Optional static attribute array of shape (basins, n_attrs). When
        provided, each sample includes the corresponding basin's attributes.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        seq_len: int = 365,
        stride: int = 1,
        attrs: np.ndarray | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.has_attrs = attrs is not None
        self.samples = []
        n_time, n_basins, _ = x.shape
        for basin in range(n_basins):
            for t in range(0, n_time - seq_len, stride):
                x_win = x[t:t + seq_len, basin, :].astype(np.float32)
                y_win = y[t:t + seq_len, basin, 0].astype(np.float32)
                if self.has_attrs:
                    self.samples.append((x_win, attrs[basin].astype(np.float32), y_win))
                else:
                    self.samples.append((x_win, y_win))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.has_attrs:
            x, a, y = self.samples[idx]
            return torch.from_numpy(x), torch.from_numpy(a), torch.from_numpy(y)
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# ------------------------------------------------------------------------------
# 11. Feature engineering helpers
# ------------------------------------------------------------------------------

def _find_precipitation_index(forcing_names: list[str]) -> int:
    """Find the precipitation column in a forcing-name list.

    Falls back to column 0 because most CAMELS forcing tables place
    precipitation first.

    Parameters
    ----------
    forcing_names
        Ordered list of forcing variable names.

    Returns
    -------
    int
        Index of the precipitation column.
    """
    names = [str(name).lower() for name in forcing_names]
    for key in ('prcp', 'precip', 'precipitation', 'pr'):
        for i, name in enumerate(names):
            if key in name:
                return i
    return 0


def _trailing_rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling sum along the time axis using available history.

    For each day t, computes the sum of values[max(0, t-window+1) : t+1].
    This avoids introducing NaNs at the beginning of the record.

    Parameters
    ----------
    values
        Array of shape (time, basins).
    window
        Number of days to include in each rolling sum.

    Returns
    -------
    np.ndarray
        Float32 array of the same shape as values.
    """
    values = np.nan_to_num(values.astype(np.float32), nan=0.0)
    out = np.empty_like(values, dtype=np.float32)
    cumsum = np.cumsum(values, axis=0, dtype=np.float64)

    for t in range(values.shape[0]):
        start = max(0, t - window + 1)
        if start == 0:
            out[t] = cumsum[t]
        else:
            out[t] = cumsum[t] - cumsum[start - 1]

    return out.astype(np.float32)


def make_feature_engineered_inputs(
    loader: Any,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    forcing_names: list[str],
    precip_windows: tuple[int, ...] = (7, 30),
    include_seasonal_encoding: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Create feature-engineered forcing arrays and normalize them.

    Same LSTM architecture + richer input representation -> compare results

    Engineered features
    -------------------
    1. Original forcing variables
    2. Trailing precipitation sums, e.g. 7-day and 30-day accumulated rainfall
    3. Seasonal encoding using sin/cos of day-of-year

    Normalization
    -------------
    Mean and standard deviation are computed from the training period only,
    then reused for validation and testing.

    Parameters
    ----------
    loader
        CamelsSubsetLoader-like object exposing forcings and dates.
    train_mask
        Boolean index array selecting the training time steps.
    val_mask
        Boolean index array selecting the validation time steps.
    test_mask
        Boolean index array selecting the test time steps.
    forcing_names
        Ordered list of forcing variable names in loader.forcings.
    precip_windows
        Accumulation windows (in days) to compute trailing precipitation sums.
    include_seasonal_encoding
        If True, append sin/cos day-of-year columns.

    Returns
    -------
    x_train_fe_norm
        Normalised feature-engineered training array.
    x_val_fe_norm
        Normalised feature-engineered validation array.
    x_test_fe_norm
        Normalised feature-engineered test array.
    feature_names
        Names of all engineered features in column order.
    x_fe_mean
        Training-set feature means used for normalization.
    x_fe_std
        Training-set feature standard deviations used for normalization.
    """
    x_base = np.asarray(loader.forcings, dtype=np.float32)
    dates = np.asarray(loader.dates, dtype='datetime64[D]')
    forcing_names = list(forcing_names)

    feature_arrays = [x_base]
    feature_names = list(forcing_names)

    prcp_idx = _find_precipitation_index(forcing_names)
    prcp = x_base[:, :, prcp_idx]

    for window in precip_windows:
        rolled = _trailing_rolling_sum(prcp, int(window))[:, :, None]
        feature_arrays.append(rolled)
        feature_names.append(f"prcp_{int(window)}day_sum")

    if include_seasonal_encoding:
        years = dates.astype('datetime64[Y]')
        doy = (dates - years).astype(int) + 1
        doy_angle = 2.0 * np.pi * doy.astype(np.float32) / 365.25

        sin_doy = np.sin(doy_angle)[:, None, None].astype(np.float32)
        cos_doy = np.cos(doy_angle)[:, None, None].astype(np.float32)

        n_basins = x_base.shape[1]
        feature_arrays.append(np.repeat(sin_doy, n_basins, axis=1))
        feature_arrays.append(np.repeat(cos_doy, n_basins, axis=1))
        feature_names.extend(['doy_sin', 'doy_cos'])

    x_all_fe = np.concatenate(feature_arrays, axis=2)

    train_mask = np.asarray(train_mask)
    val_mask = np.asarray(val_mask)
    test_mask = np.asarray(test_mask)

    x_train_fe = x_all_fe[train_mask]
    x_val_fe = x_all_fe[val_mask]
    x_test_fe = x_all_fe[test_mask]

    x_fe_mean = x_train_fe.mean(axis=(0, 1), keepdims=True)
    x_fe_std = x_train_fe.std(axis=(0, 1), keepdims=True) + 1e-8

    x_train_fe_norm = (x_train_fe - x_fe_mean) / x_fe_std
    x_val_fe_norm = (x_val_fe - x_fe_mean) / x_fe_std
    x_test_fe_norm = (x_test_fe - x_fe_mean) / x_fe_std

    return (
        x_train_fe_norm,
        x_val_fe_norm,
        x_test_fe_norm,
        feature_names,
        x_fe_mean,
        x_fe_std,
    )


def evaluate(
    model: nn.Module,
    x_test_norm: np.ndarray,
    obs_cfs_test: np.ndarray,
    denormalize_target: Callable,
    seq_len: int,
    gage_ids: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Run full-test inference and compute NSE by basin.

    Parameters
    ----------
    model
        Trained module to evaluate.
    x_test_norm
        Normalised forcing array for the test period,
        shape (time, basins, features).
    obs_cfs_test
        Observed streamflow in ft3/s, shape (time, basins).
    denormalize_target
        Callable that converts normalised predictions back to ft3/s.
    seq_len
        Context window length used during training.
    gage_ids
        Array of gage identifiers, one per basin.

    Returns
    -------
    pred_cfs_test
        Denormalized predictions in ft3/s, shape (time, basins).
    nse_by_basin
        Dictionary mapping each gage ID to its NSE score.
    """
    pred_norm_test = predict(model, x_test_norm, seq_len=seq_len)
    pred_cfs_test = denormalize_target(pred_norm_test)

    nse_by_basin = {}
    for i, gid in enumerate(gage_ids):
        nse_by_basin[gid] = nse_score(pred_cfs_test[:, i], obs_cfs_test[:, i])

    return pred_cfs_test, nse_by_basin


# ------------------------------------------------------------------------------
# 13. make_attr_dataloaders
# ------------------------------------------------------------------------------

def make_attr_dataloaders(
    x_train_norm: np.ndarray,
    y_train_norm: np.ndarray,
    x_val_norm: np.ndarray,
    y_val_norm: np.ndarray,
    attrs_norm: np.ndarray,
    seq_len: int,
    stride: int,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:
    """Build training and validation DataLoaders for static-attribute models.

    Parameters
    ----------
    x_train_norm
        Normalised training forcing array of shape (time, basins, features).
    y_train_norm
        Normalised training target array of shape (time, basins, 1).
    x_val_norm
        Normalised validation forcing array of shape (time, basins, features).
    y_val_norm
        Normalised validation target array of shape (time, basins, 1).
    attrs_norm
        Normalised static attribute array of shape (basins, n_attrs).
    seq_len
        Sliding-window length in days.
    stride
        Days between consecutive windows.
    batch_size
        Batch size for the training DataLoader.

    Returns
    -------
    train_loader_a
        Shuffled DataLoader yielding (x_dynamic, x_static, y) triples.
    val_loader_a
        DataLoader yielding (x_dynamic, x_static, y) triples.
    """
    train_ds_a = StreamflowDataset(x_train_norm, y_train_norm, seq_len, stride, attrs=attrs_norm)
    val_ds_a = StreamflowDataset(x_val_norm, y_val_norm, seq_len, stride, attrs=attrs_norm)
    train_loader_a = DataLoader(train_ds_a, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader_a = DataLoader(val_ds_a, batch_size=batch_size)
    return train_loader_a, val_loader_a
