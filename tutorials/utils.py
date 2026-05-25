"""
utils.py — Workshop utility functions
CIROH Developer's Conference 2026 | Foundations of Machine Learning

Functions here were deliberately moved OUT of the notebooks because they are
infrastructure, not teaching content.  Participants should understand *what*
each one does (described below) without needing to read the implementation
during the workshop.

Contents
--------
count_params            Count trainable parameters in an nn.Module
nse_score               Nash-Sutcliffe Efficiency (numpy, NaN-safe)
masked_mse_loss         MSE loss that skips NaN positions in the target
train_model             Full training loop with scheduler + best-weight restore
train_model_with_attrs  Same, but for models that take (x_dynamic, x_static)
load_or_train           Train or load pre-saved weights based on workshop flags
load_or_train_with_attrs  Same, for models that take (x_dynamic, x_static)
predict_full_timeseries Sliding-window inference over a full time series
predict_ts_with_attrs   Same, for attribute-conditioned models
StreamflowDataset       Sliding-window PyTorch Dataset (dynamic forcings only)
make_overfit_loader     Build a tiny one-year DataLoader for overfitting demo
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Device — auto-detected once at import time
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. count_params
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 2. nse_score
# ---------------------------------------------------------------------------

def nse_score(pred: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency on raw (denormalized) streamflow arrays.

    NSE = 1 - SS_res / SS_tot

    Returns NaN when there are fewer than 2 valid paired observations.

    Parameters
    ----------
    pred : predicted streamflow (ft³/s), any shape
    obs  : observed streamflow (ft³/s), same shape as pred
    """
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    if mask.sum() < 2:
        return float("nan")
    p, o = pred[mask], obs[mask]
    denom = np.sum((o - np.mean(o)) ** 2)
    return float(1 - np.sum((p - o) ** 2) / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# 3. masked_mse_loss
# ---------------------------------------------------------------------------

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss that ignores NaN positions in *target*.

    Streamflow records contain gaps (sensor outages, ice, etc.).  Standard
    MSE treats NaN as a valid value and produces NaN gradients.  This version
    masks those positions out before computing the mean.
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0          # differentiable zero
    return ((pred[mask] - target[mask]) ** 2).mean()


# ---------------------------------------------------------------------------
# 4. train_model
# ---------------------------------------------------------------------------

def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    verbose: bool = True,
):
    """Train *model* and return (train_losses, val_losses) per epoch.

    Production details handled here so the notebook can stay focused on
    concepts:
      - Adam optimizer with ReduceLROnPlateau scheduler (halves LR after
        5 epochs of no val improvement)
      - Gradient clipping (max_norm=1.0) to prevent exploding gradients
        in the LSTM
      - Best-weight checkpointing: restores the lowest-val-loss weights
        at the end of training

    Parameters
    ----------
    model        : nn.Module — must accept (x_batch,) in forward()
    train_loader : DataLoader yielding (x, y) batches
    val_loader   : DataLoader yielding (x, y) batches
    n_epochs     : number of full passes over the training set
    lr           : initial learning rate
    verbose      : if True, print a summary line every 5 epochs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        # ---- training pass ------------------------------------------------
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

        # ---- validation pass ----------------------------------------------
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
            print(
                f"  Epoch {epoch+1:3d}/{n_epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}"
            )

    model.load_state_dict(best_state)
    return train_losses, val_losses


# ---------------------------------------------------------------------------
# 5. train_model_with_attrs
# ---------------------------------------------------------------------------

def train_model_with_attrs(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    verbose: bool = True,
):
    """Same as train_model, but for models whose forward() takes
    (x_dynamic, x_static) — used in Notebook 2 for LstmWithAttrs.

    DataLoaders must yield (x_dynamic, x_static, y) triples.
    """
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
            x_dyn, x_stat, y_b = x_dyn.to(device), x_stat.to(device), y_b.to(device)
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
                x_dyn, x_stat, y_b = x_dyn.to(device), x_stat.to(device), y_b.to(device)
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
            print(
                f"  Epoch {epoch+1:3d}/{n_epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}"
            )

    model.load_state_dict(best_state)
    return train_losses, val_losses


# ---------------------------------------------------------------------------
# 6. load_or_train
# ---------------------------------------------------------------------------

def load_or_train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    verbose: bool = True,
    weights_path=None,
    train_from_scratch: bool = True,
    save_weights: bool = False,
):
    """Train *model* or load pre-saved weights, controlled by workshop flags.

    This wrapper exists so notebook training cells stay clean while still
    supporting both workshop modes:

      TRAIN_FROM_SCRATCH = True  — normal training run, optionally saves weights
      TRAIN_FROM_SCRATCH = False — loads weights from *weights_path* instantly,
                                   skips training entirely

    When loading, dummy loss lists of the correct length are returned so that
    any downstream plotting calls (e.g. plot_learning_curves) still work.

    Parameters
    ----------
    model             : nn.Module to train or load into
    train_loader      : DataLoader yielding (x, y) batches
    val_loader        : DataLoader yielding (x, y) batches
    n_epochs          : number of epochs to train (used for training and for
                        the length of dummy loss lists when loading)
    lr                : initial learning rate
    verbose           : if True, print a summary line every 5 epochs
    weights_path      : Path-like — where to save / load the .pt checkpoint.
                        If None, saving and loading are both skipped silently.
    train_from_scratch: if True, train normally; if False, load from weights_path
    save_weights      : if True (and train_from_scratch is True), save after training

    Returns
    -------
    train_losses, val_losses : lists of per-epoch loss values
    """
    from pathlib import Path

    weights_path = Path(weights_path) if weights_path is not None else None

    if not train_from_scratch:
        if weights_path is not None and weights_path.exists():
            checkpoint = torch.load(weights_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            train_losses = checkpoint.get("train_losses", [0.0] * n_epochs)
            val_losses   = checkpoint.get("val_losses",   [0.0] * n_epochs)
            print(f"Loaded weights from {weights_path.name}  "
                  f"(best val loss: {min(val_losses):.4f})")
            return train_losses, val_losses
        else:
            print(
                f"Warning: TRAIN_FROM_SCRATCH=False but no weights found at "
                f"'{weights_path}'. Training from scratch instead."
            )

    # --- train normally -----------------------------------------------------
    if verbose:
        print(f"Training for {n_epochs} epochs ...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, n_epochs=n_epochs, lr=lr, verbose=verbose
    )

    if save_weights and weights_path is not None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state":  model.state_dict(),
                "train_losses": train_losses,
                "val_losses":   val_losses,
            },
            weights_path,
        )
        print(f"Weights saved to {weights_path}")

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# 7. load_or_train_with_attrs
# ---------------------------------------------------------------------------

def load_or_train_with_attrs(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    verbose: bool = True,
    weights_path=None,
    train_from_scratch: bool = True,
    save_weights: bool = False,
):
    """Same as load_or_train, but wraps train_model_with_attrs.

    Use this for models whose forward() takes (x_dynamic, x_static), such as
    LstmWithAttrs in Notebook 2. DataLoaders must yield (x_dynamic, x_static, y)
    triples.

    Parameters
    ----------
    model             : nn.Module whose forward() takes (x_dyn, x_stat)
    train_loader      : DataLoader yielding (x_dynamic, x_static, y) triples
    val_loader        : DataLoader yielding (x_dynamic, x_static, y) triples
    n_epochs          : number of epochs to train
    lr                : initial learning rate
    verbose           : if True, print a summary line every 5 epochs
    weights_path      : Path-like — where to save / load the .pt checkpoint
    train_from_scratch: if True, train normally; if False, load from weights_path
    save_weights      : if True (and train_from_scratch is True), save after training

    Returns
    -------
    train_losses, val_losses : lists of per-epoch loss values
    """
    from pathlib import Path

    weights_path = Path(weights_path) if weights_path is not None else None

    if not train_from_scratch:
        if weights_path is not None and weights_path.exists():
            checkpoint = torch.load(weights_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            train_losses = checkpoint.get("train_losses", [0.0] * n_epochs)
            val_losses   = checkpoint.get("val_losses",   [0.0] * n_epochs)
            print(f"Loaded weights from {weights_path.name}  "
                  f"(best val loss: {min(val_losses):.4f})")
            return train_losses, val_losses
        else:
            print(
                f"Warning: TRAIN_FROM_SCRATCH=False but no weights found at "
                f"'{weights_path}'. Training from scratch instead."
            )

    # --- train normally -----------------------------------------------------
    if verbose:
        print(f"Training for {n_epochs} epochs ...")
    train_losses, val_losses = train_model_with_attrs(
        model, train_loader, val_loader, n_epochs=n_epochs, lr=lr, verbose=verbose
    )

    if save_weights and weights_path is not None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state":  model.state_dict(),
                "train_losses": train_losses,
                "val_losses":   val_losses,
            },
            weights_path,
        )
        print(f"Weights saved to {weights_path}")

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# 8. predict_full_timeseries
# ---------------------------------------------------------------------------

def predict_full_timeseries(
    model,
    x_norm: np.ndarray,
    seq_len: int = 365,
) -> np.ndarray:
    """Sliding-window inference over a complete time series.

    For each day t >= seq_len, feeds the window x[t-seq_len : t] into the
    model and records the prediction at day t (the last step of the output
    sequence).  Days 0 … seq_len-1 are left as NaN (no full context window
    available yet).

    Parameters
    ----------
    model   : trained nn.Module
    x_norm  : (time, basins, features) normalised forcing array
    seq_len : context window length in days

    Returns
    -------
    preds : (time, basins) float32 array in normalised space
    """
    model.eval()
    n_time, n_basins, _ = x_norm.shape
    preds = np.full((n_time, n_basins), np.nan, dtype=np.float32)

    with torch.no_grad():
        for t in range(seq_len, n_time):
            window = x_norm[t - seq_len:t, :, :].transpose(1, 0, 2)   # (basins, seq, feat)
            x_t = torch.from_numpy(window).float().to(device)
            out = model(x_t)
            preds[t] = out[:, -1].cpu().numpy()

    return preds


# ---------------------------------------------------------------------------
# 9. predict_ts_with_attrs
# ---------------------------------------------------------------------------

def predict_ts_with_attrs(
    model,
    x_norm: np.ndarray,
    attrs_norm: np.ndarray,
    seq_len: int = 365,
) -> np.ndarray:
    """Same as predict_full_timeseries, for attribute-conditioned models.

    Parameters
    ----------
    model      : trained nn.Module whose forward() takes (x_dyn, x_stat)
    x_norm     : (time, basins, features) normalised forcing array
    attrs_norm : (basins, n_attrs) normalised static attribute array
    seq_len    : context window length in days
    """
    model.eval()
    n_time, n_basins, _ = x_norm.shape
    preds = np.full((n_time, n_basins), np.nan, dtype=np.float32)
    attrs_t = torch.from_numpy(attrs_norm).float().to(device)   # (basins, n_attrs)

    with torch.no_grad():
        for t in range(seq_len, n_time):
            window = x_norm[t - seq_len:t, :, :].transpose(1, 0, 2)
            x_t = torch.from_numpy(window).float().to(device)
            out = model(x_t, attrs_t)
            preds[t] = out[:, -1].cpu().numpy()

    return preds


# ---------------------------------------------------------------------------
# 10. StreamflowDataset
# ---------------------------------------------------------------------------

class StreamflowDataset(Dataset):
    """Sliding-window PyTorch Dataset for sequence-to-sequence prediction.

    Kept here because the sliding-window indexing is infrastructure, not the
    main teaching target during a short hands-on workshop. Participants can
    read this well-commented version after the session if they want to inspect
    how PyTorch Dataset objects are built.

    Conceptually, each training sample is a moving window:
        x[t : t + seq_len]  ->  y[t : t + seq_len]
    repeated for every basin and every valid starting time.

    Each sample
    -----------
    x : (seq_len, n_features)  normalised climate forcings  [float32]
    y : (seq_len,)             normalised streamflow         [float32, NaNs ok]

    Parameters
    ----------
    x       : (time, basins, features)
    y       : (time, basins, 1)
    seq_len : window length in days
    stride  : days between consecutive windows
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        seq_len: int = 365,
        stride: int = 1,
    ):
        self.seq_len = seq_len
        self.samples = []
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

# ---------------------------------------------------------------------------
# 11. make_overfit_loader
# ---------------------------------------------------------------------------

def make_overfit_loader(
    loader,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    normalize_target,
    seq_len: int,
    start_date: str = "1990-10-01",
    end_date: str = "1991-09-30",
    batch_size: int = 32,
    stride: int = 1,
    fallback_loader: DataLoader | None = None,
):
    """Create a tiny one-year DataLoader for the overfitting demonstration.

    This helper hides workshop infrastructure so the notebook can focus on the
    concept: overfitting is easier to create when a high-capacity model is
    trained on a very small dataset.

    Parameters
    ----------
    loader : CamelsSubsetLoader-like object
        Must expose dates, forcings, and target.
    x_mean, x_std : arrays
        Training-set forcing normalization statistics from Notebook 1.
    normalize_target : callable
        Function defined in Notebook 1, using training-set target statistics.
    seq_len : int
        Sliding-window length in days.
    start_date, end_date : str
        Date range used to create the tiny training subset.
    batch_size : int
        Batch size for the returned DataLoader.
    stride : int
        Sliding-window stride.
    fallback_loader : DataLoader or None
        If the chosen tiny period is too short to create samples, return this
        loader instead. This keeps the live workshop from failing if data are
        changed later.

    Returns
    -------
    DataLoader
        A small training loader for the overfitting experiment.
    """
    dates = np.asarray(loader.dates, dtype="datetime64[D]")
    tiny_mask = (dates > np.datetime64(start_date)) & (dates <= np.datetime64(end_date))

    x_tiny_norm = (loader.forcings[tiny_mask] - x_mean) / x_std
    y_tiny_norm = normalize_target(loader.target[tiny_mask])

    tiny_ds = StreamflowDataset(x_tiny_norm, y_tiny_norm, seq_len=seq_len, stride=stride)

    if len(tiny_ds) == 0:
        if fallback_loader is not None:
            return fallback_loader
        raise ValueError(
            "The selected overfitting period is too short for the requested "
            "sequence length. Choose a longer date range or pass fallback_loader."
        )

    return DataLoader(tiny_ds, batch_size=batch_size, shuffle=True)



# ---------------------------------------------------------------------------
# 12. Feature engineering helpers
# ---------------------------------------------------------------------------

def _find_precipitation_index(forcing_names) -> int:
    """Find the precipitation column in a forcing-name list.

    Falls back to column 0 because most CAMELS forcing tables place
    precipitation first.
    """
    names = [str(name).lower() for name in forcing_names]
    for key in ("prcp", "precip", "precipitation", "pr"):
        for i, name in enumerate(names):
            if key in name:
                return i
    return 0


def _trailing_rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling sum along time using available history.

    values has shape (time, basins). For each day t, this computes the sum of
    values[max(0, t-window+1) : t+1]. This avoids introducing NaNs at the
    beginning of the record.
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
    loader,
    train_mask,
    val_mask,
    test_mask,
    forcing_names,
    precip_windows=(7, 30),
    include_seasonal_encoding: bool = True,
):
    """Create feature-engineered forcing arrays and normalize them.

    This helper keeps feature-construction infrastructure out of the notebook.
    The notebook should focus on the teaching point:

        same LSTM architecture + richer input representation -> compare results

    Engineered features
    -------------------
    1. Original forcing variables
    2. Trailing precipitation sums, e.g. 7-day and 30-day accumulated rainfall
    3. Seasonal encoding using sin/cos of day-of-year

    Normalization
    -------------
    Mean and standard deviation are computed from the training period only,
    then reused for validation and testing.
    """
    x_base = np.asarray(loader.forcings, dtype=np.float32)
    dates = np.asarray(loader.dates, dtype="datetime64[D]")
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
        years = dates.astype("datetime64[Y]")
        doy = (dates - years).astype(int) + 1
        doy_angle = 2.0 * np.pi * doy.astype(np.float32) / 365.25

        sin_doy = np.sin(doy_angle)[:, None, None].astype(np.float32)
        cos_doy = np.cos(doy_angle)[:, None, None].astype(np.float32)

        n_basins = x_base.shape[1]
        feature_arrays.append(np.repeat(sin_doy, n_basins, axis=1))
        feature_arrays.append(np.repeat(cos_doy, n_basins, axis=1))
        feature_names.extend(["doy_sin", "doy_cos"])

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


def evaluate_streamflow_model(
    model,
    x_test_norm: np.ndarray,
    obs_cfs_test: np.ndarray,
    denormalize_target,
    seq_len: int,
    gage_ids,
):
    """Run full-test inference and compute NSE by basin.

    Kept as a helper because it repeats the evaluation infrastructure already
    shown earlier in the notebook.
    """
    pred_norm_test = predict_full_timeseries(model, x_test_norm, seq_len=seq_len)
    pred_cfs_test = denormalize_target(pred_norm_test)

    nse_by_basin = {}
    for i, gid in enumerate(gage_ids):
        nse_by_basin[gid] = nse_score(pred_cfs_test[:, i], obs_cfs_test[:, i])

    return pred_cfs_test, nse_by_basin

# ---------------------------------------------------------------------------
# 13. StreamflowDatasetWithAttrs
# ---------------------------------------------------------------------------

class StreamflowDatasetWithAttrs(Dataset):
    """Extends StreamflowDataset to also return per-basin static attributes.

    Moved out of Notebook 2 because the sliding-window indexing and PyTorch
    Dataset plumbing are infrastructure. Conceptually, each sample is:
        (x_dynamic, x_static, y)

    This is the same logic that was originally shown in the notebook.
    """
    def __init__(self, x, y, attrs, seq_len=365, stride=1):
        self.samples = []
        n_time, n_basins, _ = x.shape
        for basin in range(n_basins):
            for t in range(0, n_time - seq_len, stride):
                self.samples.append((
                    x[t:t + seq_len, basin, :].astype(np.float32),
                    attrs[basin].astype(np.float32),
                    y[t:t + seq_len, basin, 0].astype(np.float32),
                ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x_dyn, x_stat, y = self.samples[idx]
        return torch.from_numpy(x_dyn), torch.from_numpy(x_stat), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# 14. make_attr_dataloaders
# ---------------------------------------------------------------------------

def make_attr_dataloaders(
    x_train_norm: np.ndarray,
    y_train_norm: np.ndarray,
    x_val_norm: np.ndarray,
    y_val_norm: np.ndarray,
    attrs_norm: np.ndarray,
    seq_len: int,
    stride: int,
    batch_size: int = 128,
):
    """Build training and validation DataLoaders for static-attribute models.

    This keeps the notebook focused on the concept — adding static basin
    attributes — rather than repeating Dataset/DataLoader boilerplate.
    """
    train_ds_a = StreamflowDatasetWithAttrs(x_train_norm, y_train_norm, attrs_norm, seq_len, stride)
    val_ds_a = StreamflowDatasetWithAttrs(x_val_norm, y_val_norm, attrs_norm, seq_len, stride)
    train_loader_a = DataLoader(train_ds_a, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader_a = DataLoader(val_ds_a, batch_size=batch_size)
    return train_loader_a, val_loader_a
