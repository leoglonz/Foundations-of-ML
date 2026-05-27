"""Plotting helpers.

CIROH Developer's Conference 2026 | Foundations of Machine Learning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from scipy import stats


def plot_gage_locations(
    loader: Any,
    data_dir: str | Path,
    nse_by_basin: dict[int, float] | None = None,
    title: str | None = None,
) -> None:
    """Map CAMELS basin watershed polygons, optionally colored by NSE.

    Parameters
    ----------
    loader
        CamelsSubsetLoader exposing a gage_ids attribute.
    data_dir
        Root data directory containing loc/camels_subset.shp.
    nse_by_basin
        Mapping of gage ID to NSE value. When provided, watersheds are filled
        by NSE on a RdYlGn colormap clipped to [-0.5, 1].
    title
        Optional figure title.
    """
    shp_path = Path(data_dir) / 'loc' / 'camels_subset.shp'
    gdf = gpd.read_file(shp_path)

    gage_ids = loader.gage_ids

    if nse_by_basin is not None:
        gdf['nse'] = gdf['hru_id'].map(
            {int(g): nse_by_basin.get(g, np.nan) for g in gage_ids}
        )
        cmap = plt.cm.RdYlGn
        norm = mcolors.Normalize(vmin=-0.5, vmax=1.0)
        colors = [cmap(norm(v)) if not np.isnan(v) else '#cccccc' for v in gdf['nse']]
    else:
        colors = ['steelblue'] * len(gdf)

    fig, ax = plt.subplots(figsize=(8, 7))

    gdf.plot(ax=ax, color=colors, edgecolor='k', linewidth=0.6, zorder=2)

    # Scatter centroid markers
    ax.scatter(
        gdf['lon_cen'], gdf['lat_cen'],
        color='white', edgecolors='k', linewidths=0.8,
        s=50, zorder=3,
    )

    # Annotate with gage IDs
    for _, row in gdf.iterrows():
        ax.annotate(
            str(int(row['hru_id'])),
            xy=(row['lon_cen'], row['lat_cen']),
            xytext=(5, 4),
            textcoords='offset points',
            fontsize=7.5,
            color='#111111',
            zorder=4,
        )

    if nse_by_basin is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('NSE', fontsize=11)

    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(
        title or 'CAMELS Subset — Watershed Boundaries (Southern Appalachians)',
        fontsize=12,
    )
    ax.set_facecolor("#eef2f7")
    ax.grid(alpha=0.35)

    plt.tight_layout()
    plt.show()


def plot_basin_overview(loader: Any, basin_idx: int = 0) -> None:
    """Plot precipitation, temperature, and streamflow for one CAMELS basin.

    Parameters
    ----------
    loader
        CamelsSubsetLoader exposing dates, forcings, target, and
        gage_ids.
    basin_idx
        Index of the basin to plot.
    """
    dates = loader.dates
    forcing_names = getattr(loader, 'forcing_names', None)

    # The workshop loader exposes FORCING_NAMES separately, but some loader
    # versions also attach forcing names to the object. Fall back to common names.
    if forcing_names is None:
        try:
            from camels_loader import FORCING_NAMES as forcing_names
        except Exception:
            forcing_names = ['prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']

    prcp_idx = forcing_names.index('prcp')
    if 'tmean' in forcing_names:
        temp = loader.forcings[:, basin_idx, forcing_names.index('tmean')]
        temp_label = 'Mean Temperature'
    elif {'tmax', 'tmin'}.issubset(set(forcing_names)):
        temp = 0.5 * (
            loader.forcings[:, basin_idx, forcing_names.index('tmax')]
            + loader.forcings[:, basin_idx, forcing_names.index('tmin')]
        )
        temp_label = 'Mean Temperature'
    else:
        temp = loader.forcings[:, basin_idx, min(1, loader.forcings.shape[-1] - 1)]
        temp_label = 'Temperature-like forcing'

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"Basin {loader.gage_ids[basin_idx]} — Daily Climate Forcings & Streamflow",
        fontsize=13,
        fontweight='bold',
    )

    axes[0].bar(
        dates,
        loader.forcings[:, basin_idx, prcp_idx],
        color='steelblue',
        width=1,
        alpha=0.75,
        label='Precipitation',
    )
    axes[0].set_ylabel('Precip (mm/day)')
    axes[0].legend(loc='upper right', fontsize=8)

    axes[1].plot(
        dates,
        temp,
        color='darkorange',
        linewidth=0.5,
        label=temp_label,
    )
    axes[1].axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
    axes[1].set_ylabel('Temp (degC)')
    axes[1].legend(loc='upper right', fontsize=8)

    sf = loader.target[:, basin_idx, 0]
    axes[2].fill_between(dates, sf, alpha=0.35, color='navy', linewidth=0)
    axes[2].plot(dates, sf, color='navy', linewidth=0.4, label='Streamflow')
    axes[2].set_ylabel('Streamflow (ft3/s)')
    axes[2].set_ylim(bottom=0)
    axes[2].legend(loc='upper right', fontsize=8)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_target_normalization_histogram(
    y_train: np.ndarray,
    y_train_norm: np.ndarray,
) -> None:
    """Show how log + z-score normalization changes the target distribution.

    Parameters
    ----------
    y_train
        Raw training streamflow values (ft3/s), any shape with possible NaNs.
    y_train_norm
        Normalized training streamflow values, same shape as y_train.
    """
    raw_vals = y_train[~np.isnan(y_train)].ravel()
    norm_vals = y_train_norm[~np.isnan(y_train_norm)].ravel()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    axes[0].hist(raw_vals, bins=80, color='navy', alpha=0.75, edgecolor='none')
    axes[0].set_title('Raw streamflow (ft3/s)')
    axes[0].set_xlabel('ft3/s')

    axes[1].hist(norm_vals, bins=80, color='steelblue', alpha=0.75, edgecolor='none')
    axes[1].set_title('Normalized (log + z-score)')
    axes[1].set_xlabel('σ units')

    plt.tight_layout()
    plt.show()


def plot_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    title: str = 'Learning Curves',
) -> None:
    """Plot train and validation loss curves.

    Parameters
    ----------
    train_losses
        Per-epoch training MSE loss values.
    val_losses
        Per-epoch validation MSE loss values.
    title
        Figure title.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, color='steelblue', linewidth=2, label='Train loss')
    ax.plot(epochs, val_losses, color='darkorange', linewidth=2, linestyle='--', label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss  (normalized log-space)')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_hydrograph(
    dates_test: pd.DatetimeIndex,
    obs_cfs_test: np.ndarray,
    pred_cfs_test: np.ndarray,
    gage_ids: np.ndarray,
    nse_by_basin: dict,
    basin_idx: int = 0,
    zoom: slice = slice(-548, None),
) -> None:
    """Plot observed vs predicted streamflow for one basin.

    Parameters
    ----------
    dates_test
        Date index for the test period.
    obs_cfs_test
        Observed streamflow in ft3/s, shape (time, basins).
    pred_cfs_test
        Predicted streamflow in ft3/s, shape (time, basins).
    gage_ids
        Array of gage identifiers, one per basin.
    nse_by_basin
        Mapping of gage ID to NSE score.
    basin_idx
        Index of the basin to plot.
    zoom
        Slice selecting the zoomed subplot period (default: last 18 months).
    """
    gid = gage_ids[basin_idx]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)

    ax = axes[0]
    ax.plot(dates_test, obs_cfs_test[:, basin_idx], color='navy', lw=0.8, label='Observed')
    ax.plot(
        dates_test,
        pred_cfs_test[:, basin_idx],
        color='tomato',
        lw=0.8,
        label='Predicted',
        alpha=0.85,
    )
    ax.set_title(f"Full test period — Basin {gid}   NSE = {nse_by_basin[gid]:.3f}")
    ax.set_ylabel("Streamflow (ft3/s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[1]
    ax.plot(dates_test[zoom], obs_cfs_test[zoom, basin_idx], color="navy", lw=1.0, label="Observed")
    ax.plot(
        dates_test[zoom],
        pred_cfs_test[zoom, basin_idx],
        color="tomato",
        lw=1.0,
        label="Predicted",
        alpha=0.85,
    )
    ax.set_title("Zoom: last 18 months")
    ax.set_ylabel('Streamflow (ft3/s)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.show()


def plot_scatter_basins(
    obs_cfs_test: np.ndarray,
    pred_cfs_test: np.ndarray,
    gage_ids: np.ndarray,
    nse_by_basin: dict,
) -> None:
    """Plot observed vs predicted scatter plots for all basins.

    Parameters
    ----------
    obs_cfs_test
        Observed streamflow in ft3/s, shape (time, basins).
    pred_cfs_test
        Predicted streamflow in ft3/s, shape (time, basins).
    gage_ids
        Array of gage identifiers, one per basin.
    nse_by_basin
        Mapping of gage ID to NSE score.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, (ax, gid) in enumerate(zip(axes.ravel(), gage_ids)):
        obs = obs_cfs_test[:, i]
        pred = pred_cfs_test[:, i]
        mask = ~np.isnan(obs) & ~np.isnan(pred)

        q99 = float(np.nanpercentile(obs, 99))
        ax.scatter(obs[mask], pred[mask], alpha=0.08, s=2, color='steelblue', rasterized=True)
        ax.plot([0, q99], [0, q99], 'r--', lw=1)
        ax.set_xlim(0, q99)
        ax.set_ylim(0, q99)
        ax.set_title(f"{gid}\nNSE={nse_by_basin[gid]:.2f}", fontsize=9)
        if i >= 5:
            ax.set_xlabel('Observed (ft3/s)', fontsize=8)
        if i % 5 == 0:
            ax.set_ylabel('Predicted (ft3/s)', fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle('Observed vs. Predicted Streamflow — Test Period', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_bias_variance(
    dates_test: pd.DatetimeIndex,
    obs_cfs_test: np.ndarray,
    configs: list[tuple],
    basin_idx: int = 0,
    n_plot: int = 365,
    seq_len: int = 365,
    save_path: str | None = 'bias_variance_demo.png',
) -> None:
    """Compare underfitting, baseline, and overfitting runs.

    Parameters
    ----------
    dates_test
        Date index for the test period.
    obs_cfs_test
        Observed streamflow in ft3/s, shape (time, basins).
    configs
        List of tuples, one per model variant. Each tuple must contain:
        (title, train_losses, val_losses, prediction_array, nse_value, color).
    basin_idx
        Index of the basin to plot in hydrograph panels.
    n_plot
        Number of days to show in the hydrograph zoom.
    seq_len
        Context window length; used as the starting offset for the zoom.
    save_path
        File path to save the figure, or None to skip saving.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    start = seq_len
    plot_slice = slice(start, start + n_plot)
    d_plot = dates_test[plot_slice]

    for col, (title, tl, vl, preds, nse, c) in enumerate(configs):
        ax = axes[0, col]
        ax.plot(tl, color=c, lw=2, label='Train')
        ax.plot(vl, color='gray', lw=2, linestyle='--', label='Val')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        if col == 0:
            ax.set_ylabel('MSE Loss')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        ax.plot(d_plot, obs_cfs_test[plot_slice, basin_idx], color='navy', lw=1.0, label='Observed', alpha=0.9)
        ax.plot(d_plot, preds[plot_slice, basin_idx], color=c, lw=1.0, label='Predicted', alpha=0.85)
        ax.set_title(f"Mean NSE = {nse:.3f}", fontweight='bold')
        if col == 0:
            ax.set_ylabel('Streamflow (ft3/s)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=30, labelsize=8)

    fig.suptitle('Underfitting  <-  Sweet Spot -> Overfitting', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_feature_engineering_comparison(
    dates_test: pd.DatetimeIndex,
    obs_cfs_test: np.ndarray,
    pred_baseline: np.ndarray,
    pred_feature_engineered: np.ndarray,
    gage_ids: np.ndarray,
    nse_baseline: dict,
    nse_feature_engineered: dict,
    basin_idx: int = 0,
    zoom: slice = slice(-548, None),
) -> None:
    """Compare baseline and feature-engineered LSTM predictions for one basin.

    Parameters
    ----------
    dates_test
        Date index for the test period.
    obs_cfs_test
        Observed streamflow in ft3/s, shape (time, basins).
    pred_baseline
        Baseline model predictions in ft3/s, shape (time, basins).
    pred_feature_engineered
        Feature-engineered model predictions in ft3/s, shape (time, basins).
    gage_ids
        Array of gage identifiers, one per basin.
    nse_baseline
        Mapping of gage ID to baseline NSE score.
    nse_feature_engineered
        Mapping of gage ID to feature-engineered NSE score.
    basin_idx
        Index of the basin to plot.
    zoom
        Slice selecting the zoomed subplot period (default: last 18 months).
    """
    gid = gage_ids[basin_idx]

    baseline_nse = nse_baseline[gid]
    fe_nse = nse_feature_engineered[gid]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)

    ax = axes[0]
    ax.plot(dates_test, obs_cfs_test[:, basin_idx], color='navy', lw=0.8, label='Observed')
    ax.plot(
        dates_test,
        pred_baseline[:, basin_idx],
        color='tomato',
        lw=0.8,
        alpha=0.75,
        label=f"Baseline LSTM (NSE={baseline_nse:.2f})",
    )
    ax.plot(
        dates_test,
        pred_feature_engineered[:, basin_idx],
        color='seagreen',
        lw=0.8,
        alpha=0.75,
        label=f"Feature-engineered LSTM (NSE={fe_nse:.2f})",
    )
    ax.set_title(f"Full test period — Basin {gid}")
    ax.set_ylabel('Streamflow (ft3/s)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax = axes[1]
    ax.plot(dates_test[zoom], obs_cfs_test[zoom, basin_idx], color='navy', lw=1.0, label='Observed')
    ax.plot(
        dates_test[zoom],
        pred_baseline[zoom, basin_idx],
        color='tomato',
        lw=1.0,
        alpha=0.75,
        label='Baseline',
    )
    ax.plot(
        dates_test[zoom],
        pred_feature_engineered[zoom, basin_idx],
        color='seagreen',
        lw=1.0,
        alpha=0.75,
        label='Feature engineered',
    )
    ax.set_title('Zoom: last 18 months')
    ax.set_ylabel('Streamflow (ft3/s)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.show()


def plot_nse_comparison(
    nse_baseline: dict,
    nse_feature_engineered: dict,
) -> None:
    """Bar plot comparing basin-level NSE before and after feature engineering.

    Parameters
    ----------
    nse_baseline
        Mapping of gage ID to baseline NSE score.
    nse_feature_engineered
        Mapping of gage ID to feature-engineered NSE score.
    """
    gage_ids = list(nse_baseline.keys())
    baseline_vals = np.array([nse_baseline[gid] for gid in gage_ids], dtype=float)
    fe_vals = np.array([nse_feature_engineered[gid] for gid in gage_ids], dtype=float)

    x = np.arange(len(gage_ids))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width / 2, baseline_vals, width, label='Baseline')
    ax.bar(x + width / 2, fe_vals, width, label='Feature engineered')

    ax.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(gage_ids, rotation=45, ha='right')
    ax.set_ylabel('NSE')
    ax.set_title('Basin-level NSE: baseline vs. feature-engineered inputs')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_seasonal_bias_and_error_magnitude(
    dates_test: pd.DatetimeIndex,
    obs_cfs_test: np.ndarray,
    pred_base_cfs: np.ndarray,
) -> None:
    """Plot seasonal residual bias and error-vs-flow magnitude.

    Moved from Notebook 2, Cell 5. The plotting code is infrastructure; the
    teaching point is how to interpret the residual patterns.

    Parameters
    ----------
    dates_test
        Date index for the test period.
    obs_cfs_test
        Observed streamflow in ft3/s, shape (time, basins).
    pred_base_cfs
        Baseline model predictions in ft3/s, shape (time, basins).
    """
    # Seasonal bias: residuals by calendar month
    errors = pred_base_cfs - obs_cfs_test  # positive = over-prediction
    months = pd.DatetimeIndex(dates_test).month

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    monthly_errors = [errors[months == m, :].ravel() for m in range(1, 13)]
    monthly_errors = [e[~np.isnan(e)] for e in monthly_errors]

    axes[0].boxplot(monthly_errors, labels=list('JFMAMJJASOND'),
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6))
    axes[0].axhline(0, color='red', linewidth=1.5, linestyle='--')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Prediction error (ft^3/s)')
    axes[0].set_title('Seasonal Bias')
    axes[0].grid(alpha=0.3)

    obs_flat = obs_cfs_test.ravel()
    err_flat = errors.ravel()
    mask = ~np.isnan(obs_flat) & ~np.isnan(err_flat)
    axes[1].hexbin(np.log1p(obs_flat[mask]), err_flat[mask],
                   gridsize=40, cmap='Blues', mincnt=1)
    axes[1].axhline(0, color='red', linewidth=1.5, linestyle='--')
    axes[1].set_xlabel('log(1 + Observed streamflow)')
    axes[1].set_ylabel('Prediction error (ft3/s)')
    axes[1].set_title('Error vs. Flow Magnitude')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_flow_duration_curves(
    obs_cfs_test: np.ndarray,
    pred_base_cfs: np.ndarray,
    gage_ids: np.ndarray,
    nse_base: dict,
) -> None:
    """Plot flow duration curves for each basin.

    Parameters
    ----------
    obs_cfs_test
        Observed streamflow in ft3/s, shape (time, basins).
    pred_base_cfs
        Baseline model predictions in ft3/s, shape (time, basins).
    gage_ids
        Array of gage identifiers, one per basin.
    nse_base
        Mapping of gage ID to baseline NSE score.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, (ax, gid) in enumerate(zip(axes.ravel(), gage_ids)):
        obs = obs_cfs_test[:, i]
        pred = pred_base_cfs[:, i]
        mask = ~np.isnan(obs) & ~np.isnan(pred)

        obs_sorted = np.sort(obs[mask])[::-1]
        pred_sorted = np.sort(pred[mask])[::-1]
        ep = np.linspace(0, 1, len(obs_sorted))

        ax.semilogy(ep, obs_sorted, color='navy', lw=1.2, label='Observed')
        ax.semilogy(ep, pred_sorted, color='tomato', lw=1.2, linestyle='--', label='Predicted')
        ax.set_title(f"{gid}\nNSE={nse_base[gid]:.2f}", fontsize=9)
        ax.grid(alpha=0.2, which='both')
        if i == 0:
            ax.legend(fontsize=7)
        if i >= 5:
            ax.set_xlabel('Exceedance prob.', fontsize=8)
        if i % 5 == 0:
            ax.set_ylabel('Streamflow (ft3/s)', fontsize=8)

    fig.suptitle('Flow Duration Curves — Baseline', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_nse_vs_attributes(
    attributes: np.ndarray,
    attribute_names: list[str],
    gage_ids: np.ndarray,
    nse_base: dict,
    attr_keys: list[str],
) -> None:
    """Plot basin-level NSE against selected static basin attributes.

    Parameters
    ----------
    attributes
        Static attribute array of shape (basins, n_attrs).
    attribute_names
        Ordered list of attribute names corresponding to columns in attributes.
    gage_ids
        Array of gage identifiers, one per basin.
    nse_base
        Mapping of gage ID to baseline NSE score.
    attr_keys
        Names of the attributes to plot (must be present in attribute_names).
    """
    nse_arr = np.array(list(nse_base.values()))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key in zip(axes.ravel(), attr_keys):
        attr_vals = attributes[:, attribute_names.index(key)]
        r, p = stats.pearsonr(attr_vals, nse_arr)
        ax.scatter(attr_vals, nse_arr, color='steelblue', s=60, zorder=3)
        for j, gid in enumerate(gage_ids):
            ax.annotate(str(gid)[-4:], (attr_vals[j], nse_arr[j]),
                        fontsize=7, ha='center', va='bottom')
        ax.set_xlabel(key, fontsize=9)
        ax.set_ylabel('NSE', fontsize=9)
        ax.set_title(f"r = {r:.2f}  (p={p:.2f})", fontsize=9)
        ax.axhline(0, color='gray', lw=0.8, linestyle='--')
        ax.grid(alpha=0.3)

    fig.suptitle('NSE vs. Basin Attributes — Where Does the Baseline Struggle?',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    all_models: dict[str, dict],
    gage_ids: np.ndarray,
    save_path: str | None = 'model_comparison.png',
) -> None:
    """Plot mean NSE and per-basin NSE for all model variants.

    Parameters
    ----------
    all_models
        Mapping of model name to a dict of {gage_id: nse_value} scores.
    gage_ids
        Array of gage identifiers, one per basin.
    save_path
        File path to save the figure, or None to skip saving.
    """
    mean_nse = {name: np.nanmean(list(scores.values())) for name, scores in all_models.items()}
    colors = ['#7f8c8d', '#2980b9', '#27ae60', '#8e44ad']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bars = axes[0].bar(mean_nse.keys(), mean_nse.values(), color=colors, alpha=0.85)
    axes[0].axhline(0, color='k', lw=0.8)
    axes[0].set_ylabel('Mean NSE (test period)')
    axes[0].set_title('Mean NSE — All Variants')
    axes[0].set_ylim(min(mean_nse.values()) - 0.1, 1.0)
    for bar, val in zip(bars, mean_nse.values()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha='center', fontsize=10, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')

    for (name, nse_d), c in zip(all_models.items(), colors):
        vals = [nse_d[gid] for gid in gage_ids]
        axes[1].scatter(range(len(gage_ids)), vals, label=name, color=c, s=60, zorder=3)
        axes[1].plot(range(len(gage_ids)), vals, color=c, lw=1, alpha=0.6)

    axes[1].set_xticks(range(len(gage_ids)))
    axes[1].set_xticklabels([str(g) for g in gage_ids], rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('NSE')
    axes[1].set_title('Per-Basin NSE — All Variants')
    axes[1].axhline(0, color='gray', lw=0.8, linestyle='--')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_validation_loss_comparison(lc_data: list[tuple]) -> None:
    """Plot validation loss curves for all model variants.

    Parameters
    ----------
    lc_data
        List of tuples, one per variant: (name, val_losses, color).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, vl, c in lc_data:
        ax.plot(vl, color=c, lw=2, label=name)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MSE Loss')
    ax.set_title('Validation Loss — All Variants')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
