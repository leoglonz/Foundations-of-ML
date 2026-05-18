"""
Lightweight loader for the 10-basin CAMELS daymetv2 subset pickle.

Pickle structure (matches camels_daymetv2 format):
    tuple(
        forcings   : np.ndarray  (basins, time, n_forcings)
        target     : np.ndarray  (basins, time, 1)          -- streamflow ft³/s
        attributes : np.ndarray  (basins, n_attrs)
    )

Time axis spans 1980-10-01 to 2014-09-30 (12,418 daily steps).
"""

import pickle
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray


FORCING_NAMES = ["prcp", "tmean", "pet", "dayl", "srad", "vp"]

ATTRIBUTE_NAMES = [
    "p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    "elev_mean", "slope_mean", "area_gages2", "frac_forest", "lai_max",
    "lai_diff", "gvf_max", "gvf_diff", "dom_land_cover_frac", "dom_land_cover",
    "root_depth_50", "soil_depth_pelletier", "soil_depth_statsgo",
    "soil_porosity", "soil_conductivity", "max_water_content", "sand_frac",
    "silt_frac", "clay_frac", "geol_1st_class", "glim_1st_class_frac",
    "geol_2nd_class", "glim_2nd_class_frac", "carbonate_rocks_frac",
    "geol_porosity", "geol_permeability",
]

_ALL_DATES = pd.date_range("1980-10-01", "2014-09-30", freq="D")


class CamelsSubsetLoader:
    """Load and provide named access to the 10-basin CAMELS daymetv2 subset.

    Parameters
    ----------
    pickle_path
        Path to the subset pickle file.
    gage_id_path
        Path to the corresponding gage_id .npy file.
    start_date, end_date
        Optional date strings (``"YYYY-MM-DD"``) to slice the time axis.
    """

    def __init__(
        self,
        pickle_path: str,
        gage_id_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        with open(pickle_path, "rb") as f:
            forcings_raw, target_raw, attributes_raw = pickle.load(f)

        self.gage_ids: NDArray = np.load(gage_id_path, allow_pickle=True)
        self.dates: pd.DatetimeIndex = _ALL_DATES

        # Resolve time slice
        t0 = pd.Timestamp(start_date) if start_date else _ALL_DATES[0]
        t1 = pd.Timestamp(end_date) if end_date else _ALL_DATES[-1]
        idx_start = _ALL_DATES.get_loc(t0)
        idx_end = _ALL_DATES.get_loc(t1) + 1

        self.dates = _ALL_DATES[idx_start:idx_end]

        # Slice time axis and transpose to (time, basins, vars) to match HydroLoader
        self.forcings: NDArray = np.transpose(
            forcings_raw[:, idx_start:idx_end], (1, 0, 2)
        ).astype(np.float32)
        self.target: NDArray = np.transpose(
            target_raw[:, idx_start:idx_end], (1, 0, 2)
        ).astype(np.float32)
        self.attributes: NDArray = attributes_raw.astype(np.float32)

    # ------------------------------------------------------------------
    # Named access helpers
    # ------------------------------------------------------------------

    def get_forcing(self, name: str) -> NDArray:
        """Return a single forcing variable, shape ``(time, basins)``."""
        idx = FORCING_NAMES.index(name)
        return self.forcings[:, :, idx]

    def get_attribute(self, name: str) -> NDArray:
        """Return a single basin attribute, shape ``(basins,)``."""
        idx = ATTRIBUTE_NAMES.index(name)
        return self.attributes[:, idx]

    def as_dict(self) -> dict[str, NDArray]:
        """Return all data as a flat dictionary of named arrays.

        Returns
        -------
        dict with keys:
            - each forcing name  → shape (time, basins)
            - each attribute name → shape (basins,)
            - ``"streamflow"``   → shape (time, basins)  [ft³/s]
            - ``"dates"``        → pd.DatetimeIndex
            - ``"gage_ids"``     → 1-D int array
        """
        d: dict = {
            "dates": self.dates,
            "gage_ids": self.gage_ids,
            "streamflow": self.target[:, :, 0],
        }
        for i, name in enumerate(FORCING_NAMES):
            d[name] = self.forcings[:, :, i]
        for i, name in enumerate(ATTRIBUTE_NAMES):
            d[name] = self.attributes[:, i]
        return d

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_basins(self) -> int:
        return self.forcings.shape[1]

    @property
    def n_timesteps(self) -> int:
        return self.forcings.shape[0]

    def __repr__(self) -> str:
        return (
            f"CamelsSubsetLoader("
            f"basins={self.n_basins}, "
            f"timesteps={self.n_timesteps}, "
            f"dates={self.dates[0].date()} to {self.dates[-1].date()})"
        )


if __name__ == "__main__":
    import os

    base = os.path.dirname(__file__)
    loader = CamelsSubsetLoader(
        pickle_path=os.path.join(base, "camels_daymetv2_10basin_subset"),
        gage_id_path=os.path.join(base, "gage_id_10basin_subset.npy"),
        start_date="1999-10-01",
        end_date="2008-09-30",
    )

    print(loader)
    print(f"Gage IDs : {loader.gage_ids}")
    print(f"Forcings : {loader.forcings.shape}  (time, basins, vars)")
    print(f"Target   : {loader.target.shape}    (time, basins, 1)")
    print(f"Attrs    : {loader.attributes.shape} (basins, attrs)")

    prcp = loader.get_forcing("prcp")
    print(f"\nprcp mean per basin : {prcp.mean(axis=0).round(3)}")

    area = loader.get_attribute("area_gages2")
    print(f"basin area (km²)    : {area.round(1)}")
