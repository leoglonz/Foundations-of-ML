# Foundations of Machine Learning
**CIROH Developer's Conference 2026**

Hands-on introduction to building, training, and diagnosing ML models for streamflow prediction using the [CAMELS](https://ral.ucar.edu/solutions/products/camels) dataset.

---

## Workshop notebooks

| Notebook | Topic | Key concepts |
|----------|-------|-------------|
| `01_ml_pipeline_streamflow.ipynb` | End-to-end ML pipeline | Data loading, preprocessing, LSTM, training, NSE evaluation, feature engineering |
| `02_model_augmentation.ipynb` | Diagnostics & architecture | Bias analysis, static attribute conditioning, deep LSTM, causal convolution |

---

## Repo structure

```
Foundations-of-ML/
├── tutorials/
│   ├── 01_ml_pipeline_streamflow.ipynb
│   ├── 02_model_augmentation.ipynb
│   ├── utils.py          # dataset, training, and evaluation helpers
│   └── plots.py          # all plotting functions
├── data/
│   ├── camels_daymetv2_subset    # 10-basin CAMELS pickle (1980–2014)
│   ├── gage_id_subset.npy        # gage IDs for the 10 basins
│   └── loc/                      # watershed boundary shapefiles
├── weights/                      # pre-trained model weights
├── camels_loader.py              # data loader for the CAMELS subset
└── pyproject.toml
```

---

## Setup

Requires Python 3.12+. Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Then open either notebook in JupyterLab or VS Code.

---

## Dataset

A curated 10-basin subset of CAMELS from the southern Appalachians (NC/VA), with:
- **12,418 daily timesteps** — 1980-10-01 through 2014-09-30
- **6 climate forcings** — precipitation, temperature, PET, day length, solar radiation, vapor pressure
- **1 target** — daily streamflow (ft³/s)
- **35 static basin attributes** — area, slope, soil, geology, land cover

Pre-trained weights are included so the notebooks run end-to-end without waiting for training.
