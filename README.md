# Foundations of Machine Learning
**CIROH Developer's Conference 2026**

Hands-on introduction to building, training, and diagnosing ML models for streamflow prediction using a subset of the [CAMELS](https://ral.ucar.edu/solutions/products/camels) dataset.

---

## Workshop notebooks

| Notebook | Topic | Key concepts |
|----------|-------|-------------|
| `01_ml_pipeline_streamflow.ipynb` | End-to-end ML pipeline | Data loading, preprocessing, LSTM, training, NSE evaluation, feature engineering |
| `02_model_augmentation.ipynb` | Diagnostics & architecture | Bias analysis, static attribute conditioning, deep LSTM, causal convolution |

---

## Repo structure

```text
Foundations-of-ML/
├── tutorials/
│   ├── 01_ml_pipeline_streamflow.ipynb
│   ├── 02_model_augmentation.ipynb
│   ├── utils.py          # dataset, training, and evaluation helpers
│   └── plots.py          # all plotting functions
├── data/
│   ├── camels_daymetv2_subset    # CAMELS dataset
│   ├── gage_id_subset.npy        # gage IDs for the 10 basins
│   └── loc/                      # watershed boundary shapefiles
├── weights/                      # pre-trained model weights
└── camels_loader.py              # data loader for the CAMELS subset
```

---

## Setup

### 2i2c (During workshop only)

1. From a browser, login to [2i2c](https://workshop.ciroh.awi.2i2c.cloud). Passwords will be provided.
2. Launch a **Medium** server (~14 GB RAM, ~4 CPUs) with image **Foundations of ML**.
3. Open a new terminal and clone this repository;

    ```bash
    git clone https://github.com/leoglonz/Foundations-of-ML.git
    ```

4. Install Python dependencies;

    ```bash
    pip install -e Foundations-of-ML/
    ```

5. You should now be able to start using notebooks **01** and **02** provided in `tutorials/`.

### Local

1. Open a new terminal and clone this repository;

    ```bash
    git clone https://github.com/leoglonz/Foundations-of-ML.git
    ```

2. Create a new conda environment;

    ```bash
    conda create -n workshop python=3.11
    conda activate workshop
    ```

3. Install Python dependencies;

    ```bash
    pip install -e Foundations-of-ML/
    ```

4. You should now be able to start using notebooks **01** and **02** provided in `tutorials/`.

> `uv pip` may alternatively be used in place of `pip`.

---

## Dataset

10 medium-size basins selected from the CAMELS dataset:

- **12,418 daily timesteps** — 1980-10-01 through 2014-09-30
- **6 meteorologic forcings** — precipitation, temperature, PET, day length, solar radiation, vapor pressure
- **1 target** — daily streamflow (ft³/s)
- **35 static basin attributes** — area, slope, soil, geology, land cover

> Pre-trained weights are included so the notebooks can be run end-to-end without waiting for training.
