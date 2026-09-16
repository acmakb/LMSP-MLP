# LMSP-MLP: Lag-aware Multi-task Sparse Patch MLP

> **面向综合能源系统的时滞感知多任务稀疏补丁 MLP 预测模型**
>
> A pure-MLP architecture for joint short-term forecasting of **electricity (KW)**, **cooling (CHWTON)**, and **heating (HTmmBTU)** loads in Integrated Energy Systems (IES).

---

## 📋 Overview

LMSP-MLP addresses the challenge of simultaneously forecasting three highly correlated energy loads under complex weather influences. It introduces three key architectural innovations:

| # | Contribution | Description |
|---|---|---|
| C1 | **Pure MLP Multi-task Framework** | Joint forecasting of electricity / cooling / heating without any attention or recurrence |
| C2 | **Lag-aware Exogenous Injection** | Lag bank + task-specific gating to model delayed weather–load responses |
| C3 | **Patch + Sparse Dual-scale Structure** | PatchBranch captures local seasonal patterns; SparseBranch handles multi-period trend prediction |

## 

---

## 📁 Project Structure

```
LMSP/
├── data/                                         # ← Place dataset here
│   └── 2019-2022-full-feature-final_dataset.csv
└── LMSP-MLP/
    ├── train.py                                  # Training entry point
    ├── evaluate.py                               # Evaluation & benchmarking
    ├── models/
    │   ├── lmsp_mlp.py                           # Main model (LMSP_MLP)
    │   └── modules/
    │       ├── revin.py                          # Reversible Instance Normalization
    │       ├── decomp.py                         # Series decomposition (moving average)
    │       ├── patch_branch.py                   # Patch MLP branch (seasonal)
    │       ├── sparse_branch.py                  # Sparse MLP branch (trend)
    │       └── lag_exo_mixer.py                  # Lag-aware exogenous mixer
    └── data/
        ├── dataset.py                            # IESDataset + create_dataloaders
        └── dataset_heew.py                       # Alternative dataset loader
```

---

## 🗂️ Dataset

This project uses the **ASU Integrated Energy System (IES) dataset** (2019–2022).

### Step 1 — Place the dataset file

Download the dataset and put the CSV file in the `data/` folder at the project root:

```
LMSP/
└── data/
    └── 2019-2022-full-feature-final_dataset.csv   ← required
```

### Step 2 — Required CSV columns

| Type | Columns |
|---|---|
| Endogenous (targets) | `KW`, `CHWTON`, `HTmmBTU` |
| Weather (exogenous) | `Temperature`, `Dew Point`, `Relative Humidity`, `Pressure`, `Wind Speed`, `Wind Direction`, `Precipitable Water` |
| Time features | `Hour`, `Month`, `is_weekdays`, `is_holidays` |
| Timestamp | `date` |

> **Data split**: 70% train / 15% validation / 15% test (chronological).
>
> Two additional weather features — **Heat Comfort Index** and **Building Heat Loss Coefficient** — are derived automatically when `use_feature_engineering=True` (default).

---

## ⚙️ Requirements

```bash
pip install torch numpy pandas matplotlib tqdm
# Optional — for FLOPs estimation in evaluate.py:
pip install thop
```

> Requires **Python ≥ 3.8** and **PyTorch ≥ 1.12**.

---

## 🚀 Quick Start

All scripts are run from within the `LMSP-MLP/` directory.

```bash
cd LMSP/LMSP-MLP
```

### 1. Train

```bash
# Default: 24-hour forecast horizon, 168-hour (1 week) look-back window
python train.py --pred_len 24

# 48-hour forecast with custom model size
python train.py --pred_len 48 --d_model 256 --epochs 100

# View all options
python train.py --help
```

**Key training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `../data/2019-2022-full-feature-final_dataset.csv` | Path to dataset CSV |
| `--input_len` | `168` | Look-back window length (hours) |
| `--pred_len` | `24` | Forecast horizon (hours) |
| `--d_model` | `256` | Model hidden dimension |
| `--d_exo` | `64` | Exogenous context dimension |
| `--patch_size` | `24` | Patch size (daily cycle) |
| `--batch_size` | `32` | Training batch size |
| `--epochs` | `100` | Maximum training epochs |
| `--lr` | `1e-4` | Learning rate (AdamW optimizer) |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--patience` | `15` | Early stopping patience |
| `--seed` | `42` | Random seed |

Checkpoints are saved to `checkpoints/lmsp_mlp_pred{H}_{timestamp}/`:

```
checkpoints/lmsp_mlp_pred24_20260916_103700/
├── best_model.pth       # Best checkpoint (lowest val MAE)
├── config.json          # Full experiment configuration
├── history.json         # Per-epoch loss & metrics
└── training_log.txt     # Plain-text training log
```

---

### 2. Evaluate

```bash
python evaluate.py --checkpoint_dir lmsp_mlp_pred24_20260916_103700
```

**Key evaluation arguments:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint_dir` | *(required)* | Experiment folder name under `checkpoints/` |
| `--data_path` | `../data/2019-2022-full-feature-final_dataset.csv` | Path to dataset CSV |
| `--batch_size` | `32` | Evaluation batch size |
| `--compute_flops` | `True` | Estimate FLOPs via `thop` (best-effort) |
| `--benchmark_latency` | `True` | Measure forward inference latency |
| `--latency_batch_size` | `1` | Batch size used for latency benchmark |
| `--latency_warmup` | `20` | Warmup iterations before timing |
| `--latency_iters` | `100` | Timed iterations for latency estimate |

Results are written to `checkpoints/<exp>/evaluation/` **and** `results/<exp>/test/`:

```
evaluation/
├── metrics.json            # All quantitative metrics (JSON)
├── evaluation_report.txt   # Human-readable evaluation report
├── predictions.npz         # Saved predictions & ground-truth arrays
├── summary.json            # Experiment summary
└── sample_1.png …          # Per-sample forecast visualizations
```

**Reported metrics** (per task and average):

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Square Error |
| sMAPE | Symmetric Mean Absolute Percentage Error |
| MAPE | Mean Absolute Percentage Error |
| Params (M) | Total parameter count |
| FLOPs (M) | Estimated floating-point operations (via `thop`) |
| Inference (ms/sample) | Mean forward-pass latency |

---

## 📄 Citation

If you find this code useful in your research, please cite:

```bibtex
@article{lmsp_mlp_2026,
  title   = {LMSP-MLP: Lag-aware Multi-task Sparse Patch MLP
             for Integrated Energy System Forecasting},
  author  = {},
  journal = {},
  year    = {2026},
}
```

---

## 📝 License

This project is released for research purposes.
