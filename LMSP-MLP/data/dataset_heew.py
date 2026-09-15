"""
dataset_heew.py  —  Dataset for HEEW supplemental experiment.

Key differences vs dataset.py (main experiment):
  - Z-score normalization on BOTH endo and exo features (like ASHRAE).
  - Train-set statistics (mean/std) are passed to val/test splits
    so there is no data leakage.
  - __getitem__ returns raw (un-normalized) endo_future as well,
    so that metrics can be computed in the original physical units.
  - HEEW uses 'Precip' instead of 'Wind Direction' and 'Precipitable Water'.

This file is intentionally separate from dataset.py to avoid any
interference with the main Arizona IES experiment.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


class HEEWDataset(Dataset):
    """
    IES Dataset for HEEW benchmark with Z-score normalisation.

    Endogenous  (3):  KW, CHWTON, HTmmBTU
    Exogenous   (up to 8):
        Raw     (6):  Temperature, Dew Point, Relative Humidity,
                      Pressure, Wind Speed, Precip
        Derived (2):  Heat Comfort Index, Building Heat Loss Coefficient
    Time feats  (4):  Hour, Month, is_weekdays, is_holidays
    """

    ENDO_COLS = ['KW', 'CHWTON', 'HTmmBTU']
    EXO_RAW   = [
        'Temperature', 'Dew Point', 'Relative Humidity',
        'Pressure', 'Wind Speed', 'Precip',
    ]
    TIME_COLS = ['Hour', 'Month', 'is_weekdays', 'is_holidays']

    def __init__(
        self,
        data_path:               str,
        split:                   str   = 'train',
        input_len:               int   = 168,
        pred_len:                int   = 24,
        use_feature_engineering: bool  = True,
        stats:                   dict  = None,   # pass train stats to val/test
    ):
        self.input_len             = input_len
        self.pred_len              = pred_len
        self.use_feature_engineering = use_feature_engineering

        # ── Load CSV ──────────────────────────────────────────────────────
        df = pd.read_csv(data_path)
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        # ── Train / Val / Test split (70 / 15 / 15) ──────────────────────
        total   = len(df)
        n_train = int(total * 0.70)
        n_val   = int(total * 0.15)

        if split == 'train':
            df = df.iloc[:n_train]
        elif split == 'val':
            df = df.iloc[n_train: n_train + n_val]
        elif split == 'test':
            df = df.iloc[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        df = df.reset_index(drop=True)

        # ── Endogenous (raw, before normalisation) ────────────────────────
        endo_raw = df[self.ENDO_COLS].values.astype(np.float32)

        # ── Exogenous ─────────────────────────────────────────────────────
        avail_raw = [c for c in self.EXO_RAW if c in df.columns]
        exo_raw   = df[avail_raw].values.astype(np.float32)

        if use_feature_engineering and 'Temperature' in df.columns:
            T  = df['Temperature'].values.astype(np.float32)
            RH = df['Relative Humidity'].values.astype(np.float32) \
                 if 'Relative Humidity' in df.columns else np.full_like(T, 50.0)
            WS = df['Wind Speed'].values.astype(np.float32) \
                 if 'Wind Speed' in df.columns else np.zeros_like(T)

            heat_comfort = T - 0.55 * (1 - RH / 100.0) * (T - 14.5)
            heat_loss    = (18.0 - T) * (1.0 + 0.05 * WS)

            exo_full = np.column_stack([
                exo_raw,
                heat_comfort.reshape(-1, 1),
                heat_loss.reshape(-1, 1),
            ]).astype(np.float32)
        else:
            exo_full = exo_raw

        self.n_exo = exo_full.shape[1]

        # ── Z-score normalisation ─────────────────────────────────────────
        if stats is None:
            # Compute stats from THIS split (should only be called for train)
            endo_mean = endo_raw.mean(axis=0)
            endo_std  = endo_raw.std(axis=0) + 1e-8
            exo_mean  = exo_full.mean(axis=0)
            exo_std   = exo_full.std(axis=0) + 1e-8
            self.stats = {
                'endo_mean': endo_mean.tolist(),
                'endo_std':  endo_std.tolist(),
                'exo_mean':  exo_mean.tolist(),
                'exo_std':   exo_std.tolist(),
            }
        else:
            # Use provided stats (from train split) — no leakage
            self.stats   = stats
            endo_mean = np.array(stats['endo_mean'], dtype=np.float32)
            endo_std  = np.array(stats['endo_std'],  dtype=np.float32)
            exo_mean  = np.array(stats['exo_mean'],  dtype=np.float32)
            exo_std   = np.array(stats['exo_std'],   dtype=np.float32)

        if stats is None:
            # Re-read stats from self.stats now that they're set
            endo_mean = np.array(self.stats['endo_mean'], dtype=np.float32)
            endo_std  = np.array(self.stats['endo_std'],  dtype=np.float32)
            exo_mean  = np.array(self.stats['exo_mean'],  dtype=np.float32)
            exo_std   = np.array(self.stats['exo_std'],   dtype=np.float32)

        # Normalised arrays
        self.endo_norm = (endo_raw - endo_mean) / endo_std
        self.exo_norm  = (exo_full - exo_mean)  / exo_std

        # Keep raw endo so we can denormalise predictions for metric calc
        self.endo_raw  = endo_raw

        # Store denorm constants as tensors for fast __getitem__
        self.endo_mean_t = torch.from_numpy(endo_mean)  # (3,)
        self.endo_std_t  = torch.from_numpy(endo_std)   # (3,)

        # ── Time features ─────────────────────────────────────────────────
        avail_time     = [c for c in self.TIME_COLS if c in df.columns]
        self.time_data = df[avail_time].values.astype(np.float32)

        # ── Valid sample count ────────────────────────────────────────────
        self.valid_len = len(df) - input_len - pred_len + 1
        assert self.valid_len > 0, \
            f"[{split}] Not enough samples: {len(df)} rows, " \
            f"need at least {input_len + pred_len}"

        print(f"[{split.upper():5s}] rows={len(df):6d}, "
              f"valid_sequences={self.valid_len:6d}, n_exo={self.n_exo}  "
              f"(normalised)")

    def __len__(self) -> int:
        return self.valid_len

    def __getitem__(self, idx: int) -> dict:
        sl_hist = slice(idx, idx + self.input_len)
        sl_fut  = slice(idx + self.input_len, idx + self.input_len + self.pred_len)

        sample = {
            # Normalised inputs for the model
            'endo_hist':       torch.from_numpy(self.endo_norm[sl_hist]),   # (L, 3)
            'exo_hist':        torch.from_numpy(self.exo_norm[sl_hist]),    # (L, n_exo)
            # Normalised future endo — used as training target (loss in norm space)
            'endo_future':     torch.from_numpy(self.endo_norm[sl_fut]),    # (H, 3)
            # Raw (un-normalised) future endo — used for metric computation
            'endo_future_raw': torch.from_numpy(self.endo_raw[sl_fut]),     # (H, 3)
            # Denorm constants — so we can recover physical units from predictions
            'endo_mean':       self.endo_mean_t.clone(),                    # (3,)
            'endo_std':        self.endo_std_t.clone(),                     # (3,)
        }

        # Time features
        if self.time_data.shape[1] > 0:
            sample['hour']        = torch.from_numpy(self.time_data[sl_hist, 0])
        else:
            sample['hour']        = torch.zeros(self.input_len)

        if self.time_data.shape[1] > 1:
            sample['month']       = torch.from_numpy(self.time_data[sl_hist, 1])
        else:
            sample['month']       = torch.ones(self.input_len)

        if self.time_data.shape[1] > 2:
            sample['is_weekdays'] = torch.from_numpy(self.time_data[sl_hist, 2])
        else:
            sample['is_weekdays'] = torch.ones(self.input_len)

        if self.time_data.shape[1] > 3:
            sample['is_holidays'] = torch.from_numpy(self.time_data[sl_hist, 3])
        else:
            sample['is_holidays'] = torch.zeros(self.input_len)

        return sample


def create_heew_dataloaders(
    data_path:               str,
    batch_size:              int  = 32,
    input_len:               int  = 168,
    pred_len:                int  = 24,
    use_feature_engineering: bool = True,
    num_workers:             int  = 0,
):
    """
    Create train / val / test DataLoaders for HEEW benchmark.
    Stats are computed on train split and shared to val/test (no leakage).
    Returns (train_loader, val_loader, test_loader, n_exo, train_stats)
    """
    kwargs = dict(
        data_path               = data_path,
        input_len               = input_len,
        pred_len                = pred_len,
        use_feature_engineering = use_feature_engineering,
    )

    # Train — compute normalisation stats here
    train_ds = HEEWDataset(split='train', stats=None, **kwargs)
    # Val & test — use train stats (no leakage)
    val_ds   = HEEWDataset(split='val',   stats=train_ds.stats, **kwargs)
    test_ds  = HEEWDataset(split='test',  stats=train_ds.stats, **kwargs)

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, train_ds.n_exo, train_ds.stats
