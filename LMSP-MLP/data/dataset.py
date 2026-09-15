"""
Dataset for LMSP-MLP.
Copied and adapted from ies_forecasting/data/dataset_v8.py.
Adds full weather variables to exo features (7 raw + 2 derived = 9 channels).
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


class IESDataset(Dataset):
    """
    IES Dataset for LMSP-MLP.

    Endogenous  (3):  KW, CHWTON, HTmmBTU
    Exogenous   (up to 9):
        Raw     (7):  Temperature, Dew Point, Relative Humidity,
                      Pressure, Wind Speed, Wind Direction, Precipitable Water
        Derived (2):  Heat Comfort Index, Building Heat Loss Coefficient
    Time feats  (4):  Hour, Month, is_weekdays, is_holidays
                      (returned as separate batch keys, not in exo_hist)
    """

    ENDO_COLS = ['KW', 'CHWTON', 'HTmmBTU']
    EXO_RAW   = [
        'Temperature', 'Dew Point', 'Relative Humidity',
        'Pressure', 'Wind Speed', 'Wind Direction', 'Precipitable Water',
    ]
    TIME_COLS = ['Hour', 'Month', 'is_weekdays', 'is_holidays']

    def __init__(
        self,
        data_path:             str,
        split:                 str   = 'train',
        input_len:             int   = 168,
        pred_len:              int   = 24,
        use_feature_engineering: bool = True,
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

        # ── Endogenous ────────────────────────────────────────────────────
        self.endo_data = df[self.ENDO_COLS].values.astype(np.float32)

        # ── Exogenous ─────────────────────────────────────────────────────
        # Use only the raw cols that actually exist in the CSV
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

            self.exo_data = np.column_stack([
                exo_raw,
                heat_comfort.reshape(-1, 1),
                heat_loss.reshape(-1, 1),
            ]).astype(np.float32)
        else:
            self.exo_data = exo_raw

        self.n_exo = self.exo_data.shape[1]

        # ── Time features ─────────────────────────────────────────────────
        avail_time     = [c for c in self.TIME_COLS if c in df.columns]
        self.time_data = df[avail_time].values.astype(np.float32)

        # ── Valid sample count ────────────────────────────────────────────
        self.valid_len = len(df) - input_len - pred_len + 1
        assert self.valid_len > 0, \
            f"[{split}] Not enough samples: {len(df)} rows, " \
            f"need at least {input_len + pred_len}"

        print(f"[{split.upper():5s}] rows={len(df):6d}, "
              f"valid_sequences={self.valid_len:6d}, n_exo={self.n_exo}")

    def __len__(self) -> int:
        return self.valid_len

    def __getitem__(self, idx: int) -> dict:
        sl_hist = slice(idx, idx + self.input_len)
        sl_fut  = slice(idx + self.input_len, idx + self.input_len + self.pred_len)

        sample = {
            'endo_hist':   torch.from_numpy(self.endo_data[sl_hist]),    # (L, 3)
            'endo_future': torch.from_numpy(self.endo_data[sl_fut]),     # (H, 3)
            'exo_hist':    torch.from_numpy(self.exo_data[sl_hist]),     # (L, n_exo)
            'exo_future':  torch.from_numpy(self.exo_data[sl_fut]),      # (H, n_exo)
            'hour':        torch.from_numpy(self.time_data[sl_hist, 0]) if self.time_data.shape[1] > 0 else torch.zeros(self.input_len),
            'month':       torch.from_numpy(self.time_data[sl_hist, 1]) if self.time_data.shape[1] > 1 else torch.ones(self.input_len),
            'is_weekdays': torch.from_numpy(self.time_data[sl_hist, 2]) if self.time_data.shape[1] > 2 else torch.ones(self.input_len),
            'is_holidays': torch.from_numpy(self.time_data[sl_hist, 3]) if self.time_data.shape[1] > 3 else torch.zeros(self.input_len),
        }
        return sample


def create_dataloaders(
    data_path:               str,
    batch_size:              int   = 32,
    input_len:               int   = 168,
    pred_len:                int   = 24,
    use_feature_engineering: bool  = True,
    num_workers:             int   = 0,
):
    """Create train / val / test DataLoaders."""
    kwargs = dict(
        data_path               = data_path,
        input_len               = input_len,
        pred_len                = pred_len,
        use_feature_engineering = use_feature_engineering,
    )
    train_ds = IESDataset(split='train', **kwargs)
    val_ds   = IESDataset(split='val',   **kwargs)
    test_ds  = IESDataset(split='test',  **kwargs)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_ds.n_exo
