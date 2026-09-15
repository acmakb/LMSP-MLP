"""
Sparse Branch
Inspired by SparseTSF (Cross-Period Sparse Forecasting):
  - Downsample the trend sequence by fixed periods (24h, 168h)
  - Apply 1-D convolution for within-period aggregation
  - MLP across the cross-period dimension → direct trend prediction
  - Two-period outputs are fused via learned weights

Design choice:
  Each SparsePeriodForecaster outputs (B, n_vars, pred_len) directly,
  so the Sparse branch provides the trend prediction without a separate
  linear decoder — consistent with SparseTSF's approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsePeriodForecaster(nn.Module):
    """
    Cross-period forecasting for one period length w.

    Steps:
        1. Subtract per-sequence mean (light normalization)
        2. Conv1d to aggregate context within each period
        3. Reshape: (B*V, L) → (B*V, w, seg_x)  where seg_x = L//w
        4. MLP: seg_x → seg_y  (cross-period map)
        5. Reshape back to (B, V, pred_len)  [tile/trim as needed]
        6. Add mean back
    """

    def __init__(
        self,
        input_len: int,
        pred_len:  int,
        period:    int,
        d_model:   int,
        n_vars:    int   = 3,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.period   = period
        self.pred_len = pred_len
        self.n_vars   = n_vars

        self.seg_x = input_len // period                    # segments in input
        self.seg_y = max(1, (pred_len + period - 1) // period)  # ceil

        # Conv1d for within-period context aggregation
        ksize = 1 + 2 * (period // 2)
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=ksize, stride=1,
            padding=period // 2,
            padding_mode='zeros', bias=False,
        )

        # Cross-period MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.seg_x, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.seg_y),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, n_vars)  — trend component
        Returns:
            (B, n_vars, pred_len)
        """
        B, L, V = x.shape
        w        = self.period
        seg_x    = self.seg_x
        seg_y    = self.seg_y

        # --- Instance normalization (subtract mean) ---
        seq_mean = x.mean(dim=1, keepdim=True)              # (B, 1, V)
        x = x - seq_mean

        # Flatten: (B*V, L)
        x = x.permute(0, 2, 1).reshape(B * V, L)

        # Conv aggregation: slide over the full sequence
        x_conv = self.conv(x.unsqueeze(1)).squeeze(1) + x  # (B*V, L)

        # --- Sparse reshape ---
        # Trim to multiple of w
        trim_len = seg_x * w
        x_ds = x_conv[:, :trim_len].reshape(B * V, seg_x, w)  # (B*V, seg_x, w)
        x_ds = x_ds.permute(0, 2, 1)                           # (B*V, w, seg_x)

        # Cross-period MLP: (B*V, w, seg_x) → (B*V, w, seg_y)
        y = self.mlp(x_ds)                                  # (B*V, w, seg_y)

        # --- Upsample back ---
        y = y.permute(0, 2, 1).reshape(B * V, seg_y * w)   # (B*V, seg_y*w)

        # Trim / pad to pred_len
        if y.shape[1] >= self.pred_len:
            y = y[:, : self.pred_len]
        else:
            y = F.pad(y, (0, self.pred_len - y.shape[1]), mode='replicate')

        # Restore variable dimension
        y = y.reshape(B, V, self.pred_len)                  # (B, V, pred_len)

        # Denormalize
        y = y + seq_mean.permute(0, 2, 1)                   # (B, V, pred_len)
        return y


class SparseBranch(nn.Module):
    """
    Two-period sparse branch (w=24, w=168) with learnable fusion.

    Both forecasters share the same pred_len and n_vars; their outputs
    are fused via a softmax-normalised scalar weight.
    """

    def __init__(
        self,
        input_len: int,
        pred_len:  int,
        periods:   list  = None,
        d_model:   int   = 128,
        n_vars:    int   = 3,
        dropout:   float = 0.1,
    ):
        super().__init__()
        if periods is None:
            periods = [24, 168]

        # Only include periods that are valid given input_len
        valid_periods = [p for p in periods if input_len >= p]
        assert valid_periods, "No valid periods for given input_len"

        self.forecasters = nn.ModuleList([
            SparsePeriodForecaster(input_len, pred_len, p, d_model, n_vars, dropout)
            for p in valid_periods
        ])
        n = len(self.forecasters)
        # Learnable log-weights (converted to probs via softmax)
        self.log_weights = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, n_vars)  — trend component
        Returns:
            (B, n_vars, pred_len)
        """
        outs = [f(x) for f in self.forecasters]          # list of (B,V,H)

        if len(outs) == 1:
            return outs[0]

        weights = torch.softmax(self.log_weights, dim=0)  # (n,)
        out = sum(w * o for w, o in zip(weights, outs))
        return out
