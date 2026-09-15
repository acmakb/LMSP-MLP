"""
Series Decomposition Module
Inspired by MDMLP-EIA and PatchMLP:
  - Moving-average trend extraction
  - Seasonal = original - trend
"""

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    """Symmetric moving-average smoothing (same length as input)."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C)
        Returns:
            smoothed: (B, L, C)  same length as input
        """
        pad = (self.kernel_size - 1) // 2
        # Pad both ends with edge values to preserve length
        front = x[:, :1, :].repeat(1, pad, 1)          # (B, pad, C)
        back  = x[:, -1:, :].repeat(1, pad, 1)         # (B, pad, C)
        x_padded = torch.cat([front, x, back], dim=1)  # (B, L+2*pad, C)
        # Apply avg pooling along time axis
        out = self.avg(x_padded.permute(0, 2, 1))       # (B, C, L)
        return out.permute(0, 2, 1)                      # (B, L, C)


class SeriesDecomp(nn.Module):
    """
    Decompose input into trend + seasonal.
      trend    = moving_average(x)
      seasonal = x - trend
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, L, C)
        Returns:
            seasonal: (B, L, C)
            trend:    (B, L, C)
        """
        trend    = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend
