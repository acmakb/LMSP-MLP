"""
RevIN (Reversible Instance Normalization)
Copied from ies_forecasting/models/modules/revin.py
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization

    Normalizes each sample independently to zero mean and unit variance.
    Stores statistics for inverse transformation.

    Args:
        num_features: Number of features to normalize (3 for loads)
        eps: Small constant for numerical stability
        affine: Whether to learn affine parameters
    """

    def __init__(self, num_features, eps=1e-5, affine=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        """
        Args:
            x: (B, L, num_features) - input tensor
            mode: 'norm' for normalization, 'denorm' for denormalization
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return x

    def _get_statistics(self, x):
        self.mean = x.mean(dim=1, keepdim=True)   # (B, 1, C)
        self.std  = x.std(dim=1, keepdim=True)    # (B, 1, C)

    def _normalize(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * (self.std + self.eps) + self.mean
        return x
