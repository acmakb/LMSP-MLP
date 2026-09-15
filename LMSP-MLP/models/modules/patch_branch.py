"""
Patch Branch
Inspired by PatchMLP:
  - Patch the seasonal component into fixed-size windows
  - Temporal MLP across patch dimension
  - Light variable-mixing MLP across the 3 load channels
  - Output: pooled patch representation per variable
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Linear projection of non-overlapping patches."""

    def __init__(self, patch_size: int, d_model: int, input_len: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = input_len // patch_size
        self.proj = nn.Linear(patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L)  — single variable time series
        Returns:
            (B, n_patches, d_model)
        """
        B, L = x.shape
        # Trim to multiple of patch_size
        x = x[:, : self.n_patches * self.patch_size]
        # Reshape → patches
        x = x.reshape(B, self.n_patches, self.patch_size)   # (B, P, ps)
        # Project + norm
        return self.norm(self.proj(x))                        # (B, P, d_model)


class TemporalMLP(nn.Module):
    """MLP mixing across the patch (temporal) dimension."""

    def __init__(self, n_patches: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(n_patches, n_patches * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_patches * 2, n_patches),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_patches, d_model)"""
        # Mix along patch dim
        y = self.ff(x.transpose(1, 2)).transpose(1, 2)   # (B, P, d_model)
        return self.norm(x + y)


class ChannelMixer(nn.Module):
    """Light MLP mixing across the 3 load variables (channel dim)."""

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(n_vars, n_vars),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_vars, n_patches, d_model)"""
        # Permute so last dim = n_vars, then mix
        y = x.permute(0, 2, 3, 1)          # (B, P, d_model, n_vars)
        y = self.ff(y)                       # (B, P, d_model, n_vars)
        y = y.permute(0, 3, 1, 2)           # (B, n_vars, P, d_model)
        return self.norm(x + y)


# ---------------------------------------------------------------------------
# Patch Branch
# ---------------------------------------------------------------------------

class PatchBranch(nn.Module):
    """
    Full Patch branch.

    Pipeline:
        seasonal (B, L, n_vars)
        → per-var PatchEmbedding → (B, n_vars, P, d_model)
        → n_layers of [TemporalMLP → ChannelMixer]
        → mean-pool over patches → (B, n_vars, d_model)
    """

    def __init__(
        self,
        input_len:  int,
        d_model:    int,
        patch_size: int   = 24,
        n_vars:     int   = 3,
        n_layers:   int   = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        n_patches = input_len // patch_size

        self.patch_embed = PatchEmbedding(patch_size, d_model, input_len)

        self.temporal_mlps = nn.ModuleList(
            [TemporalMLP(n_patches, d_model, dropout) for _ in range(n_layers)]
        )
        self.channel_mixer = ChannelMixer(n_vars, d_model, dropout)
        self.out_norm = nn.LayerNorm(d_model)
        self.n_vars = n_vars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, n_vars)   — seasonal component
        Returns:
            (B, n_vars, d_model)
        """
        B, L, V = x.shape

        # Per-variable embedding
        patches = torch.stack(
            [self.patch_embed(x[:, :, v]) for v in range(V)],
            dim=1,
        )  # (B, V, P, d_model)

        # Temporal mixing  (operate on V*B, P, d)
        B_, V_, P, D = patches.shape
        feat = patches.reshape(B_ * V_, P, D)
        for t_mlp in self.temporal_mlps:
            feat = t_mlp(feat)
        patches = feat.reshape(B_, V_, P, D)

        # Channel mixing
        patches = self.channel_mixer(patches)   # (B, V, P, d_model)

        # Pool across patches
        out = patches.mean(dim=2)               # (B, V, d_model)
        return self.out_norm(out)
