"""
LMSP-MLP: Lag-aware Multi-task Sparse Patch MLP
================================================
面向综合能源预测的时滞感知多任务稀疏补丁 MLP 模型

Architecture Overview
---------------------
Input:
    endo_hist  (B, L, 3)      — KW, CHWTON, HTmmBTU historical loads
    exo_hist   (B, L, n_exo)  — weather + calendar exogenous variables
    time feats (B, L)         — hour, month, is_weekdays, is_holidays

Pipeline:
    1.  RevIN  — per-sample normalization of endo_hist
    2.  Series Decomposition  — seasonal + trend
    3a. Patch Branch  — seasonal → local pattern features (B, 3, d_model)
    3b. Sparse Branch — trend   → cross-period trend prediction (B, 3, H)
    4.  Lag-aware Exo Mixer — time-shifted weather features (B, 3, d_exo)
    5.  Directional Coupling Layer
            cool/heat → electricity  (strong)
            electricity → cool/heat  (weak, λ-scaled)
    6.  Seasonal Decoder  — (patch_feat + exo_ctx) → (B, 3, H)
    7.  Adaptive Fusion   — α·trend + β·season   (learned per task)
    8.  RevIN denorm

Contributions (论文创新点):
    C1. Pure MLP multi-task framework for joint ele/cool/heat forecasting
    C2. Lag-aware exogenous injection via lag bank + task-specific gates
    C3. Patch + Sparse dual-scale internal structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.revin       import RevIN
from .modules.decomp      import SeriesDecomp
from .modules.patch_branch  import PatchBranch
from .modules.sparse_branch import SparseBranch
from .modules.lag_exo_mixer import LagAwareExoMixer


# ---------------------------------------------------------------------------
# Directional Coupling Layer
# ---------------------------------------------------------------------------

class DirectionalCouplingLayer(nn.Module):
    """
    Asymmetric cross-task coupling (论文 §4.5):

        ẽ_ele  = h_ele  + W_{c→e}(h_cool) + W_{h→e}(h_heat)
        ẽ_cool = h_cool + λ_ec · W_{e→c}(h_ele)
        ẽ_heat = h_heat + λ_eh · W_{e→h}(h_ele)

    λ_ec, λ_eh are small learnable scalars (initialized ~0.1).
    """

    def __init__(self, d_feat: int, dropout: float = 0.1):
        super().__init__()
        d = d_feat
        # Strong: cool/heat → electricity
        self.W_c2e = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.W_h2e = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        # Weak: electricity → cool/heat (scaled by small λ)
        self.W_e2c = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.W_e2h = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        # Learnable coupling strength for feedback (start small)
        self.lambda_ec = nn.Parameter(torch.tensor(0.1))
        self.lambda_eh = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, 3, d_feat)  — [h_ele, h_cool, h_heat]
        Returns:
            h_coupled: (B, 3, d_feat)
        """
        h_ele, h_cool, h_heat = h[:, 0], h[:, 1], h[:, 2]

        h_ele_new  = h_ele  + self.W_c2e(h_cool) + self.W_h2e(h_heat)
        h_cool_new = h_cool + torch.clamp(self.lambda_ec, 0, 1) * self.W_e2c(h_ele)
        h_heat_new = h_heat + torch.clamp(self.lambda_eh, 0, 1) * self.W_e2h(h_ele)

        coupled = torch.stack([h_ele_new, h_cool_new, h_heat_new], dim=1)  # (B,3,d)
        return self.norm(coupled)


# ---------------------------------------------------------------------------
# Seasonal Decoder (patch features + exo context → prediction)
# ---------------------------------------------------------------------------

class SeasonalDecoder(nn.Module):
    """
    Per-task decoder that fuses patch features and exogenous context,
    then maps to pred_len.

      input: (B, d_model + d_exo)
      → Linear → GELU → Dropout → Linear → (B, pred_len)
    """

    def __init__(self, d_model: int, d_exo: int, pred_len: int, dropout: float = 0.1):
        super().__init__()
        d_in = d_model + d_exo
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in, pred_len),
        )

    def forward(self, patch_feat: torch.Tensor, exo_ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_feat: (B, d_model)
            exo_ctx:    (B, d_exo)
        Returns:
            (B, pred_len)
        """
        x = torch.cat([patch_feat, exo_ctx], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class LMSP_MLP(nn.Module):
    """
    LMSP-MLP: Lag-aware Multi-task Sparse Patch MLP.

    Args:
        input_len      : look-back window length L
        pred_len       : forecast horizon H
        d_model        : internal model dimension for patch/attention representations
        d_exo          : dimension of exo context per task
        n_exo          : number of exogenous features
        patch_size     : patch size for the Patch branch (e.g. 24 = daily)
        sparse_periods : list of periods for the Sparse branch [24, 168]
        sparse_d_model : MLP hidden dim inside SparseBranch
        n_patch_layers : number of MLP layers in PatchBranch
        lag_hours      : lag offsets (h) for the exo lag bank
        decomp_kernel  : moving-average kernel size for series decomposition
        dropout        : dropout rate
    """

    def __init__(
        self,
        input_len:      int   = 168,
        pred_len:       int   = 24,
        d_model:        int   = 256,
        d_exo:          int   = 64,
        n_exo:          int   = 6,
        patch_size:     int   = 24,
        sparse_periods: list  = None,
        sparse_d_model: int   = 128,
        n_patch_layers: int   = 2,
        lag_hours:      list  = None,
        decomp_kernel:  int   = 25,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len  = pred_len
        self.d_model   = d_model
        self.d_exo     = d_exo
        self.n_vars    = 3   # KW, CHWTON, HTmmBTU

        if sparse_periods is None:
            sparse_periods = [24, 168]
        if lag_hours is None:
            lag_hours = [0, 1, 2, 3, 6, 12, 24]

        # ── 1. RevIN ──────────────────────────────────────────────────────
        self.revin = RevIN(num_features=3, eps=1e-5, affine=False)

        # ── 2. Series Decomposition ────────────────────────────────────────
        self.decomp = SeriesDecomp(kernel_size=decomp_kernel)

        # ── 3a. Patch Branch (seasonal) ────────────────────────────────────
        self.patch_branch = PatchBranch(
            input_len  = input_len,
            d_model    = d_model,
            patch_size = patch_size,
            n_vars     = self.n_vars,
            n_layers   = n_patch_layers,
            dropout    = dropout,
        )

        # ── 3b. Sparse Branch (trend → direct pred) ────────────────────────
        self.sparse_branch = SparseBranch(
            input_len = input_len,
            pred_len  = pred_len,
            periods   = sparse_periods,
            d_model   = sparse_d_model,
            n_vars    = self.n_vars,
            dropout   = dropout,
        )

        # ── 4. Lag-aware Exo Mixer ─────────────────────────────────────────
        self.exo_mixer = LagAwareExoMixer(
            n_exo     = n_exo,
            d_exo     = d_exo,
            lag_hours = lag_hours,
            n_tasks   = self.n_vars,
            dropout   = dropout,
        )

        # ── 5. Directional Coupling ────────────────────────────────────────
        d_feat = d_model + d_exo
        self.coupling = DirectionalCouplingLayer(d_feat=d_feat, dropout=dropout)

        # ── 6. Seasonal Decoders (one per task) ───────────────────────────
        self.season_decoders = nn.ModuleList([
            SeasonalDecoder(d_model, d_exo, pred_len, dropout)
            for _ in range(self.n_vars)
        ])

        # ── 7. Adaptive Fusion weights (α, β per task) ─────────────────────
        # raw logits → softmax → [α_trend, β_season] sums to 1
        # shape: (n_vars, 2)
        self.fusion_logits = nn.Parameter(torch.zeros(self.n_vars, 2))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fuse(self, trend: torch.Tensor, season: torch.Tensor) -> torch.Tensor:
        """
        Adaptive fusion of trend and seasonal predictions.
        Args:
            trend  : (B, n_vars, H)
            season : (B, n_vars, H)
        Returns:
            fused  : (B, n_vars, H)
        """
        # fusion weights per task: (n_vars, 2) → softmax → (n_vars, 2)
        w = torch.softmax(self.fusion_logits, dim=-1)  # (3, 2)
        alpha = w[:, 0].view(1, self.n_vars, 1)         # (1, 3, 1)
        beta  = w[:, 1].view(1, self.n_vars, 1)         # (1, 3, 1)
        return alpha * trend + beta * season

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: dict with keys
                'endo_hist'   : (B, L, 3)
                'exo_hist'    : (B, L, n_exo)
                'hour'        : (B, L)
                'month'       : (B, L)
                'is_weekdays' : (B, L)
                'is_holidays' : (B, L)
        Returns:
            {'predictions': (B, H, 3)}
        """
        endo_hist = batch['endo_hist']    # (B, L, 3)
        exo_hist  = batch['exo_hist']     # (B, L, n_exo)

        # ── 1. RevIN normalize ────────────────────────────────────────────
        endo_norm = self.revin(endo_hist, mode='norm')     # (B, L, 3)

        # ── 2. Decompose into seasonal + trend ────────────────────────────
        seasonal, trend = self.decomp(endo_norm)            # both (B, L, 3)

        # ── 3a. Patch branch on seasonal ──────────────────────────────────
        patch_feat = self.patch_branch(seasonal)            # (B, 3, d_model)

        # ── 3b. Sparse branch on trend → direct trend prediction ──────────
        trend_pred = self.sparse_branch(trend)              # (B, 3, H)

        # ── 4. Lag-aware exo mixing ───────────────────────────────────────
        exo_ctx = self.exo_mixer(exo_hist)                  # (B, 3, d_exo)

        # ── 5. Concat patch + exo, apply directional coupling ────────────
        # Combined per-task feature: (B, 3, d_model + d_exo)
        combined = torch.cat([patch_feat, exo_ctx], dim=-1)  # (B, 3, d_feat)
        combined = self.coupling(combined)                   # (B, 3, d_feat)

        # ── 6. Seasonal decoders ──────────────────────────────────────────
        # Split combined back into patch + exo dims for each task decoder
        season_preds = []
        for v, dec in enumerate(self.season_decoders):
            p_feat  = combined[:, v, : self.d_model]        # (B, d_model)
            e_ctx   = combined[:, v, self.d_model :]        # (B, d_exo)
            season_preds.append(dec(p_feat, e_ctx))          # (B, H)
        season_pred = torch.stack(season_preds, dim=1)       # (B, 3, H)

        # ── 7. Adaptive fusion ────────────────────────────────────────────
        fused = self._fuse(trend_pred, season_pred)           # (B, 3, H)

        # ── 8. Rearrange to (B, H, 3) and RevIN denorm ───────────────────
        predictions = fused.permute(0, 2, 1)                 # (B, H, 3)
        predictions = self.revin(predictions, mode='denorm') # (B, H, 3)

        return {'predictions': predictions}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total':        total,
            'trainable':    trainable,
            'total_M':      total     / 1e6,
            'trainable_M':  trainable / 1e6,
        }
