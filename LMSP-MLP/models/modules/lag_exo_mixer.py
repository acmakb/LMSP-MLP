"""
Lag-aware Exogenous Mixer
Inspired by TimeXer:
  "Exogenous variables should be treated separately from endogenous ones."

Novelty (论文贡献点 2):
  Instead of a simple concat, we build a *lag bank* for each weather variable
  (lags: 0, 1, 2, 3, 6, 12, 24 h) to capture delayed responses in
  cooling / heating / electricity demand.

  Each exogenous variable is encoded independently (variable-wise MLP),
  then task-specific gates produce three different exogenous contexts:
    - H_exo^ele  (electricity branch)
    - H_exo^cool (cooling branch)
    - H_exo^heat (heating branch)
"""

import torch
import torch.nn as nn


class LagAwareExoMixer(nn.Module):
    """
    Lag-aware Exogenous Variable Mixer.

    Args:
        n_exo       : number of exogenous channels
        d_exo       : output dimension per task
        lag_hours   : list of lag offsets (in hours) to sample from exo_hist
        n_tasks     : number of prediction tasks (3 for ele/cool/heat)
        dropout     : dropout rate
    """

    def __init__(
        self,
        n_exo:      int,
        d_exo:      int,
        lag_hours:  list  = None,
        n_tasks:    int   = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()
        if lag_hours is None:
            lag_hours = [0, 1, 2, 3, 6, 12, 24]
        self.lag_hours = lag_hours
        self.n_lags    = len(lag_hours)
        self.n_exo     = n_exo
        self.d_exo     = d_exo
        self.n_tasks   = n_tasks

        # Per-variable encoder: n_lags → d_exo  (shared across variables)
        self.var_encoder = nn.Sequential(
            nn.Linear(self.n_lags, d_exo),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.var_norm = nn.LayerNorm(d_exo)

        # Task-specific gating: maps flattened exo encoding → task context
        self.task_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_exo * d_exo, d_exo * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_exo * 2, d_exo),
            )
            for _ in range(n_tasks)
        ])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_lag_bank(self, exo_hist: torch.Tensor) -> torch.Tensor:
        """
        For each lag offset k, extract exo_hist[:, -(k+1), :].

        Returns:
            lag_bank: (B, n_exo, n_lags)
        """
        B, L, C = exo_hist.shape
        lags = []
        for k in self.lag_hours:
            # index of the position k steps before the last time step
            idx = max(0, L - 1 - k)
            lags.append(exo_hist[:, idx, :])            # (B, C)
        # Stack along last dim → (B, C, n_lags)
        return torch.stack(lags, dim=-1)                 # (B, n_exo, n_lags)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, exo_hist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            exo_hist: (B, L, n_exo)
        Returns:
            task_contexts: (B, n_tasks, d_exo)
                Index 0 → electricity context
                Index 1 → cooling context
                Index 2 → heating context
        """
        # Build lag bank: (B, n_exo, n_lags)
        lag_bank = self._build_lag_bank(exo_hist)

        # Per-variable encoding: (B, n_exo, d_exo)
        e = self.var_norm(self.var_encoder(lag_bank))

        # Flatten: (B, n_exo * d_exo)
        B = e.shape[0]
        e_flat = e.reshape(B, self.n_exo * self.d_exo)

        # Task-specific contexts: list of (B, d_exo)
        contexts = [gate(e_flat) for gate in self.task_gates]

        # Stack: (B, n_tasks, d_exo)
        return torch.stack(contexts, dim=1)
