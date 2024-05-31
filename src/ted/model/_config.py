# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Configuration options for the TED model."""

    num_nodes: int
    embed_dim: int = 64
    ms_sigma: float = 1.0
    ms_damping: float = 0.5
    ms_max_iter: int = 100
    ms_shift_tol: float = 1e-4
    ms_alpha: float = 2
    num_layers: int = 2
