# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor, nn

from ted.nn import Module

from ._config import Config
from ._hgnn import HGNN
from ._meanshift import MeanShift


class TED(Module):
    """Topology-based Estimantion of Distribution model."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.embeds = nn.Embedding(config.num_nodes, config.embed_dim)
        self.ms = MeanShift(
            sigma=config.ms_sigma,
            damping=config.ms_damping,
            max_iter=config.ms_max_iter,
            shift_tol=config.ms_shift_tol,
            alpha=config.ms_alpha,
        )

        self.gene_proj = nn.Linear(1, config.embed_dim)
        self.gene_embeds_proj = nn.Linear(config.embed_dim * 2, config.embed_dim)

        self.hgnn = HGNN()

        self.out_proj = nn.Linear(config.embed_dim, 1)

    def __call__(self, x: Tensor) -> Tensor:
        """Computes the energy of each sample.

        Args:
            x: The input tensor representing the batched individual. This is expected to
                be a 2D tensor of shape (N, D), where B is the batch size and N is the
                number of genes representing the individual.

        Returns:
            The energy of each individual. This is a 1D tensor of shape (B,).
        """
        _, boundary = self.ms(self.embeds.weight)

        x = self.gene_proj(x.unsqueeze(-1))  # (B, N, D)
        embeds = self.embeds.weight.expand_as(x)  # (B, N, D)
        features = torch.cat([x, embeds], dim=-1)
        features = self.gene_embeds_proj(features)

        features = self.hgnn(features, boundary)  # (B, N, D)
        pooled = features.mean(dim=1)  # (B, D)
        energy = self.out_proj(pooled).squeeze_(-1)  # (B,)

        return energy
