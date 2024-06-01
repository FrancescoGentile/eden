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

        self.register_buffer("cached_boundary", None, persistent=False)
        self.cached_boundary: Tensor | None = None

    def freeze(self) -> None:
        """Freezes the model, preventing any parameters from being updated.

        This method is useful when the model is used for inference, as it prevents
        unnecessary gradient computations. To further optimize the model for inference,
        the boundary returned by the mean shift algorithm is cached to avoid recomputing
        it for each sample.
        """
        self.requires_grad_(requires_grad=False)
        _, boundary = self.ms(self.embeds.weight)
        self.cached_boundary = boundary

    def unfreeze(self) -> None:
        """Unfreezes the model, allowing parameters to be updated.

        This method is useful when the model is used for training, as it allows the
        model to learn from the data. The boundary is also cleared to ensure that it is
        recomputed for each sample.
        """
        self.requires_grad_(requires_grad=True)
        self.cached_boundary = None

    def __call__(self, x: Tensor) -> Tensor:
        """Computes the energy of each sample.

        Args:
            x: The input tensor representing the batched individual. This is expected to
                be a 2D tensor of shape (N, D), where B is the batch size and N is the
                number of genes representing the individual.

        Returns:
            The energy of each individual. This is a 1D tensor of shape (B,).
        """
        if self.cached_boundary is not None:
            boundary = self.cached_boundary
        else:
            _, boundary = self.ms(self.embeds.weight)

        x = self.gene_proj(x.unsqueeze(-1))  # (B, N, D)
        embeds = self.embeds.weight.expand_as(x)  # (B, N, D)
        features = torch.cat([x, embeds], dim=-1)
        features = self.gene_embeds_proj(features)

        features = self.hgnn(features, boundary)  # (B, N, D)
        pooled = features.mean(dim=1)  # (B, D)
        energy = self.out_proj(pooled).squeeze_(-1)  # (B,)

        return energy
