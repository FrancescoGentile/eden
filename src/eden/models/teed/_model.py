# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torch import Tensor, nn
from typing_extensions import override

from eden.models import Model as BaseModel

from ._config import Config
from ._hgnn import HGNN
from ._meanshift import MeanShift


class Model(BaseModel):
    """The TEED model."""

    def __init__(self, config: Config | None = None, **kwargs: Any) -> None:
        """Initializes the TEED model.

        Args:
            config: The configuration of the model. If not provided, the configuration
                will be created from the provided keyword arguments.
            **kwargs: Additional arguments to pass to the model. These will be used to
                create the configuration if one is not provided. This is useful when
                instantiating the model from a configuration file.
        """
        super().__init__()

        config = config or Config.from_dict(kwargs)

        self.embeds = nn.Embedding(config.num_variables, config.embed_dim)
        self.ms = MeanShift(config.ms)

        self.gene_proj = nn.Linear(1, config.embed_dim)
        self.gene_embeds_proj = nn.Linear(config.embed_dim * 2, config.embed_dim)

        self.hgnn = HGNN()

        self.out_proj = nn.Linear(config.embed_dim, 1)

        self.register_buffer("cached_boundary", None, persistent=False)
        self.cached_boundary: Tensor | None = None

    @override
    def get_range(self) -> tuple[float, float]:
        return -1.0, 1.0

    @override
    def freeze(self) -> None:
        super().freeze()
        _, boundary = self.ms(self.embeds.weight)
        self.cached_boundary = boundary

    @override
    def unfreeze(self) -> None:
        super().unfreeze()
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
