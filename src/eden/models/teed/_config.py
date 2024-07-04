# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any

import serde
from typing_extensions import Self


@serde.serde
@dataclasses.dataclass(frozen=True)
class MeanShiftConfig:
    """Configuration for the MeanShift algorithm.

    Attributes:
        sigma: The value used to initialize the standard deviation of the Gaussian
                kernel.
        damping: The damping factor for the update rule. To avoid numerical
            instability, the new centroids are computed as a weighted average of the
            old centroids and the latest estimate. This factor controls the weight
            of the old centroids.
        max_iter: The maximum number of iterations to run the algorithm.
        shift_tol: The tolerance for the stopping criterion. The algorithm stops
            when the maximum shift of the centroids is less than this value.
        alpha: The alpha parameter for the soft-assignment of the data points to
            the clusters.
    """

    sigma: float = 1.0
    damping: float = 0.5
    max_iter: int = 100
    shift_tol: float = 1e-4
    alpha: float = 2


@serde.serde
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for the TEED model."""

    num_variables: int
    embed_dim: int = 64
    num_layers: int = 2
    ms: MeanShiftConfig = dataclasses.field(default_factory=MeanShiftConfig)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Self:
        """Creates a configuration object from a dictionary."""
        return serde.from_dict(cls, config)
