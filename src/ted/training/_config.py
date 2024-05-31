# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """The training configuration for TED."""

    genome_size: int
    batch_size: int
    buffer_size: int
    sampling_old_ratio: float
    sampling_steps: int
    sampling_noise_std: float
    sampling_step_size: float
    gradient_clip_norm: float | None
    max_evaluations: int
    device: str

    def __post_init__(self) -> None:
        if not 0 <= self.sampling_old_ratio <= 1:
            msg = "The sampling ratio must be in the range [0, 1]."
            raise ValueError(msg)

        if self.max_evaluations % self.batch_size != 0:
            msg = "The number of evaluations must be divisible by the batch size."
            raise ValueError(msg)
