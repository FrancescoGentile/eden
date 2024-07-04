# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any, Literal

import serde
from typing_extensions import Self


@serde.serde
@dataclasses.dataclass(frozen=True)
class SamplerConfig:
    """The configuration for the sampling process.

    Attributes:
        num_samples: The number of samples to generate in the sampling process.
        old_ratio: The ratio of old solutions to take from the buffer and to use
            to bootstrap the sampling process. The remaining solutions are initialized
            randomly.
        num_steps: The number of steps to take in the sampling process.
        noise_std: The standard deviation of the noise added to the solutions.
        step_size: The step size used in the sampling process.
        gradient_clip_norm: The norm used to clip the gradients. If None, no clipping
            is performed.
    """

    num_samples: int
    old_ratio: float
    num_steps: int
    noise_std: float
    step_size: float
    gradient_clip_norm: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.old_ratio <= 1:
            msg = "The old ratio must be in the range [0, 1]."
            raise ValueError(msg)


@serde.serde
@dataclasses.dataclass(frozen=True)
class BufferConfig:
    """The configuration for the buffer.

    Attributes:
        min_size: The minimum size of the buffer. The buffer is filled with random
            solutions until it reaches this size.
        max_size: The maximum size of the buffer.
        sampling_strategy: The strategy used to sample solutions from the buffer.
            At the moment, only `uniform` is supported.
        replacement_strategy: The strategy used to replace solutions in the buffer
            when it is full. At the moment, only `fifo` is supported.
    """

    min_size: int
    max_size: int
    sampling_strategy: Literal["uniform"] = "uniform"
    replacement_strategy: Literal["fifo"] = "fifo"

    def __post_init__(self) -> None:
        if self.min_size > self.max_size:
            msg = "The minimum size must be less than or equal to the maximum size."
            raise ValueError(msg)

        valid_strategies = {"uniform"}
        if self.sampling_strategy not in valid_strategies:
            msg = f"Invalid sampling strategy. Must be one of {valid_strategies}."
            raise ValueError(msg)

        valid_strategies = {"fifo"}
        if self.replacement_strategy not in valid_strategies:
            msg = f"Invalid replacement strategy. Must be one of {valid_strategies}."
            raise ValueError(msg)


@serde.serde
@dataclasses.dataclass(frozen=True)
class UpdateConfig:
    """The configuration for the update process.

    Attributes:
        num_steps: The number of update steps to take.
        batch_size: The batch size used for each update step.
        gradient_clip_norm: The norm used to clip the gradients. If None, no clipping
            is performed.
    """

    num_steps: int
    batch_size: int
    gradient_clip_norm: float | None = None


@serde.serde
@dataclasses.dataclass(frozen=True)
class Config:
    """The training configuration.

    Attributes:
        sampler: The configuration for the sampling process.
        buffer: The configuration for the buffer.
        update: The configuration for the update process.
        max_num_evaluations: The maximum number of evaluations to perform during
            training. This corresponds to the number of samples generated in the
            sampling process.
        device: The device to use for training.
    """

    sampler: SamplerConfig
    buffer: BufferConfig
    update: UpdateConfig
    max_num_evaluations: int
    device: str

    def __post_init__(self) -> None:
        if self.max_num_evaluations < 0:
            msg = "The maximum number of evaluations must be non-negative."
            raise ValueError(msg)

        if self.max_num_evaluations % self.sampler.num_samples != 0:
            msg = (
                "The maximum number of evaluations must be a multiple of the number "
                "of samples."
            )
            raise ValueError(msg)

        if self.max_num_evaluations < self.buffer.min_size:
            msg = (
                "The maximum number of evaluations must be greater than or equal "
                "to the minimum buffer size."
            )
            raise ValueError(msg)

        if self.update.batch_size * self.update.num_steps > self.buffer.min_size:
            msg = (
                "The product of the batch size and the number of update steps must "
                "be less than or equal to the minimum buffer size."
            )
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a Config instance from a dictionary."""
        return serde.from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """Convert the Config instance to a dictionary."""
        return serde.to_dict(self)
