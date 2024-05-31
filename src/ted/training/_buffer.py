# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

from collections import deque

from torch import Tensor

from ._config import Config


class Buffer:
    """A buffer of solutions for training TED."""

    def __init__(self, config: Config) -> None:
        self._samples = deque(maxlen=config.buffer_size)
        self._targets = deque(maxlen=config.buffer_size)

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def add(self, samples: Tensor, targets: Tensor) -> None:
        """Adds samples and targets to the buffer.

        !!! note

            The current implementation overwrites the oldest elements in the buffer
            when it is full. In the future, we may implement a more sophisticated
            strategy, such as prioritized experience replay.
        """
        self._samples.extend(samples)
        self._targets.extend(targets)

    def clear(self) -> None:
        """Clears all elements from the buffer."""
        self._samples.clear()
        self._targets.clear()

    def is_empty(self) -> bool:
        """Returns whether the buffer is empty."""
        return len(self._samples) == 0

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._samples[index], self._targets[index]
