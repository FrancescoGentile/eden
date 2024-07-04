# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from torch import nn


class Module(nn.Module):
    """A type-friendly version of `torch.nn.Module`.

    In `torch.nn.Module`, the forward pass is defined in the `forward` method, but to
    call the module, we use the `__call__` method that dispatches to the defined
    `forward` method. However, this is not type-friendly, as the `__call__` method does
    not have the same signature as the `forward` method.

    This class fixes this by allowing its subclasses to define the forward pass directly
    in the `__call__` method. Note that all capabilities of the original
    `torch.nn.Module.__call__` method (like calling the hook functions) are preserved.
    """

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)

        if cls.__call__ != super().__call__:
            cls.forward = cls.__call__
            cls.__call__ = super().__call__
