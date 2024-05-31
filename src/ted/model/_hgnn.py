# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor

from ted.nn import Module


class HGNN(Module):
    """Hypergraph Neural Network."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor, boundary: Tensor) -> Tensor:  # noqa: ARG002
        """Updates the node embeddings using the hypergraph neural network.

        Args:
            x: The input tensor representing the node embeddings. This is expected to
                be a 3D tensor of shape (B, N, D), where B is the batch size, N is the
                number of nodes, and D is the embedding dimension.
            boundary: The input tensor representing the boundary nodes. This is expected
                to be a 2D tensor of shape (B, N), where B is the batch size and N is
                the number of nodes.

        Returns:
            The updated node embeddings.
        """
        return x
