# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

import math

import entmax
import torch
from torch import Tensor, nn

from ted.nn import Module


class MeanShift(Module):
    """Gaussian Mean-Shift clustering."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        sigma: float = 1.0,
        damping: float = 0.5,
        max_iter: int = 100,
        shift_tol: float = 1e-4,
        alpha: float = 1.5,
    ) -> None:
        """Initializes the clustering algorithm.

        Args:
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
        super().__init__()

        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.damping = damping
        self.max_iter = max_iter
        self.shift_tol = shift_tol
        self.alpha = alpha

    def find_centroids(self, x: Tensor) -> Tensor:
        """Finds the centroids of the clusters using the mean-shift algorithm.

        Args:
            x: The input data tensor. This is expected to be a 2D tensor of shape
                (N, D), where N is the number of data points and D is the dimensionality
                of the data.

        Returns:
            The centroids of the clusters. This is a 2D tensor of shape (C, D), where C
            is the number of clusters.
        """
        centroids = x.clone()
        gamma = 1 / (2 * self.sigma**2)
        for _ in range(self.max_iter):
            dist = torch.cdist(x, centroids, p=2)  # (N, N)
            weight = torch.exp(-(dist**2) * gamma)  # (N, N)

            nominator = weight @ x  # (N, D)
            denominator = weight.sum(dim=1, keepdim=True)  # (N, 1)
            new_centroids = nominator / denominator

            shift = (new_centroids - centroids).norm(dim=1, p=2)  # (N,)
            if shift.max() < self.shift_tol:
                break

            centroids = self.damping * centroids + (1 - self.damping) * new_centroids

        centroids = _nms(x, centroids, threshold=self.sigma.item())

        return centroids

    def assign(self, x: Tensor, centroids: Tensor) -> Tensor:
        """Assigns the data points to the clusters.

        Args:
            x: The input data tensor. This is expected to be a 2D tensor of shape
                (N, D), where N is the number of data points and D is the dimensionality
                of the data.
            centroids: The centroids of the clusters. This is a 2D tensor of shape
                (C, D), where C is the number of clusters.

        Returns:
            A tensor of shape (N, C), where each row is a probability distribution over
            the clusters.
        """
        cost = -torch.cdist(x, centroids, p=2)  # (N, C)

        match self.alpha:
            case 1:
                assignment = torch.softmax(cost, dim=1)
            case 1.5:
                assignment = entmax.entmax15(cost, dim=1)
            case 2:
                assignment = entmax.sparsemax(cost, dim=1)
            case math.inf:
                assigned_to = cost.argmax(dim=1)  # (N,)
                assignment = torch.zeros_like(cost)
                assignment[torch.arange(x.size(0), device=x.device), assigned_to] = 1
                # use straight-through estimator
                assignment = assignment - cost.detach() + cost
            case _:
                msg = f"Invalid alpha value: {self.alpha}."
                raise RuntimeError(msg)

        return assignment  # type: ignore

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Performs mean-shift clustering on the input data.

        Args:
            x: The input data tensor. This is expected to be a 2D tensor of shape
                (N, D), where N is the number of data points and D is the dimensionality
                of the data.

        Returns:
            A tuple containing the centroids and the assignment of the data points to
            the clusters. The centroids tensor has shape (C, D), where C is the number
            of clusters. The assignment tensor has shape (N, C), where each row is a
            probability distribution over the clusters.
        """
        centroids = self.find_centroids(x)
        assignment = self.assign(x, centroids)

        return centroids, assignment


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #


def _nms(x: Tensor, centroids: Tensor, threshold: float) -> Tensor:
    pc_dist = torch.cdist(x, centroids, p=2)  # (N, N)
    _, clostest_centroid = pc_dist.min(dim=1)  # (N,)
    uniques, counts = clostest_centroid.unique(return_counts=True)
    scores = torch.zeros_like(clostest_centroid)
    scores[uniques] = counts

    cc_dist = torch.cdist(centroids, centroids, p=2)
    overlap = cc_dist < threshold
    overlap.fill_diagonal_(fill_value=False)

    order = scores.argsort(descending=True)
    keep = torch.ones_like(scores, dtype=torch.bool)
    for idx in order:
        if keep[idx]:
            keep &= ~overlap[idx]
            keep[idx] = scores[idx] > 0

    return centroids[keep]  # (C, D)
