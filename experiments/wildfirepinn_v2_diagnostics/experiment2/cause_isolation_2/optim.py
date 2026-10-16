"""Curriculum, regional losses, and PCGrad for Experiment 2A.2.

The equal-weight regional loss is the uniform mean when the two regions
contain the same number of points. PCGrad is a different update: conflicting
gradients are projected apart before they are added.
"""

from __future__ import annotations

import torch

from experiments.wildfirepinn_v2_diagnostics.experiment2.fields import R0_M_S, spatial_rate_torch


def lambda_at(epoch: int, epochs: int) -> float:
    """0 for the first quarter, linear to 1 over the next half, then 1."""
    if epochs < 4:
        raise ValueError("schedule needs at least 4 epochs")
    warm = int(round(0.25 * epochs))
    ramp = int(round(0.50 * epochs))
    if epoch <= warm:
        return 0.0
    if epoch > warm + ramp:
        return 1.0
    return (epoch - warm) / float(ramp)


def rate_lambda(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """R_lambda = R0 * (1 + lambda * a * sin(kx) cos(ky)). lambda = 1 is the 2A field."""
    full = spatial_rate_torch(x, y)
    if lam >= 1.0:
        return full
    if lam <= 0.0:
        return torch.full_like(full, R0_M_S)
    return (1.0 - float(lam)) * R0_M_S + float(lam) * full


def region_mask(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fixed partition from the full R field. True where R >= R0."""
    return spatial_rate_torch(x, y) >= R0_M_S


def uniform_and_balanced(squared: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean over the batch, and the average of the two regional means."""
    if int(high.sum()) == 0 or int((~high).sum()) == 0:
        raise ValueError("both R regions need at least one point")
    uniform = squared.mean()
    balanced = 0.5 * squared[high].mean() + 0.5 * squared[~high].mean()
    return uniform, balanced


def pcgrad(gradients: list[torch.Tensor]) -> torch.Tensor:
    """Project each task gradient off the others when their dot product is negative, then sum."""
    if len(gradients) < 2:
        raise ValueError("PCGrad needs at least two gradients")
    originals = [grad.detach().reshape(-1).float() for grad in gradients]
    projected = []
    for index, current in enumerate(originals):
        updated = current.clone()
        for other_index, other in enumerate(originals):
            if index == other_index:
                continue
            dot = torch.dot(updated, other)
            if float(dot) < 0.0:
                denom = torch.dot(other, other).clamp(min=1e-12)
                updated = updated - (dot / denom) * other
        projected.append(updated)
    return torch.stack(projected, dim=0).sum(dim=0)
