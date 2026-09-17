"""Per-tensor merge operators for LERP and TIES.

Each operator takes the aligned stack of source tensors for one parameter and
returns the merged tensor. The higher-level runner (:mod:`wmerge.runner`) walks
the state dict, feeds each parameter through the selected operator and collects
the result.

Notation for a single parameter:
    * ``theta_i`` - the i-th model's tensor for this parameter.
    * ``theta_0`` - the base model's tensor (TIES only).
    * ``tau_i = theta_i - theta_0`` - the i-th task vector (TIES only).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .sparsify import magnitude_prune


def lerp_merge(
    tensors: Sequence[torch.Tensor],
    weights: Sequence[float],
    normalize: bool = True,
) -> torch.Tensor:
    """Weighted linear interpolation of parameters.

        merged = sum_i w_i * theta_i           (normalize = False)
        merged = sum_i w_i * theta_i / sum_i w_i   (normalize = True)

    With two models and ``w = [1 - a, a]`` this is the familiar
    ``(1 - a) * theta_0 + a * theta_1`` interpolation.
    """
    stacked = torch.stack(list(tensors), dim=0)
    w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
    while w.dim() < stacked.dim():
        w = w.unsqueeze(-1)

    merged = (w * stacked).sum(dim=0)
    if normalize:
        merged = merged / w.sum(dim=0)
    return merged


def _elect_sign(weighted_deltas: torch.Tensor, int8_mask: bool) -> torch.Tensor:
    """ELECT step: boolean mask of entries agreeing with the majority sign.

    The majority sign per entry is decided by the sign of the *sum* of the
    weighted task vectors across models (the total-magnitude vote from the TIES
    paper, ties at exactly zero counting as positive).
    """
    mask_dtype = torch.int8 if int8_mask else weighted_deltas.dtype

    signs = weighted_deltas.sign().to(mask_dtype)
    total = weighted_deltas.sum(dim=0)
    majority = (total >= 0).to(mask_dtype) * 2 - 1
    return signs == majority


def ties_merge(
    base: torch.Tensor,
    tensors: Sequence[torch.Tensor],
    weights: Sequence[float],
    densities: Sequence[float],
    normalize: bool = True,
    int8_mask: bool = False,
    lambda_: float = 1.0,
) -> torch.Tensor:
    """TIES merge for a single parameter (TRIM -> ELECT SIGN -> DISJOINT MERGE).

    1. TRIM   - sparsify each task vector ``tau_i`` to its top-``density_i``
                magnitude entries.
    2. WEIGHT - scale each trimmed task vector by ``w_i``.
    3. ELECT  - per entry, pick the sign backed by the greater total magnitude.
    4. MERGE  - average only the entries agreeing with the elected sign, then
                add the result back onto the base parameter.
    """
    deltas: List[torch.Tensor] = []
    for theta, density in zip(tensors, densities):
        tau = theta.to(base.dtype) - base
        deltas.append(magnitude_prune(tau, density))

    if not deltas:
        return base

    stacked = torch.stack(deltas, dim=0)
    w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
    while stacked.dim() > w.dim():
        w = w.unsqueeze(-1)

    weighted = stacked * w
    mask = _elect_sign(weighted, int8_mask)

    merged_delta = (weighted * mask).sum(dim=0)
    if normalize:
        divisor = (w * mask).sum(dim=0)
        divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)
        merged_delta = merged_delta / divisor

    if lambda_ != 1.0:
        merged_delta = merged_delta * lambda_

    return (base + merged_delta).to(base.dtype)


# Convenience export for the pruning step, so downstream code can reuse it.
__all__ = ["lerp_merge", "ties_merge", "magnitude_prune"]
