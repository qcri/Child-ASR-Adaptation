"""Magnitude-based sparsification (the TRIM step of TIES).

Only the magnitude scheme is needed for ``lerp``/``ties``; other schemes
(random/DARE, magnitude-outliers) are deliberately out of scope for this
release.
"""

from __future__ import annotations

import torch


def magnitude_prune(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the ``density`` fraction of largest-magnitude entries, zero the rest.

    The number of retained entries is ``floor(density * numel)``. Ordering is by
    absolute value; the returned tensor has the same shape and dtype as the input
    with pruned positions set to zero (no rescaling of the survivors).
    """
    if density >= 1.0:
        return tensor
    if density <= 0.0:
        return torch.zeros_like(tensor)

    k = int(density * tensor.numel())
    if k <= 0:
        raise ValueError(f"density {density} prunes the entire tensor")

    flat_abs = tensor.abs().reshape(-1)
    # argsort on fp16 is unsupported on CPU; rank in fp32 which is exact here.
    if flat_abs.device.type == "cpu":
        flat_abs = flat_abs.float()

    keep_idx = torch.argsort(flat_abs, descending=True)[:k]

    mask = torch.zeros_like(tensor)
    mask.reshape(-1)[keep_idx] = 1
    return tensor * mask
