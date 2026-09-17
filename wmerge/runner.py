"""Drive a full state-dict merge from a :class:`MergeConfig`.

The runner loads every source model once, walks the shared set of parameter
names and applies the per-tensor operator selected by the recipe. Parameters are
processed on CPU in the recipe's ``dtype`` so results are deterministic and
reproducible across machines.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .config import MergeConfig
from .io_utils import load_state_dict, save_state_dict, torch_dtype
from .methods import lerp_merge, ties_merge


def _aligned_keys(state_dicts: List[Dict[str, torch.Tensor]]) -> List[str]:
    """Parameter names present in every source model, in first-model order."""
    common = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        common &= set(sd.keys())
    missing = set(state_dicts[0].keys()) - common
    if missing:
        raise RuntimeError(
            f"{len(missing)} parameter(s) are not shared by all models, "
            f"e.g. {sorted(missing)[:3]}"
        )
    return [k for k in state_dicts[0].keys() if k in common]


def run_merge(config: MergeConfig, out_path: str, verbose: bool = True) -> str:
    """Execute the merge described by ``config`` and write the result.

    Returns the path of the written ``model.safetensors``.
    """
    dtype = torch_dtype(config.dtype)
    weights = [m.weight for m in config.models]
    densities = [m.density for m in config.models]

    if verbose:
        print(f"[wmerge] method={config.method} dtype={config.dtype} "
              f"models={len(config.models)}")

    sources = [load_state_dict(m.path) for m in config.models]
    base_sd = load_state_dict(config.base_model) if config.base_model else None

    reference = [base_sd] + sources if base_sd is not None else sources
    keys = _aligned_keys(reference)

    merged: Dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [sd[key].to(dtype) for sd in sources]

        if config.method == "lerp":
            out = lerp_merge(tensors, weights, normalize=config.normalize)
        elif config.method == "ties":
            base = base_sd[key].to(dtype)
            out = ties_merge(
                base=base,
                tensors=tensors,
                weights=weights,
                densities=densities,
                normalize=config.normalize,
                int8_mask=config.int8_mask,
                lambda_=config.lambda_,
            )
        else:  # pragma: no cover - guarded by config parsing
            raise ValueError(config.method)

        merged[key] = out.to(dtype)

    target = save_state_dict(merged, out_path)
    if verbose:
        print(f"[wmerge] wrote {len(merged)} tensors -> {target}")
    return target
