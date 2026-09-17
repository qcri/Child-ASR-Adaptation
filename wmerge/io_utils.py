"""Loading and saving of flat tensor collections.

Source models may be either PyTorch pickles (``*.pt``) or safetensors files;
the merged output is always written as a single ``model.safetensors`` shard so
it can be dropped straight back into an ESPnet state dict.
"""

from __future__ import annotations

import os
from typing import Dict

import torch

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def torch_dtype(name: str) -> torch.dtype:
    return _DTYPE_MAP[name]


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load a flat ``{name: tensor}`` mapping from ``.pt`` or ``.safetensors``."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    obj = torch.load(path, map_location="cpu")
    # A checkpoint may wrap the parameters under a "model" key.
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    return obj


def save_state_dict(state: Dict[str, torch.Tensor], out_path: str) -> str:
    """Write ``state`` as ``out_path/model.safetensors`` and return the file path."""
    from safetensors.torch import save_file

    os.makedirs(out_path, exist_ok=True)
    target = os.path.join(out_path, "model.safetensors")
    # safetensors refuses non-contiguous tensors; make everything contiguous.
    state = {k: v.contiguous() for k, v in state.items()}
    save_file(state, target)
    return target
