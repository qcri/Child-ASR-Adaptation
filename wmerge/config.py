"""Parsing of the merge recipe (YAML) into typed dataclasses.

The recipe format is intentionally minimal::

    merge_method: ties            # "lerp" (aka "linear") or "ties"
    base_model: M0/encoder.pt     # required for ties, ignored for lerp
    dtype: float16                # compute + storage dtype
    parameters:                   # method-level knobs (optional)
      normalize: true
      int8_mask: true
      lambda: 1.0
    models:
      - model: B/encoder.pt
        parameters: {weight: 0.45, density: 0.6}
      - model: A/encoder.pt
        parameters: {weight: 0.35, density: 0.3}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

_DTYPES = {"float16", "bfloat16", "float32", "float64"}

# Aliases so a recipe written for the reference implementation still parses.
_METHOD_ALIASES = {"linear": "lerp", "lerp": "lerp", "ties": "ties"}


@dataclass
class ModelEntry:
    """A single source model and its per-tensor merge parameters."""

    path: str
    weight: float = 1.0
    density: float = 1.0

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ModelEntry":
        params = raw.get("parameters") or {}
        weight = params.get("weight")
        if weight is None:
            raise ValueError(f"model '{raw.get('model')}' is missing a 'weight'")
        return cls(
            path=raw["model"],
            weight=float(weight),
            density=float(params.get("density", 1.0)),
        )


@dataclass
class MergeConfig:
    method: str
    models: List[ModelEntry]
    base_model: Optional[str] = None
    dtype: str = "float32"
    normalize: bool = True
    int8_mask: bool = False
    lambda_: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "MergeConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MergeConfig":
        method_raw = str(raw["merge_method"]).lower()
        if method_raw not in _METHOD_ALIASES:
            raise ValueError(
                f"unsupported merge_method '{method_raw}'; "
                f"this package only ships 'lerp' and 'ties'"
            )
        method = _METHOD_ALIASES[method_raw]

        dtype = str(raw.get("dtype", "float32"))
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported dtype '{dtype}'")

        models = [ModelEntry.from_dict(m) for m in raw["models"]]
        if not models:
            raise ValueError("recipe lists no models")

        params = raw.get("parameters") or {}
        cfg = cls(
            method=method,
            models=models,
            base_model=raw.get("base_model"),
            dtype=dtype,
            normalize=bool(params.get("normalize", True)),
            int8_mask=bool(params.get("int8_mask", False)),
            lambda_=float(params.get("lambda", 1.0)),
            params=params,
        )

        if cfg.method == "ties" and cfg.base_model is None:
            raise ValueError("ties requires a 'base_model'")
        return cfg
