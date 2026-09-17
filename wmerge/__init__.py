"""wmerge - lightweight weight-space merging for PyTorch state dicts.

Implements two task-vector-free / task-vector-based merging schemes used in our
bilingual ASR encoder study:

* ``lerp``  - linear interpolation (weighted average) of parameters.
* ``ties``  - TRIM, ELECT-SIGN & MERGE (Yadav et al., 2023).

Both operate directly on flat PyTorch state dicts (``*.pt``) or safetensors
shards, so they can merge the encoder sub-module of an ESPnet checkpoint without
depending on any particular model class.
"""

from .methods import lerp_merge, ties_merge
from .config import MergeConfig
from .runner import run_merge

__all__ = ["lerp_merge", "ties_merge", "MergeConfig", "run_merge"]
__version__ = "0.1.0"
