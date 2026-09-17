#!/usr/bin/env python
"""Split / rebuild helpers for slotting merged sub-modules into ESPnet models.

An ESPnet ASR checkpoint is a flat state dict whose keys are prefixed by
sub-module. We merge the encoder and (optionally) the decoder, and keep the rest
verbatim from a reference model. The workflow is:

    export  -> pull the state dict out of a *.pth checkpoint
    split   -> separate encoder / decoder / ctc / rest into their own files
    merge   -> run merge.py on the per-model encoder/decoder shards
    rebuild -> merged encoder (+ decoder) + reference ctc/rest -> full model

Two groups of parameters are held out of the merge and restored at rebuild time:

* ``small_tensors``  - tensors with < ``--min-size`` elements. Magnitude pruning
  (the TIES TRIM step) keeps ``floor(density * numel)`` entries, which rounds to
  zero for tiny tensors and is undefined, so they are excluded and taken from the
  reference model instead.
* ``bn_stats``       - BatchNorm running statistics (``running_mean`` /
  ``running_var`` / ``num_batches_tracked``). These are dataset statistics, not
  learned weights, so averaging them is meaningless; they are kept from the
  reference model.

Both groups are empty for a pure encoder+CTC model (e.g. OWSM-CTC), so the same
tool covers the CTC-only and the encoder-decoder cases.

Usage
-----
    python espnet_tools.py export  ckpt.pth              full.pt
    python espnet_tools.py split   ckpt.pth              model_A/      [--min-size 10]
    python espnet_tools.py rebuild --encoder merged_enc/ --source model_base/ --out out.pth
                                   [--decoder merged_dec/ | --decoder-from model_A/]
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from typing import Dict

import torch
from safetensors.torch import load_file, save_file

# BatchNorm statistics are dataset-dependent and must not be averaged.
BN_STAT_KEYS = ("running_mean", "running_var", "num_batches_tracked")
# Buckets written by ``split`` that a rebuild pulls from the reference model.
KEEP_BUCKETS = ("ctc", "rest", "small_tensors", "bn_stats")


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export(pth_path: str, out_path: str) -> None:
    """Extract the plain parameter state dict from an ESPnet ``.pth``."""
    ckpt = torch.load(pth_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    torch.save(state, out_path)
    print(f"exported {len(state)} tensors -> {out_path}")


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def split(checkpoint_path: str, out_dir: str, min_size: int = 10) -> None:
    """Split an ESPnet checkpoint into mergeable shards + held-out shards."""
    obj = torch.load(checkpoint_path, map_location="cpu")
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    buckets: Dict[str, "OrderedDict[str, torch.Tensor]"] = {
        name: OrderedDict()
        for name in ("encoder", "decoder", "ctc", "rest", "small_tensors", "bn_stats")
    }

    for key, value in state.items():
        if any(k in key for k in BN_STAT_KEYS):
            buckets["bn_stats"][key] = value
        elif value.numel() < min_size:
            buckets["small_tensors"][key] = value
        elif key.startswith("encoder"):
            buckets["encoder"][key] = value
        elif key.startswith("decoder"):
            buckets["decoder"][key] = value
        elif key.startswith("ctc"):
            buckets["ctc"][key] = value
        else:  # frontend, normalize, embed, ...
            buckets["rest"][key] = value

    os.makedirs(out_dir, exist_ok=True)
    for name, sd in buckets.items():
        if not sd:
            continue  # e.g. no decoder for a CTC-only model
        save_file(dict(sd), os.path.join(out_dir, f"{name}.safetensors"))

    summary = "  ".join(f"{n}={len(sd)}" for n, sd in buckets.items() if sd)
    print(f"[split] {summary}  ->  {out_dir}")


# --------------------------------------------------------------------------- #
# rebuild
# --------------------------------------------------------------------------- #
def _load(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        return dict(load_file(path))
    return torch.load(path, map_location="cpu")


def _merged_or_shard(merged_dir: str | None, shard_dir: str, bucket: str) -> Dict[str, torch.Tensor]:
    """Load a merged ``model.safetensors`` if given, else the plain shard."""
    if merged_dir:
        return _load(os.path.join(merged_dir, "model.safetensors"))
    path = os.path.join(shard_dir, f"{bucket}.safetensors")
    return _load(path) if os.path.exists(path) else {}


def rebuild(
    encoder_dir: str,
    source_dir: str,
    out_path: str,
    decoder_merged_dir: str | None = None,
    decoder_from: str | None = None,
) -> None:
    """Recombine a merged encoder (+ optional merged/alternate decoder) with the
    CTC head, frontend and held-out tensors from ``source_dir``.

    * ``encoder_dir``        - dir holding the merged encoder ``model.safetensors``.
    * ``source_dir``         - split dir supplying ctc / rest / small_tensors /
                               bn_stats (and the decoder, unless overridden).
    * ``decoder_merged_dir`` - dir holding a merged decoder ``model.safetensors``.
    * ``decoder_from``       - split dir whose ``decoder.safetensors`` to use verbatim.
    """
    encoder = _load(os.path.join(encoder_dir, "model.safetensors"))

    if decoder_merged_dir:
        decoder = _load(os.path.join(decoder_merged_dir, "model.safetensors"))
    else:
        decoder = _merged_or_shard(None, decoder_from or source_dir, "decoder")

    full: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for bucket in ("rest", "small_tensors", "bn_stats", "ctc"):
        full.update(_merged_or_shard(None, source_dir, bucket))
    full.update(encoder)
    full.update(decoder)

    out_parent = os.path.dirname(out_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    torch.save(dict(full), out_path)

    n_params = sum(v.numel() for v in full.values())
    print(f"[rebuild] {len(full)} tensors ({n_params:,} params) -> {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export", help="pth -> state dict")
    p.add_argument("pth")
    p.add_argument("out")

    p = sub.add_parser("split", help="checkpoint -> encoder/decoder/ctc/rest shards")
    p.add_argument("checkpoint")
    p.add_argument("out_dir")
    p.add_argument("--min-size", type=int, default=10,
                   help="tensors smaller than this are held out of the merge")

    p = sub.add_parser("rebuild", help="merged shards + reference -> full model")
    p.add_argument("--encoder", required=True, help="merged encoder dir (model.safetensors)")
    p.add_argument("--source", required=True, help="split dir for ctc/rest/small/bn/decoder")
    p.add_argument("--out", required=True, help="output .pth path")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--decoder", dest="decoder_merged", help="merged decoder dir (model.safetensors)")
    g.add_argument("--decoder-from", dest="decoder_from", help="split dir to take decoder.safetensors from")

    args = parser.parse_args()
    if args.cmd == "export":
        export(args.pth, args.out)
    elif args.cmd == "split":
        split(args.checkpoint, args.out_dir, args.min_size)
    elif args.cmd == "rebuild":
        rebuild(args.encoder, args.source, args.out,
                decoder_merged_dir=args.decoder_merged, decoder_from=args.decoder_from)


if __name__ == "__main__":
    main()
