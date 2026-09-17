#!/usr/bin/env python
"""Command-line entry point: merge model weights from a YAML recipe.

Usage
-----
    python merge.py RECIPE.yaml OUT_DIR

The merged parameters are written to ``OUT_DIR/model.safetensors``.

Examples
--------
    python merge.py recipes/lerp.yaml  out/lerp
    python merge.py recipes/ties.yaml  out/ties
"""

from __future__ import annotations

import argparse

from wmerge import MergeConfig, run_merge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("recipe", help="path to the YAML merge recipe")
    parser.add_argument("out_dir", help="output directory for model.safetensors")
    parser.add_argument("-q", "--quiet", action="store_true", help="silence progress")
    args = parser.parse_args()

    config = MergeConfig.load(args.recipe)
    run_merge(config, args.out_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
