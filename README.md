# wmerge — weight-space merging for ASR models

A small, self-contained implementation of the two model-merging schemes used in
our bilingual ASR study:

- **LERP** (linear interpolation) — a normalized weighted average of parameters.
- **TIES** (Yadav et al., 2023) — *TRIM → ELECT SIGN → DISJOINT MERGE* on task
  vectors relative to a base model.

It operates directly on flat PyTorch state dicts, so it can merge individual
sub-modules of an ESPnet checkpoint — the **encoder** and/or the **decoder** —
while keeping the CTC head, frontend and normalisation statistics from a
reference model. No training framework is required: just `torch` and
`safetensors`. It supports both model families in our study:

- **CTC** models (encoder + CTC head, e.g. OWSM-CTC) — merge the encoder.
- **Encoder-decoder** models — merge the encoder and/or the decoder.

```
weight_merge/
├── merge.py             # CLI: python merge.py RECIPE.yaml OUT_DIR
├── espnet_tools.py      # export / split / rebuild an ESPnet checkpoint
├── wmerge/
│   ├── config.py        # YAML recipe -> typed config
│   ├── io_utils.py      # load/save .pt and .safetensors
│   ├── sparsify.py      # magnitude pruning (TRIM step)
│   ├── methods.py       # lerp_merge / ties_merge operators
│   └── runner.py        # walk the state dict, apply the operator
└── recipes/
    ├── ctc/             # encoder recipes for the CTC model
    └── enc_dec/         # encoder + decoder recipes for the enc-dec model
```

## Install

```bash
pip install torch safetensors pyyaml
```

## How a merge works

An ESPnet state dict is flat, with keys prefixed per sub-module
(`encoder.*`, `decoder.*`, `ctc.*`, `frontend.*`, …). We only merge the
sub-modules we want and keep the rest verbatim, so the pipeline is:

```
export  ->  pull the state dict out of a .pth checkpoint
split   ->  separate encoder / decoder / ctc / rest into their own files
merge   ->  run merge.py on the per-model encoder (and/or decoder) shards
rebuild ->  merged encoder (+ decoder) + reference ctc/rest -> full model
```

`split` also holds two groups out of the merge and restores them at rebuild:

- **small tensors** (`< --min-size` elements): magnitude pruning keeps
  `floor(density · numel)` entries, which rounds to zero for tiny tensors, so
  they are excluded and taken from the reference model.
- **BatchNorm stats** (`running_mean` / `running_var` / `num_batches_tracked`):
  dataset statistics, not learned weights — averaging them is meaningless, so
  they are kept from the reference model.

Both groups are empty for a pure CTC model, so the same tool covers both cases.

## Quick start — encoder-decoder model

```bash
# 1. split each checkpoint into encoder/decoder/ctc/rest (+ small/bn) shards
python espnet_tools.py split baseline.pth model_base/
python espnet_tools.py split ft_A.pth     model_A/
python espnet_tools.py split ft_B.pth     model_B/

# 2. merge the encoder, and (optionally) the decoder
python merge.py recipes/enc_dec/encoder_ties.yaml out/enc      # -> out/enc/model.safetensors
python merge.py recipes/enc_dec/decoder_ties.yaml out/dec      # -> out/dec/model.safetensors

# 3a. encoder-only merge: keep decoder/ctc/rest from the baseline
python espnet_tools.py rebuild --encoder out/enc --source model_base/ --out merged.pth

# 3b. merge both encoder and decoder
python espnet_tools.py rebuild --encoder out/enc --source model_base/ \
      --decoder out/dec --out merged.pth
```

`rebuild` flags: `--decoder DIR` uses a *merged* decoder; `--decoder-from DIR`
takes a decoder verbatim from one model's split; omit both to keep the
baseline's decoder.

## Quick start — CTC model

```bash
python espnet_tools.py split baseline.pth M0_model/
python merge.py recipes/ctc/ties.yaml out/ties         # -> out/ties/model.safetensors
python espnet_tools.py rebuild --encoder out/ties --source M0_model/ --out merged.pth
```

## Recipe format

```yaml
merge_method: ties                       # "lerp" (alias "linear") or "ties"
base_model: model_base/encoder.safetensors   # required by TIES, ignored by LERP
dtype: float32                           # compute + storage dtype
parameters:
  normalize: true                        # divide by the (masked) weight sum
  int8_mask: true                        # store the sign-consensus mask as int8
  lambda: 1.0                            # global scale on the merged task vector
models:
  - model: model_A/encoder.safetensors
    parameters: {weight: 0.15, density: 0.2}
  - model: model_B/encoder.safetensors
    parameters: {weight: 0.4,  density: 0.5}
  - model: model_C/encoder.safetensors
    parameters: {weight: 0.5,  density: 0.7}
```

A decoder recipe is identical but points at `decoder.safetensors`. For LERP,
drop `base_model`/`density` and give each model a `weight`; the result is
`Σ_i w_i·θ_i / Σ_i w_i`. Inputs may be `.safetensors` or `.pt`.

## What the algorithms compute

**LERP** — per parameter θ:

```
merged = Σ_i w_i · θ_i           (normalize: divide by Σ_i w_i)
```

**TIES** — per parameter, with task vectors `τ_i = θ_i − θ_base`:

1. **TRIM** — keep the top-`density_i` magnitude entries of each `τ_i`, zero the rest.
2. **WEIGHT** — scale each trimmed vector by `w_i`.
3. **ELECT SIGN** — per entry, elect the sign backed by the larger total
   (signed) magnitude across models.
4. **DISJOINT MERGE** — average only the entries whose sign matches the elected
   sign (`normalize` divides by the summed weight of those entries), scale by
   `lambda`, and add back onto `θ_base`.

## Reproducibility / correctness

`wmerge` is fully deterministic: running the same recipe twice yields
bit-identical outputs. Each operator was validated for numerical correctness
against a standard reference implementation of LERP and TIES on our ASR
checkpoints (CTC encoder = 508 tensors; enc-dec encoder = 452, decoder = 161):

| model        | sub-module | method | dtype   | agreement            |
|--------------|------------|--------|---------|----------------------|
| CTC          | encoder    | TIES   | float16 | bit-identical        |
| CTC          | encoder    | LERP   | float32 | rel ≤ 1.2e-7         |
| encoder-dec  | encoder    | TIES   | float32 | rel ≤ 2.6e-7         |
| encoder-dec  | encoder    | LERP   | float32 | rel ≤ 1e-7           |
| encoder-dec  | decoder    | TIES   | float32 | rel ≤ 1.6e-7         |
| encoder-dec  | decoder    | LERP   | float32 | rel ≤ 1.2e-7         |

In float16 the outputs are bit-identical; in float32 they agree to machine
precision — the worst per-tensor normalized error `‖Δ‖∞ / ‖ref‖∞` is ~1e-7
(float32 epsilon), i.e. within reduction-order rounding.
