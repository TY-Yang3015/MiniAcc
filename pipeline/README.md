# Integrated serving pipeline

Everything needed to run MiniMax-H3 FL2VA (LightX2V4 4-step adapter) with the integrated
AdaLN + Kitchen INT8 + SageAttention pipeline on SGLang 0.5.19.

## Contents

## Quickstart

```bash
for p in patches/*.patch; do patch -p1 < "$p"; done   # apply the runtime patches
python pipeline/setup.py                              # verify prerequisites
python pipeline/serve.py --output outputs/demo        # one timed request
```

## Contents

- `patches/` — three small runtime patches (apply to the SGLang `site-packages` tree).
  Each only activates when `MINIACC_STAGE4_ADALN_KITCHEN=1` is set; default behavior is
  unchanged otherwise:
  1. allow the AdaLN sidecar with 8-bit-quantized weights;
  2. allow SageAttention to run on the quantized pipeline;
  3. allow Sol (sparse attention) to run on the quantized pipeline.
- `serve.py` — the friendly entry point (timed run or quality cohort, Sage on by default,
  `--no-sage` for the fidelity-first configuration).
- `setup.py` — prerequisite checker: runtime, model snapshot, adapter and sidecar presence
  and hashes; copies the frozen deployment manifest into place if missing.
- `compose.py` — builds the launch configuration (quantization, sidecar path, optional
  attention variants) from the pinned receipts in `receipts/`.
- `run_stage4.py` + `run_followup.py` — the campaign driver and adapter used by `serve.py`.
- `deployment.json` — hash-pinned manifest of every runtime file the pipeline touches.
- `eval.yaml` — the frozen 16-prompt evaluation set used by quality runs.
- `receipts/` — source-verification records for the pinned runtime (what changed, and the
  line-level checks showing behavior-critical code is intact), plus small pinned data receipts.
- `demo/` — demonstration clips and montages (1344×768, 124 frames, stereo audio).

## Data dependency (not in git)

The AdaLN sidecar is a data artifact: `stage3/adaln-sidecar-light4-20260915/cache.safetensors`
SHA-256 `8f794c6049b8fdfbb85793be29fe79e15ce02a3d00bbd729c2137bdadcdc34d0`
(204 precomputed conditioning keys, verified bit-for-bit against the BF16 merged weights).
The driver's request-admission binding also expects the frozen manifest at
`stage3/followup-20260916/deployment-v2.json` (a byte-identical reference copy is
`receipts/deployment-v2.reference.json`); do not edit it.

## Defaults and alternatives

- Default: AdaLN + Kitchen INT8 + SageAttention (fastest; measurable quality cost in
  overall consistency and imaging — see the main README tables).
- Fidelity-first: AdaLN + Kitchen INT8 (quality unchanged from baseline within noise).
- Sol is available for research but is slower than the dense baseline.
