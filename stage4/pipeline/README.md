# Integrated Stage 4 pipeline: AdaLN sidecar + Kitchen INT8 (+ optional Sage/Sol)

This directory is the complete integrated-pipeline change set for serving MiniMax-H3 FL2VA
with the LightX2V4 4-step adapter on SGLang 0.5.19.

## What it is
- `patches/` — three bounded runtime patches (apply to the SGLang `site-packages` tree):
  1. `0001-adaln-kitchen-guard-relaxation.patch` — allows the AdaLN modulation sidecar with
     quantized weights, only when `MINIACC_STAGE4_ADALN_KITCHEN=1` (default raise preserved).
  2. `0002-stacking-relaxation-sage.patch` — lets the follow-up stacking guard admit exactly
     {sage, quantization, adaln} under the same flag; everything else stays rejected.
  3. `0003-sol-stacking-relaxation.patch` — same for Sol's isolation guard
     ({quantization, AdaLN cache} only).
- `compose_stage4.py` — builds the composed launch config: `--quantization kitchen_int8`,
  `--minimax-h3-adaln-cache-path stage3/adaln-sidecar-light4-20260915/cache.safetensors`,
  kitchen row/split limits, optional `MINIACC_STAGE3_SAGE=1` / `MINIACC_STAGE3_SOL=1`.
- `run_stage4.py` — campaign driver (speed/quality) using the retained runner in `scripts/`.
- `run_followup.py` — the retained follow-up adapter (feature flags, warmup/dispatch hooks,
  request validation) the driver composes with; pinned copy so the pipeline is self-contained.
- `deployment-stage4.json` — hash-pinned source manifest of every runtime file the pipeline
  touches (patched hashes included).
- `receipts/` — re-bound source gates for the composition-v2 runtime
  (`adaln-current-source.json`, `kitchen-current-sources/`), each with line-level
  verification notes. The AdaLN sidecar artifact itself is data, not code:
  `stage3/adaln-sidecar-light4-20260915/cache.safetensors`
  SHA-256 `8f794c6049b8fdfbb85793be29fe79e15ce02a3d00bbd729c2137bdadcdc34d0`
  (204 modulation keys, bitwise-verified against the BF16 merged weights).

## Runtime path contract
`run_followup.py` pins its admission binding to the frozen composition-v2 manifest at
`stage3/followup-20260916/deployment-v2.json` (kept local, git-ignored; a reference copy with
the same bytes is `receipts/deployment-v2.reference.json`). Do not edit that file: measured
evidence binds its hash.

## Recommendation
Run **AdaLN + Kitchen INT8 without Sage** as the default pipeline. Sage and Sol composition is
wired for experimentation, but paired quality evaluation showed Sage degrades semantic/image
quality on this pipeline and Sol is slower than the dense baseline; both remain off by default.

## Evidence
Measured evidence is kept local (git-ignored): `stage4/results-stage4.json`,
`stage4/stage4-acceptance.json`, `report/stage4_report_20260917.pdf`. No result data is tracked
in git.
