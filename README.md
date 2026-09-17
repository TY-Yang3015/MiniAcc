# MiniAcc

**Local-deployable MiniMax-H3 serving, accelerated by the integrated AdaLN + Kitchen INT8 pipeline.**
SGLang 0.5.19 · LightX2V4 4-step distilled adapter · 1344×768 · 124 frames · 24 FPS · stereo 32 kHz

---

## Overview

MiniAcc serves the MiniMax-H3 audio–video diffusion model on a single workstation GPU. The
production path is the **Stage 4 integrated pipeline**: a per-layer AdaLN modulation sidecar
composed with Kitchen INT8 weight-only quantization, hard-coded onto the frozen LightX2V4
baseline. Weights stream from host memory every request, so the pipeline attacks exactly that
bottleneck — halved weight traffic plus fused modulation — instead of faster math.

Optional SageAttention and Sol composition paths are wired for experimentation behind the same
flag; the recommended default is **AdaLN + Kitchen without Sage** (paired quality evaluation
showed semantic/image degradation from Sage on this pipeline; Sol is slower than the dense
baseline it replaces).

## Demo

Same prompt, same seed, three pipelines (RTX 4090):

![Control vs pipeline vs pipeline+Sage](stage4/pipeline/demo/trio-0195-montage.jpg)

| Control (baseline) | AdaLN + Kitchen pipeline |
|---|---|
| <video src="https://github.com/TY-Yang3015/MiniAcc/raw/main/stage4/pipeline/demo/vbench-0195-control.mp4" controls width="100%"></video> | <video src="https://github.com/TY-Yang3015/MiniAcc/raw/main/stage4/pipeline/demo/vbench-0195-adaln_kitchen.mp4" controls width="100%"></video> |

Adopted-pipeline clips across four evaluation prompts:

![Pipeline demo montage](stage4/pipeline/demo/pipeline-montage.jpg)

All clips are 1344×768, 124 frames, stereo audio, generated under strict admission with
full-AV validation; hashes are kept with the local evidence.

## Performance

| Arm (RTX 4090, one 5.17 s AV request) | Saved-AV caller time | Speedup |
|---|---:|---:|
| Control (v2 baseline) | 326.5 s | 1.00× |
| **AdaLN + Kitchen pipeline** | **149.3 s** | **2.19×** |
| Pipeline + Sage (wired, not recommended) | 130.2 s | 2.51× |

Paired VBench quality (16 clips, six metrics n=16, overall n=4) evaluates every arm before
adoption; full measurements are kept local and are intentionally not tracked in git.

## Resource budgets

| Budget | Contract |
|---|---|
| Serving GPU | 1 × RTX 4090 24 GB (SM89) or 1 × A100 80 GB (SM80), one server per device |
| GPU headroom | assigned device keeps **≥ 8,192 MiB free** at all times (fail-closed) |
| Host memory | **MemAvailable ≥ 16 GiB** enforced through setup/warmup/measurement; plan ≥ 64 GB during model load (unpinned CPU-weight placement) |
| Disk | ≈ 120 GB for model snapshot + merged-adapter/diffusion caches; AdaLN sidecar 77.6 MB (hash-pinned) |
| Timing | saved-AV caller time includes submission, recurring offload, denoising, decode and file write; setup/JIT/warmup always excluded |
| Quality scoring | frozen VBench stack, six metrics + overall consistency, paired same-hardware controls only |

## Quickstart

```bash
# 1. Apply the three runtime patches to your SGLang 0.5.19 site-packages
patch -p1 < stage4/pipeline/patches/0001-adaln-kitchen-guard-relaxation.patch
patch -p1 < stage4/pipeline/patches/0002-stacking-relaxation-sage.patch
patch -p1 < stage4/pipeline/patches/0003-sol-stacking-relaxation.patch

# 2. Verify the source manifest (hashes must match your runtime)
python -c "import json;print(len(json.load(open('stage4/pipeline/deployment-stage4.json'))['installed']),'files pinned')"

# 3. Launch the pipeline (speed or quality campaign)
python stage4/pipeline/run_stage4.py adaln_kitchen speed  <output-dir>
python stage4/pipeline/run_stage4.py adaln_kitchen quality <output-dir>
```

The AdaLN sidecar artifact is data, not code: `stage3/adaln-sidecar-light4-20260915/cache.safetensors`
(SHA-256 `8f794c60…dc34d0`, 204 modulation keys, bitwise-verified against the BF16 merged weights).

## Repository layout

```
stage4/pipeline/   integrated pipeline: patches, composer, deployment manifest, receipts, driver, demo
miniacc_core/      serving/eval harness (timing, resource guards, VBench adapter)
scripts/           retained runner, assembler, VBench scoring entry points
exp_configs/       frozen evaluation configurations
tests/             core unit tests
```

## Evidence governance

Source changes are hash-pinned (`stage4/pipeline/deployment-stage4.json`) and every admission
re-checks them. Campaign data, media, logs and measurement ledgers stay local and are
git-ignored by design; only the integrated pipeline code is tracked here.
