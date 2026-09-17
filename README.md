# MiniAcc

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![SGLang](https://img.shields.io/badge/SGLang-0.5.19-green)](https://github.com/sgl-project/sglang)
[![Model](https://img.shields.io/badge/MiniMax--H3-FL2VA-purple)](https://github.com/MiniMax-AI)
[![Adapter](https://img.shields.io/badge/LightX2V4-4--step-orange)]()
[![Hardware](https://img.shields.io/badge/RTX%204090%20%7C%20A100-tested-76B900)]()
[![Status](https://img.shields.io/badge/status-research-success)]()

**Making large audio–video diffusion models practical on a single GPU.**
A research project on serving MiniMax-H3 (text → 5-second video with synchronized audio)
from one workstation, roughly 2.5× faster than the stock serving path.

---

## What this project studies

Serving a model this size from one GPU means its weights cannot stay on the GPU — they are
copied from main memory to the GPU layer by layer on **every request**. Our central finding is
that this copying, not the neural-network math, dominates the time. Everything in the
integrated pipeline follows from that:

| Technique | What it does in plain terms |
|---|---|
| **AdaLN sidecar** | Precomputes small per-layer conditioning values once instead of recomputing them on every layer of every pass |
| **Kitchen INT8** | Stores weights as 8-bit integers, halving the bytes copied to the GPU |
| **SageAttention** (default) | Computes attention scores in 8-bit integer math, cutting the remaining compute share |

## Demo

Same prompt, same seed, baseline vs pipeline:

![Control vs pipeline vs pipeline+Sage](pipeline/demo/trio-0195-montage.jpg)

| Baseline | Integrated pipeline |
|---|---|
| <video src="https://github.com/TY-Yang3015/MiniAcc/raw/main/pipeline/demo/vbench-0195-control.mp4" controls width="100%"></video> | <video src="https://github.com/TY-Yang3015/MiniAcc/raw/main/pipeline/demo/vbench-0195-adaln_kitchen.mp4" controls width="100%"></video> |

More clips from the pipeline across four research prompts:

![Pipeline demo montage](pipeline/demo/pipeline-montage.jpg)

All clips: 1344×768, 124 frames, 24 FPS, stereo 32 kHz, full audio, validated end-to-end.

## Results

Time from submitting a request to the finished video file on disk (one 5.2-second
audio–video request, RTX 4090):

| Configuration | Time | Speedup |
|---|---:|---:|
| Stock serving (baseline) | 326.5 s | 1.00× |
| AdaLN + Kitchen INT8 | 149.3 s | 2.19× |
| **AdaLN + Kitchen INT8 + SageAttention (default)** | **130.2 s** | **2.51×** |
| AdaLN + Kitchen INT8 + Sol (sparse attention) | 151.4 s | 2.16× |

Paired video-quality scores against the AdaLN + Kitchen pipeline (16 clips, six VBench
metrics at n=16 and overall consistency at n=4; negative = worse than the pipeline):

| Metric | Sage (default) Δ | Sol Δ |
|---|---:|---:|
| Subject consistency | +0.14 | +0.63 |
| Background consistency | −0.31 | −0.51 |
| Motion smoothness | +0.91 | −0.03 |
| Dynamic degree | +6.25 | 0.00 |
| Aesthetic quality | +1.60 | +1.28 |
| Imaging quality | **−4.71** | −0.46 |
| Overall consistency | **−29.82** | +0.20 |

**How to read this:** the default configuration is the fastest, and it pays a measurable
quality cost in overall consistency and imaging sharpness (Sage's 8-bit attention stacked on
8-bit weights drifts the distilled model off its trained trajectory). If fidelity matters
more than the last 1.15×, drop Sage and run AdaLN + Kitchen alone — quality there is
unchanged from baseline within measurement noise. Sol is quality-neutral but slower.

## Why the pieces only work together

Individually, attention-level optimizations changed almost nothing (the copying hid them).
Once weight traffic halved, attention became a third of the remaining work, and 8-bit
attention finally paid. Sparse attention (Sol) gets no such rescue: its operator is slower
than the dense one it replaces. The full analysis is in the local technical report.

## Resource budgets

| Budget | Contract |
|---|---|
| Serving GPU | 1 × RTX 4090 24 GB or 1 × A100 80 GB, one server per device |
| GPU headroom | ≥ 8 GiB free on the assigned device at all times (checked continuously, run aborts otherwise) |
| Host memory | ≥ 16 GiB available enforced throughout; plan ≥ 64 GB during model load |
| Disk | ≈ 120 GB for model weights and caches; the AdaLN sidecar is 77.6 MB (hash-pinned) |
| Timing | request-to-file wall time, including weight streaming, decoding and saving; setup and warmup always excluded |
| Quality | frozen VBench metrics, scored in like-for-like pairs only (same hardware, same seed) |

## Quickstart

```bash
# 1. Apply the runtime patches to your SGLang 0.5.19 installation
patch -p1 < pipeline/patches/0001-adaln-kitchen-guard-relaxation.patch
patch -p1 < pipeline/patches/0002-stacking-relaxation-sage.patch
patch -p1 < pipeline/patches/0003-sol-stacking-relaxation.patch

# 2. Check the pinned source manifest against your runtime
python -c "import json;print(len(json.load(open('pipeline/deployment.json'))['installed']),'files pinned')"

# 3. Run the pipeline (default includes SageAttention)
python pipeline/run_stage4.py adaln_kitchen_sage speed   <output-dir>
python pipeline/run_stage4.py adaln_kitchen_sage quality <output-dir>
# Without Sage:
python pipeline/run_stage4.py adaln_kitchen speed <output-dir>
```

## Repository layout

```
pipeline/          the integrated pipeline: patches, composer, source manifest, receipts, driver, demo
miniacc_core/      serving and evaluation harness (timing, resource guards, VBench scoring)
scripts/           runner, media assembler, scoring entry points
exp_configs/       frozen evaluation configurations
tests/             core unit tests
```

## Research notes

- Every source change is hash-pinned and re-verified before any measurement counts.
- Campaign data, media archives, and full measurement ledgers stay local and are not
  tracked in git; only the pipeline code and demonstration clips live here.
- This is a research codebase, not a product: interfaces may change between findings.
