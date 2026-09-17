# MiniAcc

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![SGLang](https://img.shields.io/badge/SGLang-0.5.19-green)](https://github.com/sgl-project/sglang)
[![Model](https://img.shields.io/badge/MiniMax--H3-FL2VA-purple)](https://github.com/MiniMax-AI)
[![Adapter](https://img.shields.io/badge/LightX2V4-4--step-orange)]()
[![Hardware](https://img.shields.io/badge/RTX%204090%20%7C%20A100%2080GB-tested-76B900)]()

**See the research blog at https://ty-yang3015.github.io/blog/2026/09/17/miniacc/ !**

**Investigating how to optimise large audio–video diffusion models at inference time on a single GPU with MiniMax-H3.**
In our current single-GPU measurements, the integrated pipeline finishes
a request about 2.5× sooner than the baseline pipeline (LightX2V 4-step).

---

## What this project studies

Serving a model this size from one GPU means its weights cannot stay on the GPU, as they are
copied from main memory to the GPU layer by layer on **every request**. The evidence we have
gathered so far consistently points to this copying, rather than the neural-network math, as
the dominant cost. The pipeline configurations we test are built around that observation:

| Technique | What it does in plain terms |
|---|---|
| **AdaLN sidecar** | Precomputes small per-layer conditioning values once instead of recomputing them on every layer of every pass |
| **Kitchen INT8** | Stores weights as 8-bit integers, halving the bytes copied to the GPU |
| **SageAttention** (default) | Computes attention scores in 8-bit integer math, cutting the remaining compute share |

## Demo

Same prompt, same seed — baseline (top) vs integrated pipeline (bottom):

![Control vs pipeline vs pipeline+Sage](pipeline/demo/trio-0195-montage.jpg)

**Baseline serving:**

![Baseline demo clip](pipeline/demo/demo-baseline.gif)

**Integrated pipeline (AdaLN + Kitchen INT8):**

![Pipeline demo clip](pipeline/demo/demo-pipeline.gif)

Uncurated frames from the pipeline across four research prompts:

![Pipeline demo montage](pipeline/demo/pipeline-montage.jpg)

(1344×768, 124 frames, 24 FPS, stereo 32 kHz, full audio)

## Results

Time from submitting a request to the finished video file on disk (one 5.2-second
audio–video request, RTX 4090):

| Configuration | Time | Speedup |
|---|---:|---:|
| Stock serving (baseline) | 326.5 s | 1.00× |
| AdaLN + Kitchen INT8 | 149.3 s | 2.19× |
| **AdaLN + Kitchen INT8 + SageAttention (default)** | **130.2 s** | **2.51×** |
| AdaLN + Kitchen INT8 + Sol (sparse attention) | 151.4 s | 2.16× |

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
for p in pipeline/patches/*.patch; do patch -p1 < "$p"; done

# 2. Check prerequisites (runtime, model, adapter, sidecar)
python pipeline/setup.py

# 3. Run the pipeline (one timed request, default includes SageAttention)
python pipeline/serve.py --output outputs/demo

# Fidelity-first (no Sage) or a quality cohort:
python pipeline/serve.py --no-sage --output outputs/demo-nosage
python pipeline/serve.py --quality --output outputs/demo-quality
```

## Repository layout

```
pipeline/          the integrated pipeline: patches, composer, source manifest, receipts, driver, demo
miniacc_core/      serving and evaluation harness (timing, resource guards, VBench scoring)
scripts/           runner, media assembler, scoring entry points
exp_configs/       frozen evaluation configurations
```
