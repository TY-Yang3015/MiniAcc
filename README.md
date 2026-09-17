# MiniAcc

Local-deployable MiniMax-H3 (SGLang 0.5.19, LightX2V4 4-step distilled adapter).

## Integrated pipeline (Stage 4, 2026-09-17)

The production serving path is the AdaLN + Kitchen INT8 integrated pipeline in
[`stage4/pipeline/`](stage4/pipeline/README.md): per-layer AdaLN modulation sidecar plus
Kitchen INT8 weight-only quantization, hard-coded on the frozen LightX2V4 baseline
(1344×768, 124 frames, 24 FPS, stereo 32 kHz).

Optional SageAttention composition is wired but not recommended (paired quality evaluation
showed semantic/image degradation on this pipeline); the recommended default is AdaLN+Kitchen
without Sage. Measured results are kept local and are not tracked in git.

- `stage4/pipeline/` — runtime patches, composition builder, deployment manifest, re-bound
  source receipts, driver, and the full integration README.
- `miniacc_core/`, `scripts/`, `exp_configs/` — serving/eval harness and frozen VBench path.
- `tests/` — core unit tests.
- Experiment data (campaign roots, media, logs, receipts, results) stays local and is
  git-ignored; evidence hashes live in `stage4/stage4-acceptance.json` locally.
