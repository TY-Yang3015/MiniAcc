# VBench evaluation preparation

This package prepares a CPU-only, score-ready ledger; it does not run a scorer or load a GPU model. After explicit authorization, the next action is Stage 1 Q0 scoring, followed by the 4/8/16 development comparisons. Stage 5 is reserved for the final four-suite assessment.

## Frozen authority

- Plan: `execution_plan.pdf`, Stage 1 funnel tiers 4/8/16 and seven development metrics.
- Frozen input: `data/stage1/vbench_dev_manifest.json` (paired prompt/seed IDs and original metadata).
- Pinned VBench source: revision `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`, whose source SHA-256 is recorded in the manifest/source audit.
- Source requirements are in the pinned checkout `requirements.txt`; they include Pillow, numpy<2, timm, scipy, opencv-python, decord, transformers==4.33.2, pyiqa, fairscale and related scorer dependencies. Install these only in a separate evaluator environment when scoring is explicitly authorized; do not modify the inference environment.

## Metrics and eligibility

The six custom-input metrics are subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality, and imaging quality. They consume the generated video plus the original prompt. `overall_consistency` is separate: it uses the standard evaluator and only prompts with the official original prompt metadata row. Eligibility and denominator are recorded per metric; absent/error scores remain absent and are never zero-filled. Raw and per-metric normalized scores are paired by prompt ID and seed. No official total or scalar quality index is computed. A five-point normalized decline is an alert for paired review, not statistical rejection.

## Preparation command

Use the preparation-only profile `exp_configs/vbench-eval-cpu.yaml` and a local artifact copy after generation is complete. This is Stage 1 preparation, not the Stage 5 four-suite comparison:

```bash
python scripts/prepare_vbench_eval.py \
  --artifacts artifacts/wolf8-independent-03 \
  --manifest data/stage1/vbench_dev_manifest.json \
  --config exp_configs/vbench-eval-cpu.yaml \
  --ffprobe /absolute/path/to/ffprobe \
  --output artifacts/evaluation/q0-prepared.json
```

Omit `--ffprobe` for a discovery-only dry run; that ledger is blocked from scoring until media is probed. Recovery records must include an output SHA-256 matching the copied clip. With recovered records, the adapter can reuse captured ffprobe metadata but labels full codec decode as unestablished; ffprobe metadata alone is not a full decode test. The adapter preserves each immutable generation result and separately records recovery validation.

After preparation and an idle-GPU check, run one native dimension at a time in the separate scoring environment. This command is Stage 1 Q0 cached rescoring (no Q0 inference timing and no reference regeneration):

```bash
python scripts/score_vbench.py \
  --prepared artifacts/evaluation/q0-prepared.json \
  --config exp_configs/vbench-eval-scoring.yaml \
  --dimension subject_consistency \
  --output artifacts/evaluation/q0-scored
```

Repeat the command for the seven dimensions, retaining `result.json` and `scored-ledger.json` after every attempt. The adapter calls the pinned upstream `compute_<dimension>` function, records its aggregate and native per-video details, computes only the pinned per-metric normalization, and records source/module/checkpoint hashes plus eligible/scored/missing/error denominators. It never computes an official total, scalar Qdev, or fills missing values. Stage 5 remains reserved for the final four-suite assessment and must not be conflated with this Stage 1 Q0 scoring. The read-only four-family support and budget ledger is `artifacts/evaluation/four-family-duration-ledger.json`; its 480-second clip values are explicitly source-backed planning assumptions, not measured candidate speeds. No comparator is launched from that ledger.
