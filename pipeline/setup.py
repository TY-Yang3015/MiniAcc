#!/usr/bin/env python3
"""Prerequisite check for the MiniAcc integrated pipeline.

Verifies (without downloading anything) that the runtime, model artifacts and
data dependencies the pipeline needs are present, and copies the frozen
deployment manifest into place if it is missing. Run this before serve.py.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('/home/arezy/.cache/miniacc/stage3-sglang-runtime/sglang-0.5.19-cu130-cp312')
CHECKS = [
    ('SGLang 0.5.19 runtime', RUNTIME / 'bin/sglang-local', None),
    ('MiniMax-H3 FL2VA snapshot', ROOT / '.local/sglang-hf-cache/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA/model_index.json', None),
    ('LightX2V4 turbo 4-step adapter', ROOT / '.local/sglang-adapters/minimax_h3_fl2v_turbo_4step_v0.1.safetensors', '5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29'),
    ('AdaLN sidecar', ROOT / 'stage3/adaln-sidecar-light4-20260915/cache.safetensors', '8f794c6049b8fdfbb85793be29fe79e15ce02a3d00bbd729c2137bdadcdc34d0'),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    problems = 0
    for name, path, expected in CHECKS:
        if not path.is_file():
            print(f'MISSING  {name}: {path}')
            problems += 1
            continue
        if expected and sha256(path) != expected:
            print(f'HASH     {name}: content does not match the pinned SHA-256')
            problems += 1
            continue
        print(f'ok       {name}')
    manifest = ROOT / 'stage3/followup-20260916/deployment-v2.json'
    reference = Path(__file__).with_name('receipts') / 'deployment-v2.reference.json'
    if manifest.is_file():
        print('ok       frozen deployment manifest already in place')
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference, manifest)
        print(f'ok       frozen deployment manifest copied to {manifest}')
    if problems:
        print(f'\n{problems} prerequisite(s) missing. See pipeline/README.md#prerequisites.')
        return 1
    print('\nAll prerequisites satisfied. Next: python pipeline/serve.py --output outputs/demo')
    return 0


if __name__ == '__main__':
    sys.exit(main())
