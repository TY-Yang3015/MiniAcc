#!/usr/bin/env python3
"""Prepare, but do not execute, the pinned VBench score ledger."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miniacc_core.eval_prep import prepare_manifest, write_manifest  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/stage1/vbench_dev_manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("exp_configs/vbench-eval-cpu.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, help="Absolute evaluator ffprobe; omit for dry preparation")
    args = parser.parse_args(argv)
    value = prepare_manifest(args.artifacts, args.manifest, ffprobe=args.ffprobe,
                             eval_config=args.config)
    write_manifest(args.output, value)
    print(f"prepared {len(value['entries'])} entries at {args.output}; no scorer executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
