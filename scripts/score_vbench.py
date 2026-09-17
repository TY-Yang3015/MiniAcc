#!/usr/bin/env python3
"""Run one official pinned VBench dimension against a cached prepared ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from miniacc_core.eval_prep import _sha256, load_eval_config  # noqa: E402
from miniacc_core.vbench_scorer import OfficialVBenchAdapter, scorer_exit_code, update_scored_ledger  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True,
                        help="CPU preparation ledger; it is never modified")
    parser.add_argument("--config", type=Path, default=ROOT / "exp_configs/vbench-eval-scoring.yaml")
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="dimension output directory; also receives scored-ledger.json")
    args = parser.parse_args(argv)

    ledger = json.loads(args.prepared.read_text())
    existing_ledger = args.output / "scored-ledger.json"
    if existing_ledger.is_file():
        candidate = json.loads(existing_ledger.read_text())
        if candidate.get("source_manifest_sha256") != ledger.get("source_manifest_sha256"):
            raise ValueError("existing scored ledger belongs to a different frozen manifest")
        ledger = candidate
    manifest = json.loads(Path(ledger["source_manifest"]).read_text())
    config = load_eval_config(args.config, manifest, scoring=True)
    if ledger.get("source_manifest_sha256") != _sha256(Path(ledger["source_manifest"])):
        raise ValueError("prepared ledger source manifest hash changed")
    source_root = Path(config["source_root"]).expanduser()
    # The official checkout must be first on sys.path so native imports cannot
    # silently resolve to an installed or different VBench revision.
    sys.path.insert(0, str(source_root))
    adapter = OfficialVBenchAdapter(source_root, device=config["device"],
                                    read_frame=bool(config.get("read_frame", False)))
    result = adapter.score_dimension(ledger, args.dimension, args.output / args.dimension)
    update_scored_ledger(ledger, args.output / "scored-ledger.json",
                         scoring_profile=config)
    print(json.dumps({"dimension": args.dimension, "status": result["status"],
                      "result": str((args.output / args.dimension / "result.json").resolve()),
                      "ledger": str((args.output / "scored-ledger.json").resolve())}))
    # Partial native output is durable but is a failed command for orchestration;
    # callers must not mistake an incomplete denominator for a completed score.
    return scorer_exit_code(result["status"])


if __name__ == "__main__":
    raise SystemExit(main())
