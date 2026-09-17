"""Adapter for the pinned, native VBench dimension functions.

The adapter deliberately does not reimplement metric math.  It supplies each
upstream ``compute_<dimension>`` function with frozen prompt metadata, retains
its aggregate and per-video details, and records the source/checkpoint hashes
needed to reproduce a cached-Q0 rescore.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

from .eval_prep import CUSTOM_DIMENSIONS, PINNED_VBENCH_REVISION, STANDARD_DIMENSIONS, normalize_metric

ALL_DIMENSIONS = (*CUSTOM_DIMENSIONS, *STANDARD_DIMENSIONS)


def scorer_exit_code(status: str) -> int:
    """Only a complete denominator is a successful scorer process."""
    return 0 if status == "completed" else 1

ACCEPTED_MEDIA = {"validated", "metadata_validated_from_recovery"}
# The upstream imaging scorer returns its aggregate divided by 100 but leaves
# per-video details in the original 0--100 MUSIQ domain.
DETAIL_TO_AGGREGATE_SCALE = {"imaging_quality": 100.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _source_revision(source_root: Path) -> tuple[str, str]:
    """Resolve the pin without pretending a copied checkout has a Git object."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        verification = "git_rev_parse"
    except (OSError, subprocess.SubprocessError):
        # Deployment copies intentionally omit .git.  The immutable cache
        # directory name and metadata hash still bind this source copy to the
        # requested release, but are weaker evidence than a Git object.
        if not source_root.name.endswith(PINNED_VBENCH_REVISION):
            raise ValueError("VBench source has no Git pin and its directory is not revision-named")
        revision, verification = PINNED_VBENCH_REVISION, "revision_named_checkout"
    if revision != PINNED_VBENCH_REVISION:
        raise ValueError(f"VBench source revision {revision!r} is not pinned revision {PINNED_VBENCH_REVISION}")
    return revision, verification


def _path_values(value: Any) -> list[Path]:
    if isinstance(value, dict):
        paths: list[Path] = []
        for item in value.values():
            paths.extend(_path_values(item))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for item in value:
            paths.extend(_path_values(item))
        return paths
    if isinstance(value, str):
        path = Path(value).expanduser()
        if path.is_file():
            return [path]
    return []


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_provenance(source_root: Path, dimension: str, submodules: Any, module: Any) -> dict[str, Any]:
    """Capture exact source revision, scorer module and resolved checkpoint hashes."""
    module_path = Path(module.__file__).resolve()
    files = {str(module_path): sha256_file(module_path)}
    for path in _path_values(submodules):
        files[str(path.resolve())] = sha256_file(path)
    metadata = source_root / "vbench" / "VBench_full_info.json"
    files[str(metadata.resolve())] = sha256_file(metadata)
    revision, revision_verification = _source_revision(source_root)
    return {
        "source_root": str(source_root.resolve()),
        "source_revision": revision,
        "source_revision_verification": revision_verification,
        "source_metadata_sha256": files[str(metadata.resolve())],
        "scorer_module": str(module_path),
        "scorer_module_sha256": files[str(module_path)],
        "model_files_sha256": {key: files[key] for key in sorted(files) if key != str(module_path.resolve()) and key != str(metadata.resolve())},
    }


def dimension_metadata(ledger: dict[str, Any], dimension: str) -> list[dict[str, Any]]:
    """Convert prepared entries to the exact one-row-per-prompt upstream input."""
    if dimension not in ALL_DIMENSIONS:
        raise ValueError(f"unsupported development dimension: {dimension}")
    if ledger.get("status") not in {"ready_for_scoring", "partial_scoring", "completed_scoring", "completed"}:
        raise ValueError(f"prepared ledger is not ready for scoring: {ledger.get('status')}")
    rows = []
    for entry in ledger.get("entries", []):
        if entry.get("media", {}).get("status") not in ACCEPTED_MEDIA:
            continue
        if dimension != "overall_consistency":
            eligible = dimension in entry.get("eligible_metrics", {}).get("custom_input", [])
            row = {"prompt_en": entry["prompt_en"], "dimension": [dimension],
                   "video_list": [entry["output"]]}
        else:
            eligible = entry.get("eligible_metrics", {}).get("overall_consistency") is True
            metadata_rows = [row for row in entry.get("official_metadata_rows", [])
                             if dimension in row.get("dimension", [])]
            row = dict(metadata_rows[0], video_list=[entry["output"]]) if metadata_rows else {}
        if eligible and row:
            rows.append(row)
    if not rows:
        raise ValueError(f"no eligible media for {dimension}")
    return rows


def _detail_rows(details: Any) -> list[dict[str, Any]]:
    if not isinstance(details, list):
        return []
    return [row for row in details if isinstance(row, dict) and "video_path" in row
            and "video_results" in row]


def _entry_for_path(ledger: dict[str, Any], path: str) -> dict[str, Any] | None:
    resolved = str(Path(path).expanduser().resolve())
    matches = [entry for entry in ledger.get("entries", []) if entry.get("output") and
               str(Path(entry["output"]).expanduser().resolve()) == resolved]
    if len(matches) == 1:
        return matches[0]
    return None


class OfficialVBenchAdapter:
    """Run one native VBench scorer dimension with durable provenance."""

    def __init__(self, source_root: Path, *, device: str = "cuda", read_frame: bool = False):
        self.source_root = Path(source_root).expanduser().resolve()
        self.device_name = device
        self.read_frame = read_frame

    def score_dimension(self, ledger: dict[str, Any], dimension: str, output_dir: Path) -> dict[str, Any]:
        if self.device_name != "cuda":
            raise ValueError("official VBench scoring profile must use cuda")
        metadata = dimension_metadata(ledger, dimension)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / "input.json"
        _atomic_json(input_path, metadata)
        started = time.monotonic()
        result: dict[str, Any] = {"dimension": dimension, "status": "failed", "input": str(input_path.resolve())}
        try:
            import torch
            from vbench.utils import init_submodules
            result["runtime"] = {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": getattr(torch.version, "cuda", None),
                "packages": {name: importlib.metadata.version(name) for name in (
                    "torchvision", "decord", "opencv-python", "openai-clip", "pyiqa",
                    "transformers", "timm", "fairscale")
                    if _package_version(name) is not None},
            }
            importlib.invalidate_caches()
            module = importlib.import_module(f"vbench.{dimension}")
            compute = getattr(module, f"compute_{dimension}")
            submodules = init_submodules([dimension], local=True, read_frame=self.read_frame)[dimension]
            result["provenance"] = model_provenance(self.source_root, dimension, submodules, module)
            with torch.inference_mode():
                raw, details = compute(str(input_path), torch.device(self.device_name), submodules)
            raw = float(raw)
            if not __import__("math").isfinite(raw):
                raise ValueError("native scorer returned a non-finite aggregate")
            result.update({
                "status": "completed", "raw": raw,
                "normalized": normalize_metric(dimension, raw),
                "upstream_details": details,
                "elapsed_seconds_including_backbone_load": time.monotonic() - started,
                "device": self.device_name,
                "videos_submitted": sum(len(row["video_list"]) for row in metadata),
            })
            details_by_path = _detail_rows(details)
            scored = 0
            missing = 0
            errors = []
            for detail in details_by_path:
                entry = _entry_for_path(ledger, detail["video_path"])
                value = detail["video_results"]
                try:
                    value = float(value) / DETAIL_TO_AGGREGATE_SCALE.get(dimension, 1.0)
                    if not __import__("math").isfinite(value):
                        raise ValueError("non-finite per-video score")
                except (TypeError, ValueError) as error:
                    errors.append({"video_path": detail["video_path"], "error": str(error)})
                    continue
                if entry is None:
                    errors.append({"video_path": detail["video_path"], "error": "detail path did not map to one ledger entry"})
                    continue
                entry.setdefault("scores", {"raw": {}, "normalized": {}})
                entry["scores"].setdefault("raw", {})[dimension] = value
                entry["scores"].setdefault("normalized", {})[dimension] = normalize_metric(dimension, value)
                scored += 1
            eligible = [row for row in ledger["entries"] if row.get("media", {}).get("status") in ACCEPTED_MEDIA
                        and ((dimension != "overall_consistency" and dimension in row.get("eligible_metrics", {}).get("custom_input", []))
                             or (dimension == "overall_consistency" and row.get("eligible_metrics", {}).get("overall_consistency") is True))]
            missing = len(eligible) - scored
            result["denominator"] = {"eligible": len(eligible), "scored": scored, "missing": max(missing, 0), "errors": len(errors)}
            result["per_prompt"] = [{"prompt_id": entry.get("prompt_id"), "seed": entry.get("seed"),
                                     "raw": entry.get("scores", {}).get("raw", {}).get(dimension),
                                     "normalized": entry.get("scores", {}).get("normalized", {}).get(dimension)}
                                    for entry in eligible]
            result["errors"] = errors
            if missing or errors:
                result["status"] = "completed_with_missing"
        except Exception as error:
            result.update({"error_type": type(error).__name__, "error": str(error),
                           "elapsed_seconds_including_backbone_load": time.monotonic() - started})
        _atomic_json(output_dir / "result.json", result)
        return result


def update_scored_ledger(ledger: dict[str, Any], output: Path, *,
                         preparation_profile: dict[str, Any] | None = None,
                         scoring_profile: dict[str, Any] | None = None) -> None:
    """Write a reproducible score copy; the prepared input remains immutable.

    ``ready_for_scoring`` and prior score-ledger statuses are accepted so each
    dimension can be run independently and resumed after an interruption.
    """
    if ledger.get("status") not in {"ready_for_scoring", "partial_scoring", "completed_scoring", "completed"}:
        raise ValueError(f"cannot update score ledger from status {ledger.get('status')!r}")
    dimensions = {dimension for entry in ledger.get("entries", [])
                  for dimension in entry.get("scores", {}).get("raw", {})}
    ledger["score_counts"] = {
        "raw": sum(1 for entry in ledger.get("entries", [])
                    for _dimension in entry.get("scores", {}).get("raw", {})),
        "normalized": sum(1 for entry in ledger.get("entries", [])
                           for _dimension in entry.get("scores", {}).get("normalized", {})),
    }
    ledger["scored_dimensions"] = sorted(dimensions)
    if preparation_profile is not None:
        ledger["preparation_profile"] = preparation_profile
    elif "preparation_profile" not in ledger and ledger.get("evaluation_profile") is not None:
        ledger["preparation_profile"] = ledger["evaluation_profile"]
    if scoring_profile is not None:
        ledger["scoring_profile"] = scoring_profile
    if dimensions == set(ALL_DIMENSIONS):
        ledger["status"] = "completed_scoring"
        ledger["scoring_status"] = {"status": "completed", "dimensions": sorted(dimensions)}
    elif dimensions:
        ledger["status"] = "partial_scoring"
        ledger["scoring_status"] = {"status": "partial", "dimensions": sorted(dimensions)}
    _atomic_json(Path(output), ledger)


def audit_dependency_provenance(source_root: Path, cache_dir: Path) -> dict[str, Any]:
    """Record post-run dependency audit without backdating missing hashes."""
    expected = {
        "aesthetic_head": cache_dir / "aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
        "overall_bpe": cache_dir / "ViCLIP/bpe_simple_vocab_16e6.txt.gz",
        "dino_code": cache_dir / "dino_model/facebookresearch_dino_main",
    }
    files = {}
    for name, path in expected.items():
        files[name] = {"path": str(path), "present_at_audit": path.exists()}
        if path.is_file():
            files[name]["sha256_at_audit"] = sha256_file(path)
        elif path.is_dir():
            files[name]["code_hash_at_audit"] = None
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "retroactive": True,
        "source_root": str(Path(source_root).resolve()),
        "note": "These checks were captured after the original scorer runs; absent hashes are not claimed retroactively.",
        "files": files,
    }
