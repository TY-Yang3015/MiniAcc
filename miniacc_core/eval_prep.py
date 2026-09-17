"""CPU-only preparation of VBench scoring manifests.

This module discovers already-generated clips and records eligibility; it never
loads a VBench scorer, torch, CUDA, or a visual model. Missing/error values are
omitted rather than converted to zero.  The official scorer remains a later,
explicit Stage 5 action.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .config import DEVELOPMENT_DIMENSIONS

CUSTOM_DIMENSIONS = tuple(DEVELOPMENT_DIMENSIONS[:-1])
STANDARD_DIMENSIONS = ("overall_consistency",)
# Official VBench script constants used only for future per-metric normalization.
NORMALIZATION = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "motion_smoothness": (0.706, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "overall_consistency": (0.0, 0.364),
}


def normalize_metric(metric: str, raw: float) -> float:
    """Normalize one supplied score; do not aggregate or fill absent scores."""
    if metric not in NORMALIZATION:
        raise ValueError(f"unknown VBench metric: {metric}")
    value = float(raw)
    low, high = NORMALIZATION[metric]
    return 100.0 * (value - low) / (high - low)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe(path: Path, ffprobe: Path | None) -> dict[str, Any] | None:
    if ffprobe is None:
        return None
    command = [str(ffprobe), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode:
        return {"status": "error", "stderr": completed.stderr.strip() or None}
    try:
        return {"status": "ok", **json.loads(completed.stdout)}
    except json.JSONDecodeError as error:
        return {"status": "error", "stderr": str(error)}


def _media_check(path: Path, ffprobe: Path | None,
                 expected_sha256: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        result["status"] = "missing"
        return result
    result["sha256"] = _sha256(path)
    if expected_sha256 is not None:
        result["expected_sha256"] = expected_sha256
        result["hash_match"] = result["sha256"] == expected_sha256
        if not result["hash_match"]:
            result["status"] = "invalid"
            result["error"] = "recovery output SHA-256 does not match the media"
            return result
    probe = _probe(path, ffprobe)
    if probe is None:
        result["status"] = "unprobed"
        return result
    result["ffprobe"] = probe
    streams = probe.get("streams", []) if probe.get("status") == "ok" else []
    result["status"] = "validated" if (
        any(s.get("codec_type") == "video" for s in streams)
        and any(s.get("codec_type") == "audio" for s in streams)
    ) else "invalid"
    return result


def _recovery_media(root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    recovery = root / "validation-recovery.json"
    if recovery.is_file():
        try:
            record = json.loads(recovery.read_text())
            output = Path(record["output"])
            # Recovery records may be copied from wolf8 with an absolute remote
            # path; resolve against the artifact root when evaluating a copy.
            if not output.is_file():
                candidates = list((root / "media").rglob("*.mp4")) if (root / "media").is_dir() else []
                # A copied recovery is safe only when it has one unambiguous output.
                output = candidates[0] if len(candidates) == 1 else output
            expected = record.get("output_sha256")
            record["output_hash_present"] = isinstance(expected, str) and len(expected) == 64
            return output, record
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return None, None
    return None, None


def _result_media(root: Path, result: dict[str, Any]) -> Path | None:
    outputs = result.get("outputs", [])
    if outputs and isinstance(outputs[0], dict) and outputs[0].get("path"):
        recorded = Path(outputs[0]["path"])
        # Remote probes record their absolute project path.  When an artifact
        # bundle is copied to another checkout, use its own media directory
        # rather than treating the stale recorded path as missing.
        if recorded.is_file():
            return recorded
    media = root / "media"
    return next(media.rglob("*.mp4"), None) if media.is_dir() else None


PINNED_VBENCH_REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"


def load_eval_config(path: Path, manifest: dict[str, Any], *, scoring: bool = False) -> dict[str, Any]:
    """Validate a preparation or official-scoring profile against frozen metadata."""
    try:
        config = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid evaluation config {path}: {error}") from error
    expected_purpose = ("official-vbench-stage1-scoring" if scoring
                        else "official-vbench-stage1-preparation")
    expected_device, expected_gpu_count, expected_run_scorers = (
        ("cuda", 1, True) if scoring else ("cpu", 0, False)
    )
    if (config.get("purpose") != expected_purpose
            or config.get("device") != expected_device
            or config.get("gpu_count") != expected_gpu_count
            or config.get("run_scorers") is not expected_run_scorers):
        mode = "scoring" if scoring else "CPU-only preparation"
        raise ValueError(f"evaluation config is not a valid {mode} profile")
    if config.get("tiers") != [4, 8, 16]:
        raise ValueError("evaluation config tiers must be [4, 8, 16]")
    if config.get("vbench_revision") != PINNED_VBENCH_REVISION:
        raise ValueError("evaluation config is not pinned to the required VBench revision")
    expected_metrics = list(DEVELOPMENT_DIMENSIONS)
    if config.get("metrics") != expected_metrics:
        raise ValueError("evaluation config metrics must match the frozen seven-metric order")
    if PINNED_VBENCH_REVISION not in manifest.get("source", {}).get("url", ""):
        raise ValueError("frozen manifest is not the pinned VBench source revision")
    source_root = Path(config.get("source_root", "")).expanduser()
    metadata = source_root / "vbench" / "VBench_full_info.json"
    if not metadata.is_file():
        raise ValueError(f"pinned VBench metadata is unavailable: {metadata}")
    if _sha256(metadata) != manifest["source"]["sha256"]:
        raise ValueError("pinned VBench metadata hash differs from frozen manifest")
    return config


def prepare_manifest(artifact_root: Path, frozen_manifest: Path, *, ffprobe: Path | None = None,
                     eval_config: Path | None = None) -> dict[str, Any]:
    """Build a score-ready ledger for current independent-job artifacts."""
    source = json.loads(Path(frozen_manifest).read_text())
    config = load_eval_config(eval_config, source) if eval_config else None
    prompts = {item["id"]: item for item in source["prompts"]}
    jobs = {item["id"]: item for item in source["jobs"]}
    entries = []
    for root in sorted(Path(artifact_root).glob("gpu-*")):
        result_path = root / "result.json"
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        recovery_path, recovery = _recovery_media(root)
        output = recovery_path or _result_media(root, result)
        job_id = result.get("job", {}).get("id")
        if not job_id:
            # The per-job command is immutable and its job ID is the only safe fallback.
            candidates = [job_id for job_id in jobs if job_id in root.name]
            job_id = candidates[0] if len(candidates) == 1 else None
        job = jobs.get(job_id)
        if job is None:
            entries.append({"artifact_root": str(root), "status": "error", "error": "job mapping unavailable"})
            continue
        prompt = prompts[job["prompt_id"]]
        rows = prompt.get("official_metadata_rows", [])
        overall_eligible = any("overall_consistency" in row.get("dimension", []) for row in rows)
        expected_hash = recovery.get("output_sha256") if recovery else None
        media = (_media_check(output, ffprobe, expected_hash) if output else
                 {"status": "missing", "path": None})
        if recovery and not recovery.get("output_hash_present"):
            media["status"] = "invalid"
            media["error"] = "recovery record has no verifiable output SHA-256"
        if recovery and media.get("status") == "unprobed" and media.get("hash_match", True):
            # Reuse the separately captured ffprobe metadata without claiming a
            # full codec decode; that requires a later explicit ffmpeg check.
            media["status"] = "metadata_validated_from_recovery"
            media["ffprobe"] = {"status": "recovery", "streams": recovery.get("streams", []),
                                 "format": recovery.get("format", {})}
            media["full_decode"] = "not_established"
            media["recovery_record"] = str((root / "validation-recovery.json").resolve())
        entries.append({
            "artifact_root": str(root.resolve()),
            "job_id": job["id"], "prompt_id": job["prompt_id"], "seed": job["seed"],
            "prompt_en": prompt["prompt_en"], "stratum": prompt["stratum"],
            "allocation": job.get("allocation"), "sample_index": job.get("sample_index"),
            "output": str(output.resolve()) if output else None,
            "generation_status": result.get("status", "missing_result"),
            "media": media,
            "official_metadata_rows": rows,
            "eligible_metrics": {
                "custom_input": list(CUSTOM_DIMENSIONS),
                "overall_consistency": overall_eligible,
            },
            "scores": {"raw": {}, "normalized": {}},
            "recovery_record": str((root / "validation-recovery.json").resolve()) if recovery else None,
        })
    entries.sort(key=lambda item: (item.get("prompt_id", ""), item.get("seed", 0)))
    eligible_counts = {
        metric: sum(1 for entry in entries
                    if entry.get("media", {}).get("status") in {"validated", "metadata_validated_from_recovery"}
                    and (metric in entry.get("eligible_metrics", {}).get("custom_input", [])
                         or entry.get("eligible_metrics", {}).get(metric) is True))
        for metric in (*CUSTOM_DIMENSIONS, *STANDARD_DIMENSIONS)
    }
    errors = [
        {"job_id": entry.get("job_id"), "status": entry.get("media", {}).get("status"),
         "error": entry.get("error")}
        for entry in entries
        if entry.get("status") == "error" or entry.get("media", {}).get("status") in {"missing", "invalid"}
    ]
    generation_failures = [
        {"job_id": entry.get("job_id"), "status": entry.get("generation_status"),
         "result_note": "historical probe result retained; recovery media is separate"}
        for entry in entries
        if entry.get("generation_status") not in {None, "success", "missing_result"}
    ]
    accepted_media = {"validated", "metadata_validated_from_recovery"}
    media_ready = bool(entries) and all(
        entry.get("media", {}).get("status") in accepted_media for entry in entries
    ) and not errors
    return {
        "schema_version": 1,
        "status": "ready_for_scoring" if media_ready else ("no_outputs" if not entries else "blocked_invalid_media"),
        "source_manifest": str(Path(frozen_manifest).resolve()),
        "source_manifest_sha256": _sha256(Path(frozen_manifest)),
        "evaluation_config": str(Path(eval_config).resolve()) if eval_config else None,
        "vbench_source": source.get("source"),
        "metrics": {"custom_input": list(CUSTOM_DIMENSIONS), "standard_metadata": list(STANDARD_DIMENSIONS)},
        "evaluation_profile": config,
        "funnel": {"tiers": [4, 8, 16], "available_prompt_count": len({e.get("prompt_id") for e in entries if e.get("prompt_id")})},
        "eligible_counts": eligible_counts,
        "errors": errors,
        "historical_generation_failures": generation_failures,
        "score_counts": {"raw": 0, "normalized": 0},
        "entries": entries,
        "limitations": [
            "No scores are fabricated or zero-filled.",
            "No scalar quality index or official total is computed.",
            "overall_consistency requires the standard evaluator and original prompt metadata.",
            "Five normalized points are an alert for paired review, not statistical rejection.",
        ],
    }


def write_manifest(output: Path, value: dict[str, Any]) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(output).with_suffix(Path(output).suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
