"""Frozen prompt/data boundary. This module never imports CUDA or model code."""

from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Callable

from .config import DEVELOPMENT_DIMENSIONS, SEEDS, SELECTION_NAMESPACE, STRATA

ROOT = Path(__file__).resolve().parents[1]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path | None, value: dict) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def command_output(args: list[str]) -> dict:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "LC_ALL": "C"},
            check=False,
        )
    except FileNotFoundError:
        return {"status": "not_installed", "stdout": ""}
    except subprocess.TimeoutExpired:
        return {"status": "timed_out", "stdout": ""}
    return {
        "status": "ok" if result.returncode == 0 else "command_failed",
        "stdout": result.stdout.strip() if result.returncode == 0 else "",
    }


def parse_gpu_inventory(text: str, reserve_mib: int = 2048) -> list[dict]:
    if reserve_mib < 2048:
        raise ValueError("At least 2048 MiB of additional free VRAM must remain")
    devices = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if len(row) != 8:
            raise ValueError("Unexpected nvidia-smi CSV column count")
        index, name, capability, total, used, free, driver, power = map(str.strip, row)
        total, used, free = (int(float(value)) for value in (total, used, free))
        if min(total, used, free) < 0 or max(used, free) > total:
            raise ValueError("Invalid GPU memory readings")
        devices.append(
            {
                "index": int(index),
                "name": name,
                "compute_capability": capability,
                "is_target_standard_4090": name == "NVIDIA GeForce RTX 4090"
                and capability == "8.9",
                "memory_total_mib": total,
                "memory_used_mib": used,
                "memory_free_mib": free,
                "additional_free_reserve_mib": reserve_mib,
                "snapshot_incremental_budget_mib": max(0, free - reserve_mib),
                "driver_version": driver,
                "power_limit_w_reported": power,
            }
        )
    return devices


def preflight() -> dict:
    memory = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory[key] = int(value.split()[0]) * 1024
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    gpu_result = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,memory.total,memory.used,memory.free,driver_version,power.limit",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = parse_gpu_inventory(gpu_result["stdout"])
    packages = {}
    for name in (
        "torch",
        "triton",
        "diffusers",
        "transformers",
        "accelerate",
        "sglang",
        "vbench",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    tools = {}
    for name, args in {
        "git": ["git", "--version"],
        "ffmpeg": ["ffmpeg", "-version"],
        "nvcc": ["nvcc", "--version"],
    }.items():
        result = command_output(args)
        lines = result["stdout"].splitlines()
        tools[name] = {
            "status": result["status"],
            "version": next(
                (line for line in lines if "release" in line),
                lines[0] if lines else None,
            ),
        }
    disk = shutil.disk_usage(ROOT)
    warnings = []
    if not any(g["is_target_standard_4090"] for g in gpus):
        warnings.append(
            "Standard RTX 4090/SM89 not confirmed; inventory is not an admission check."
        )
    if memory.get("MemTotal", 0) < 128_000_000_000:
        warnings.append(
            "Host RAM is below the PDF's 128-GB planning assumption; loading peaks are unmeasured."
        )
    return {
        "schema_version": 1,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "inventory_only_not_a_benchmark_or_runtime_admission",
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
        },
        "cpu": {"model": cpu_model, "logical_cpus": os.cpu_count()},
        "memory_bytes": memory,
        "project_filesystem_bytes": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
        },
        "gpu_query_status": gpu_result["status"],
        "gpus": gpus,
        "software": {
            "python": platform.python_version(),
            "packages_current_interpreter": packages,
            "tools": tools,
        },
        "warnings": warnings,
        "limitations": [
            "VRAM budget is a changing snapshot, not an allocator limit or residency guarantee.",
            "Recheck free VRAM immediately before each future model load and monitor process/device peaks.",
            "No CUDA execution, runtime compatibility, host loading peak, or generation performance was tested.",
            "No power settings changed; current package versions are observations, not selected dependencies.",
        ],
    }


def prompt_stratum(row: dict) -> str | None:
    dimensions = row["dimension"]
    if "human_action" in dimensions:
        return "people"
    if "subject_consistency" in dimensions and not row["prompt_en"].startswith(
        "a person "
    ):
        return "motion_objects"
    if "background_consistency" in dimensions:
        return "scenes"
    if "overall_consistency" in dimensions and len(row["prompt_en"].split()) >= 12:
        return "complex"
    return None


def build_prompt_manifest(metadata_bytes: bytes, source: dict) -> dict:
    if sha256(metadata_bytes) != source["sha256"]:
        raise ValueError(
            "Official metadata hash does not match the pinned source audit"
        )
    rows = json.loads(metadata_bytes)
    if not isinstance(rows, list):
        raise ValueError("Official metadata must be a list")
    buckets = {name: [] for name in STRATA}
    by_prompt = {}
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("prompt_en"), str)
            or not row["prompt_en"].strip()
        ):
            raise ValueError(f"Invalid official prompt at row {index}")
        dimensions = row.get("dimension")
        if not isinstance(dimensions, list) or not all(
            isinstance(dim, str) for dim in dimensions
        ):
            raise ValueError(f"Invalid official dimensions at row {index}")
        by_prompt.setdefault(row["prompt_en"], []).append((index, row))
    for entries in by_prompt.values():
        candidates = [
            (STRATA.index(stratum), index, row)
            for index, row in entries
            if (stratum := prompt_stratum(row)) is not None
        ]
        if candidates:
            group, index, row = min(candidates, key=lambda item: item[:2])
            buckets[STRATA[group]].append((index, row, entries))
    for name, bucket in buckets.items():
        bucket.sort(
            key=lambda item: (
                sha256(f"{SELECTION_NAMESPACE}\n{item[1]['prompt_en']}".encode()),
                item[0],
            )
        )
        if len(bucket) < 16:
            raise ValueError(f"Stratum {name} requires 16 prompts; found {len(bucket)}")
    prompts = []
    for rank in range(16):
        for name in STRATA:
            index, row, entries = buckets[name][rank]
            dimensions = {
                dim for _, original in entries for dim in original["dimension"]
            }
            prompts.append(
                {
                    "id": f"vbench-{index:04d}",
                    "prompt_en": row["prompt_en"],
                    "official_metadata_indices": [i for i, _ in entries],
                    "official_metadata_rows": [
                        copy.deepcopy(original) for _, original in entries
                    ],
                    "stratum": name,
                    "cheap_filter": rank < 8,
                    "eligible_standard_development_dimensions": [
                        d for d in DEVELOPMENT_DIMENSIONS if d in dimensions
                    ],
                }
            )
    jobs = []
    for seed_index, seed in enumerate(SEEDS):
        for prompt in prompts:
            jobs.append(
                {
                    "id": f"{prompt['id']}-s{seed}",
                    "prompt_id": prompt["id"],
                    "seed": seed,
                    "sample_index": seed_index,
                    "allocation": (
                        "cheap_filter"
                        if prompt["cheap_filter"] and seed_index == 0
                        else "shortlist_additional"
                    ),
                    "relative_output_path": f"{prompt['id']}/seed-{seed}.mp4",
                }
            )
    return {
        "schema_version": 1,
        "name": SELECTION_NAMESPACE,
        "status": "inputs_frozen_no_outputs_generated",
        "score_label": "VBench development subset scores; NOT an official VBench total",
        "source": {k: source[k] for k in ("url", "sha256")},
        "selection": {
            "ordering": "ascending SHA256(namespace + newline + exact prompt), interleaved by stratum rank",
            "strata": list(STRATA),
            "per_stratum": 16,
            "cheap_filter_per_stratum": 8,
            "complex_rule": "overall_consistency-eligible official prompt with at least 12 whitespace-separated words",
        },
        "seeds": list(SEEDS),
        "custom_input_dimensions": list(DEVELOPMENT_DIMENSIONS[:-1]),
        "standard_metadata_dimensions": ["overall_consistency"],
        "prompts": prompts,
        "jobs": jobs,
        "limitations": [
            "This manifest does not run generation or scoring.",
            "Resolve hashed filenames to original prompt/video_list metadata; do not infer prompts from filenames.",
            "One/two samples per prompt are development allocations, not official full-suite sampling.",
            "Report each dimension's eligible prompt count and missing scores; never zero-fill absent results.",
            "Audio/reference fidelity is not certified by the development filter.",
        ],
    }


class PromptDataModule:
    """Resolve frozen manifest jobs and prompts without selecting model behavior."""

    def __init__(self, manifest_path: Path):
        self.path = Path(manifest_path)
        self.manifest = load_manifest(self.path)

    def jobs(self):
        return tuple(self.manifest["jobs"])

    def prompt_for(self, job: dict) -> str:
        return next(
            item["prompt_en"]
            for item in self.manifest["prompts"]
            if item["id"] == job["prompt_id"]
        )

    def first_job(self) -> dict:
        return self.jobs()[0]

    def job(self, job_id: str | None = None) -> dict:
        if job_id is None:
            return self.first_job()
        matches = [job for job in self.jobs() if job["id"] == job_id]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one frozen manifest job: {job_id}")
        return matches[0]


def load_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if not manifest.get("jobs") or not manifest.get("prompts"):
        raise ValueError("manifest has no frozen jobs")
    return manifest
