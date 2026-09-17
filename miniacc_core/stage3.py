"""Stage 3 preparation contracts and fail-closed trial helpers.

This module deliberately does not launch inference or scoring.  It freezes the
selected native SGLang control, validates one-feature trial manifests, and
provides small resource/logging helpers used by the later runner.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

QUALITY_KEYS = ("vbench-0195", "vbench-0302", "vbench-0846", "vbench-0749")
METRICS = (
    "subject_consistency", "background_consistency", "motion_smoothness",
    "dynamic_degree", "aesthetic_quality", "imaging_quality",
    "overall_consistency",
)
CATEGORIES = ("kernel_fusion", "sparse_attention", "quantization", "token_reduction", "pipeline")
SPEED_KEY = "vbench-0195"
MIN_GPU_FREE_MIB = 8192
MIN_HOST_AVAILABLE_BYTES = 16 * 1024**3

COMMON_ACCOMMODATIONS = {
    "model_variant": "fl2va",
    "task": "t2va",
    "geometry": {"width": 1344, "height": 768, "frames": 124, "fps": 24},
    "audio": {"required": True, "sample_rate_hz": 32000, "channels": 2, "generation": "joint"},
    "batch_size": 1,
    "seed": 20260909,
    "num_inference_steps": 5,
    "expected_denoiser_forwards": 4,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "attention_backend": "fa",
    "layerwise_offload_components": ["dit", "text_encoder"],
    "dit_offload_prefetch_size": 1,
    "dit_layerwise_resident_layers": 0,
    "pin_cpu_memory": False,
    "lora_merge_cache_root": "/mnt/Projects/MiniAcc/.local/sglang-data-cache",
    "lora_merge_cache_policy": "capacity-checked file-backed exact BF16 post-adapter layers; executable/JIT caches remain relocated runtime",
    "vae_layerwise_offload": False,
    "vae_cpu_offload": True,
    "enable_torch_compile": False,
    "native_conditioning": True,
    "cached_embeddings": False,
    "decode_and_save_audio": True,
}


@dataclass(frozen=True)
class ResourceSnapshot:
    """A complete observation; missing telemetry is never coerced to zero."""

    gpu_free_mib: Mapping[int, int] | None
    host_available_bytes: int | None
    gpu_query_returncode: int | None = 0
    gpu_query_stderr: str = ""
    host_query_error: str = ""
    gpu_query_parse_error: str = ""


def _finite_positive(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _regular_target(path: Path) -> None:
    """Reject symlinked output files before replacing/writing them."""
    if path.is_symlink():
        raise ValueError(f"refusing symlinked output target: {path}")
    for parent in path.absolute().parents:
        if parent == Path(parent.anchor):
            break
        if parent.is_symlink():
            raise ValueError(f"refusing symlinked output parent: {parent}")


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a complete JSON document with fsync then atomic replacement."""
    path = Path(path)
    _regular_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """Append one flushed JSON record without allowing NaN or symlink targets."""
    path = Path(path)
    _regular_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def parse_gpu_free_csv(stdout: str) -> dict[int, int]:
    """Parse every nvidia-smi row and fail closed on malformed output."""
    rows = list(csv.reader(stdout.splitlines()))
    if not rows:
        raise ValueError("nvidia-smi returned no GPU rows")
    values: dict[int, int] = {}
    for row in rows:
        if len(row) != 2:
            raise ValueError(f"malformed nvidia-smi row: {row!r}")
        try:
            index = int(row[0].strip())
            free = int(row[1].strip())
        except ValueError as error:
            raise ValueError(f"malformed nvidia-smi numeric row: {row!r}") from error
        if index < 0 or free < 0 or index in values:
            raise ValueError(f"invalid/duplicate nvidia-smi row: {row!r}")
        values[index] = free
    return values


def query_resources(
    *, gpu_index: int = 0, runner=subprocess.run, timeout_seconds: float = 4.0
) -> ResourceSnapshot:
    """Sample whole-device resources with a bound shorter than monitor join."""
    if timeout_seconds <= 0:
        raise ValueError("resource query timeout must be positive")
    command = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return ResourceSnapshot(None, None, None, str(error), "nvidia-smi telemetry unavailable", "")
    if result.returncode:
        return ResourceSnapshot(None, None, result.returncode, result.stderr or "", "nvidia-smi failed", "")
    try:
        gpu = parse_gpu_free_csv(result.stdout)
    except ValueError as error:
        return ResourceSnapshot(None, None, result.returncode, result.stderr or "", "nvidia-smi telemetry unavailable", str(error))
    try:
        available = None
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) < 2:
                    raise ValueError("malformed MemAvailable")
                available = int(fields[1]) * 1024
                break
        if available is None:
            raise ValueError("MemAvailable is unavailable")
    except (OSError, ValueError) as error:
        return ResourceSnapshot(gpu, None, 0, "", str(error))
    return ResourceSnapshot(gpu, available)


def resource_violation(snapshot: ResourceSnapshot, *, gpu_index: int = 0) -> str | None:
    """Return a reason for admission failure, including unavailable telemetry."""
    if snapshot.gpu_free_mib is None:
        return f"GPU telemetry unavailable: {snapshot.gpu_query_stderr or 'unknown error'}"
    if gpu_index not in snapshot.gpu_free_mib:
        return f"GPU {gpu_index} telemetry unavailable"
    if any(not isinstance(value, int) for value in snapshot.gpu_free_mib.values()):
        return "GPU telemetry contains a non-integer value"
    if snapshot.gpu_free_mib[gpu_index] < MIN_GPU_FREE_MIB:
        return f"GPU {gpu_index} free memory {snapshot.gpu_free_mib[gpu_index]} MiB is below {MIN_GPU_FREE_MIB} MiB reserve"
    if snapshot.host_available_bytes is None:
        return f"host MemAvailable telemetry unavailable: {snapshot.host_query_error or 'unknown error'}"
    if snapshot.host_available_bytes < MIN_HOST_AVAILABLE_BYTES:
        return f"host MemAvailable {snapshot.host_available_bytes} bytes is below {MIN_HOST_AVAILABLE_BYTES} bytes reserve"
    return None


def require_resources(snapshot: ResourceSnapshot, *, gpu_index: int = 0) -> None:
    violation = resource_violation(snapshot, gpu_index=gpu_index)
    if violation:
        raise RuntimeError(violation)


def process_is_owned(pid: int, executable: str, proc_root: Path = Path("/proc")) -> bool:
    """Verify a live process command before any later signal/cleanup action."""
    if pid <= 0:
        return False
    try:
        command = (Path(proc_root) / str(pid) / "cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
    except OSError:
        return False
    return bool(command and executable in command and " serve " in f" {command} ")


def validate_trial_records(records: Iterable[Mapping[str, Any]], *, candidate_id: str) -> None:
    """Require exactly four quality keys and one independent speed request."""
    records = list(records)
    keys = [record.get("prompt_id") for record in records]
    if sorted(key for key in keys if key in QUALITY_KEYS) != sorted(QUALITY_KEYS):
        raise ValueError(f"{candidate_id}: quality records must contain exactly {QUALITY_KEYS}")
    speed = [record for record in records if record.get("prompt_id") == SPEED_KEY and record.get("purpose") == "speed"]
    if len(speed) != 1:
        raise ValueError(f"{candidate_id}: exactly one speed record for {SPEED_KEY} is required")
    if speed[0].get("timing_boundary") != "native_request_submission_through_saved_av_completion_before_validation":
        raise ValueError(f"{candidate_id}: speed record has an invalid timing boundary")
    for record in records:
        if record.get("cached_embeddings") is True or record.get("remote_conditioning") is True:
            raise ValueError(f"{candidate_id}: hidden conditioning substitution is forbidden")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the stage3 experiment design without executing it."""
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported Stage 3 manifest schema")
    quality = manifest.get("quality_contract", {})
    if tuple(quality.get("keys", ())) != QUALITY_KEYS:
        raise ValueError("quality keys are not the frozen first four in order")
    if tuple(quality.get("metrics", ())) != METRICS:
        raise ValueError("Stage 3 must retain all seven metrics")
    speed = manifest.get("speed_contract", {})
    if speed.get("key") != SPEED_KEY or speed.get("requests_per_configuration") != 1:
        raise ValueError("Stage 3 speed contract must be one vbench-0195 request")
    if speed.get("hardware") != "local RTX4090":
        raise ValueError("speed must be local RTX4090 only")
    baseline = manifest.get("baseline", {})
    if baseline.get("attention_backend") != "fa" or baseline.get("precision") != "BF16":
        raise ValueError("baseline precision/attention cannot hide an optimization")
    if baseline.get("remote_conditioning") or baseline.get("cached_embeddings"):
        raise ValueError("baseline cannot hide remote/cached conditioning")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("no Stage 3 candidates")
    seen: set[str] = set()
    categories: set[str] = set()
    for candidate in candidates:
        cid = candidate.get("id")
        if not isinstance(cid, str) or cid in seen:
            raise ValueError("candidate IDs must be unique strings")
        seen.add(cid)
        category = candidate.get("category")
        if category not in CATEGORIES:
            raise ValueError(f"unknown candidate category {category!r}")
        categories.add(category)
        deltas = candidate.get("changed_runtime_knobs")
        if not isinstance(deltas, list) or len(deltas) != 1:
            raise ValueError(f"{cid}: each trial must change exactly one runtime knob")
        if candidate.get("stacked_with") not in ([], None):
            raise ValueError(f"{cid}: stacked trials are forbidden")
        if candidate.get("remote_conditioning") or candidate.get("cached_embeddings"):
            raise ValueError(f"{cid}: hidden conditioning substitution is forbidden")
        audit = candidate.get("source_audit", {})
        for field in ("repository", "commit", "license", "compatibility", "dispatch"):
            if not audit.get(field):
                raise ValueError(f"{cid}: source audit missing {field}")
    if categories != set(CATEGORIES):
        raise ValueError(f"candidate categories must cover {CATEGORIES}; got {sorted(categories)}")


def baseline_manifest() -> dict[str, Any]:
    """Return the owner-selected baseline export, without copying model bytes."""
    return {
        "schema_version": 1,
        "kind": "stage2-baseline-export",
        "candidate_id": "lightx2v-4",
        "selection": "owner_manual_2026-09-14",
        "runtime": {"name": "SGLang", "version": "0.5.19", "model_variant": "fl2va", "task": "t2va", "attention_backend": "fa", "precision": "BF16"},
        "model": {"id": "MiniMaxAI/MiniMax-H3", "revision": "42ed227ee7df40d41602854ae760620d6eb651fe", "location": "remote_original_host_only", "path": "/home/di49map/MiniAcc/.local/sglang-hf-cache/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe", "config_sha256": {"FL2VA/transformer/config.json": "f619093a231fcfbcc3d035bec26c50ad864e7331a500d5c519f5045dc1e50458", "FL2VA/model_index.json": "d1113e0f123c69f79cd0de35ca1771606ebc3ec924270d257b771f96f584aa6b"}}, 
        "adapter": {"path": "/home/di49map/MiniAcc/.local/sglang-adapters/minimax_h3_fl2v_turbo_4step_v0.1.safetensors", "sha256": "5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29", "scale": 1.0, "alpha": 8, "merge_mode": "auto", "application": "native runtime"},
        "deployment": dict(COMMON_ACCOMMODATIONS),
        "provenance": {
            "stage2_run": "artifacts/evaluation/stage2-sglang-lightx2v-4-after-ram-release-20260914",
            "scored_ledger": "artifacts/evaluation/stage2-sglang-lightx2v-4-after-ram-release-20260914-scored/scored-ledger.json",
            "timing_observation": "artifacts/evaluation/stage2-sglang-timing-observed.json",
            "independent_review": "artifacts/review/sglang-final-caller-parent-check.json",
            "frozen_eval": "stage1/eval.yaml",
            "source_runtime": "remote /home/di49map/MiniAcc/.local/sglang-wheel-audit and .local/sglang-venv",
        },
        "quality": {"status": "retained_stage2_evidence", "clips": list(QUALITY_KEYS), "seven_metric_means_descriptive": {"subject_consistency": 94.9114493683307, "background_consistency": 95.35249089313581, "motion_smoothness": 97.92846879696039, "dynamic_degree": 37.5, "aesthetic_quality": 65.67736119031906, "imaging_quality": 71.98477880416378, "overall_consistency": 76.97810715698934}, "overall_eligible": 4, "custom_eligible": 16, "official_total_claim": False},
        "local_4090": {"status": "pending_exact_assets_and_fit_gate", "speed_seconds": None, "speed_ratio": None, "a100_timing_is_not_local": True},
        "remote_provenance_hardware": {"host": "wolpy08", "gpu": "A100 80GB SM80", "role": "retained_stage2_provenance_only"},
    }


def stage3_manifest() -> dict[str, Any]:
    """Return the non-combinatorial candidate design and dependency ledger."""
    source = lambda repo, commit, license_name, compatibility, dispatch: {"repository": repo, "commit": commit, "license": license_name, "compatibility": compatibility, "dispatch": dispatch}
    candidates = [
        {"id": "compiler_fusion", "category": "kernel_fusion", "feature_delta": "Enable existing native torch.compile graph capture/fusion only", "changed_runtime_knobs": ["enable_torch_compile"], "stacked_with": [], "status": "pending_readiness", "source_audit": source("sgl-project/sglang", "7465e42b7a1238761742f81a500046c1df6decc1", "Apache-2.0", "native H3 graph seam; compatibility with layerwise offload and LightX2V4 remains to test on SM89", "existing SGLang H3 projections/norm/RoPE/residual/MLP capture; no new kernel")},
        {"id": "cube_sparse", "category": "sparse_attention", "feature_delta": "Retain Cube sparse lead as an explicit incompatibility record", "changed_runtime_knobs": ["attention_backend"], "stacked_with": [], "status": "incompatible_pinned_runtime", "source_audit": source("sgl-project/sglang", "7465e42b7a1238761742f81a500046c1df6decc1", "Apache-2.0", "pinned runtime exposes no cube_sparse backend; available SubBlock sparse resolver rejects SM89; do not enable without an exact upstream port", "not dispatched: cube_sparse symbol/backend is absent from pinned runtime")},
        {"id": "sol_sparse_prefix3", "category": "sparse_attention", "feature_delta": "Use native Sol hybrid attention with dense_steps=3", "changed_runtime_knobs": ["attention_backend"], "stacked_with": [], "status": "pending_isolated_dependency", "source_audit": source("sgl-project/sglang + comfy-kitchen", "7465e42b7a1238761742f81a500046c1df6decc1 / 62c5bb4a5f2f818d4be7e2a51f83aaf1286a1243", "Apache-2.0", "dependency/resolver and exact packed H3 SM89 dispatch are not proven in the pinned runtime; enum presence is insufficient; missing dependency is not a hardware verdict", "sol_attn with explicit dense_steps=3; prove native packed dispatch")},
        {"id": "kitchen_int8", "category": "quantization", "feature_delta": "Enable native online Kitchen ConvRot INT8 linear modules", "changed_runtime_knobs": ["quantization"], "stacked_with": [], "status": "pending_isolated_dependency", "source_audit": source("sgl-project/sglang + Comfy-Org/comfy-kitchen", "7465e42b7a1238761742f81a500046c1df6decc1 / 62c5bb4a5f2f818d4be7e2a51f83aaf1286a1243", "Apache-2.0", "SM89 fused path; exact H3 QKV reorder and dynamic LightX2V LoRA ordering must be proven; no merge into quantized weights", "kitchen_int8 fused ConvRot linear after native H3 grouped-QKV reorder")},
        {"id": "video_ffn_merge_unmerge", "category": "token_reduction", "feature_delta": "Bounded MIT merge/unmerge around video-only FFN rows", "changed_runtime_knobs": ["video_ffn_token_merge_ratio"], "stacked_with": [], "status": "pending_compatibility_probe", "source_audit": source("facebookresearch/ToMe", "27a14a372beecaa85c101fb588631643f379b8ce", "MIT", "not a drop-in H3 implementation; only assess if rows restore before native residual and all text/audio/condition/attention/RoPE rows remain untouched", "video-only merge -> FFN -> unmerge seam; zero-ratio identity and mixed-modality isolation required")},
        {"id": "cache_dit", "category": "pipeline", "feature_delta": "Enable native SGLang Cache-DiT H3 adapter with explicit four-step cache policy", "changed_runtime_knobs": ["cache_dit_policy"], "stacked_with": [], "status": "pending_readiness", "source_audit": source("sgl-project/sglang + vipshop/cache-dit", "7465e42b7a1238761742f81a500046c1df6decc1 / 3db8d1e70fe4a898c85efa5fe0576d85c8396db4", "Apache-2.0", "native H3 adapter exists but cache is approximate; warmup=4 skips nothing at four forwards, so any lower warmup requires quality screening", "MiniMaxH3DiTModel BlockAdapter Pattern_3 with computed-block/skip logs")},
        {"id": "adaln_sidecar_post_adapter", "category": "pipeline", "feature_delta": "Use exact AdaLN sidecar generated from the post-adapter four-step plan", "changed_runtime_knobs": ["adaln_cache_sidecar"], "stacked_with": [], "status": "pending_post_adapter_proof", "source_audit": source("sgl-project/sglang", "7465e42b7a1238761742f81a500046c1df6decc1", "Apache-2.0", "exact only after adapter/timestep/flow-shift identity; base sidecar is not admissible for LightX2V4", "MiniMaxH3AdalnCache post-adapter outputs, byte/shape comparison before generation")},
    ]
    value = {
        "schema_version": 1,
        "kind": "stage3-individual-training-free-manifest",
        "baseline_ref": "stage2/baseline/manifest.json",
        "baseline": {"attention_backend": "fa", "precision": "BF16", "remote_conditioning": False, "cached_embeddings": False},
        "no_stacking": True,
        "common_deployment_accommodations": dict(COMMON_ACCOMMODATIONS),
        "quality_contract": {"keys": list(QUALITY_KEYS), "metrics": list(METRICS), "four_native_denoiser_forwards": True, "native_av_required": True, "overall_consistency_eligibility": "genuine eligible prompts only"},
        "speed_contract": {"hardware": "local RTX4090", "key": SPEED_KEY, "requests_per_configuration": 1, "timing_boundary": "native request submission through saved AV and completion before ffprobe/hash/scoring", "setup_compile_warmup_separate": True, "seconds_only": True, "ratio": "baseline_seconds / candidate_seconds", "remote_speed_claims": False},
        "resource_contract": {"min_gpu_free_mib": MIN_GPU_FREE_MIB, "min_host_mem_available_bytes": MIN_HOST_AVAILABLE_BYTES, "whole_device_sampling": True, "telemetry_unavailable": "fail_closed", "uncontended": True},
        "source_audit": {"sglang": {"repository": "sgl-project/sglang", "commit": "7465e42b7a1238761742f81a500046c1df6decc1", "license": "Apache-2.0"}, "lightx2v": {"repository": "ModelTC/LightX2V", "commit": "8335bb488ff747d079ad643c833ea5b10d2c5fd6", "license": "Apache-2.0"}, "comfy_kitchen": {"repository": "Comfy-Org/comfy-kitchen", "commit": "62c5bb4a5f2f818d4be7e2a51f83aaf1286a1243", "license": "Apache-2.0"}, "sage_attention": {"repository": "THU-ML/SageAttention", "commit": "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5", "license": "Apache-2.0"}, "cache_dit": {"repository": "vipshop/cache-dit", "commit": "3db8d1e70fe4a898c85efa5fe0576d85c8396db4", "license": "Apache-2.0"}, "tome": {"repository": "facebookresearch/ToMe", "commit": "27a14a372beecaa85c101fb588631643f379b8ce", "license": "MIT"}},
        "dependency_graph": {"baseline": [], "compiler_fusion": ["baseline"], "cube_sparse": ["baseline", "sglang_cube_sparse"], "sol_sparse_prefix3": ["baseline", "sglang_sol", "comfy_kitchen_sol"], "kitchen_int8": ["baseline", "sglang_kitchen_int8", "comfy_kitchen_int8"], "video_ffn_merge_unmerge": ["baseline", "tomesd_merge_primitive_only"], "cache_dit": ["baseline", "sglang_cache_dit"], "adaln_sidecar_post_adapter": ["baseline", "sglang_adaln_cache"]},
        "candidates": candidates,
        "pending": "No model/scorer campaigns run during preparation; all statuses remain pending until exact local assets/runtime admission and parent review.",
    }
    validate_manifest(value)
    return value
