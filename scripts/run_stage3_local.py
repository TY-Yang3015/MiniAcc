#!/usr/bin/env python3
"""Guarded local SGLang Stage 3 runner and load-only admission probe.

This runner is deliberately local: it consumes the frozen manifest, builds one
explicit native argv/env configuration, and never falls back to a remote
launcher or another backend. Generation/scoring remain separate later steps.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miniacc_core.sglang_timing import execute_caller_request, validate_caller_timing
from miniacc_core.stage3 import (  # noqa: E402
    append_jsonl,
    atomic_write_json,
    query_resources,
    resource_violation,
    stage3_manifest,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "stage3/manifest.json"
BASELINE = ROOT / "stage2/baseline/manifest.json"
POINTER = ROOT / "stage3/runtime-pointer.json"
SNAPSHOT = ROOT / ".local/sglang-hf-cache/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe"
ADAPTER = ROOT / ".local/sglang-adapters/minimax_h3_fl2v_turbo_4step_v0.1.safetensors"
DATA_CACHE = ROOT / ".local/sglang-data-cache"
MERGE_CACHE_EXPECTED_BYTES = 71 * 1024**3
MERGE_CACHE_HEADROOM = 1.15
SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_path() -> Path:
    data = json.loads(POINTER.read_text(encoding="utf-8"))
    path = Path(data["runtime_path"])
    base = Path("/home/arezy/.cache/miniacc/stage3-sglang-runtime")
    if not path.is_absolute() or not path.is_relative_to(base) or path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"invalid selected runtime pointer: {path}")
    return path


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(value)
    return value


def _adapter_evidence() -> dict[str, Any]:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    expected = baseline["adapter"]["sha256"]
    result: dict[str, Any] = {"path": str(ADAPTER), "expected_sha256": expected, "actual_sha256": None, "regular_non_symlink": False, "match": False}
    if not ADAPTER.is_file() or ADAPTER.is_symlink():
        return result
    result["regular_non_symlink"] = True
    result["actual_sha256"] = _sha256(ADAPTER)
    result["match"] = result["actual_sha256"] == expected
    if not result["match"]:
        raise RuntimeError(f"adapter digest mismatch: {result}")
    return result


def _adaln_sidecar_evidence(site: Path, adapter: dict[str, Any]) -> dict[str, Any]:
    """Admit only the locally prepared and independently recomputed sidecar."""
    root = ROOT / "stage3/adaln-sidecar-light4-20260915"
    ready_path = root / "ready.json"
    verification_path = ROOT / "stage3/adaln-sidecar-light4-20260915-native-check/verification.json"
    sidecar = root / "cache.safetensors"
    for path in (ready_path, verification_path, sidecar):
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise RuntimeError(f"missing or unsafe AdaLN evidence: {path}")
    if _sha256(ready_path) != "3012525eae02585662545c6091999cb88d43c1b18ac3bb684f29f95d297d02c6" or _sha256(verification_path) != "f0b1d12eb2cc02644e471bb224fc7452b854bc41b90344d915beb063ea334c93":
        raise RuntimeError("AdaLN preparation/verification receipt changed")
    ready = json.loads(ready_path.read_text())
    verification = json.loads(verification_path.read_text())
    expected = "8f794c6049b8fdfbb85793be29fe79e15ce02a3d00bbd729c2137bdadcdc34d0"
    provenance = ready["provenance"]
    checks = [
        ready.get("status") == "prepared_not_benchmarked",
        ready.get("native_storage_roundtrip") is True,
        ready.get("native_lookup_checks") == 4,
        ready.get("sidecar") == str(sidecar),
        ready.get("sha256") == verification.get("sidecar_sha256") == _sha256(sidecar) == expected,
        verification.get("status") == "native_projection_equivalence_passed",
        verification.get("exact_equal") is True,
        verification.get("projection_plan_comparisons") == 204,
        verification.get("normalized_adapter_keys") == 624,
        verification.get("cache_dependency_targets") == [],
        provenance["adapter"]["sha256"] == adapter["actual_sha256"],
        provenance["model_revision"] == SNAPSHOT.name,
        provenance["plan_gemm_rows"] == [1, 2, 2, 2],
        provenance["tp_size"] == 1,
        provenance["matmul_allow_tf32"] is False,
        verification.get("matmul_allow_tf32") is False,
    ]
    runtime = site / "sglang/multimodal_gen/runtime"
    sources = dict(provenance["native_source_sha256"])
    sources["layers/linear.py"] = verification["native_linear_sha256"]
    checks.extend(_sha256(runtime / name) == sha for name, sha in sources.items())
    plans = ROOT / "artifacts/review/stage3-adaln-plan-parent-20260915/native-plans.json"
    checks.append(_sha256(plans) == provenance["plan_record_sha256"])
    if not all(checks) or (root / "failure.json").exists():
        raise RuntimeError("AdaLN sidecar identity, native projection, or plan gate failed")
    return {"path": str(sidecar), "sha256": expected, "ready": str(ready_path),
            "verification": str(verification_path), "projection_plan_comparisons": 204,
            "scope": "local RTX4090 native projections; not full-model or cross-hardware equivalence"}


def _prompt_records(purpose: str) -> list[dict[str, Any]]:
    data = json.loads((ROOT / "stage1/eval.yaml").read_text(encoding="utf-8"))
    prompts = data.get("prompts", [])
    frozen = ["vbench-0195", "vbench-0302", "vbench-0846", "vbench-0749"]
    by_id = {row["id"]: row for row in prompts}
    if any(key not in by_id for key in frozen):
        raise RuntimeError("frozen prompt key missing from stage1/eval.yaml")
    keys = [frozen[0]] if purpose in {"speed", "boundary-a"} else frozen
    records = [
        {
            "prompt_id": key,
            "prompt_en": by_id[key]["prompt_en"],
            "stratum": by_id[key].get("stratum"),
            "purpose": purpose,
        }
        for key in keys
    ]
    if purpose == "boundary-a":
        records[0]["prompt_id"] = "warmup-excluded"
    return records


def build_local_config(manifest: dict[str, Any], candidate_id: str, purpose: str = "loader-fit") -> dict[str, Any]:
    """Build explicit local argv/env; unsupported candidates fail closed."""
    validate_manifest(manifest)
    baseline = candidate_id == "baseline"
    candidate = None if baseline else next((item for item in manifest["candidates"] if item["id"] == candidate_id), None)
    if not baseline and candidate is None:
        raise ValueError(f"unknown candidate: {candidate_id}")
    if candidate and candidate["status"] == "incompatible_pinned_runtime":
        raise RuntimeError(f"candidate {candidate_id} is incompatible with the pinned runtime")
    # Opt-ins require source-bound admission, not mere backend enum presence.
    supported = {"compiler_fusion", "adaln_sidecar_post_adapter", "kitchen_int8",
                 "sol_sparse_prefix3", "video_ffn_merge_unmerge", "cache_dit"}
    if candidate and candidate_id not in supported:
        raise RuntimeError(f"candidate {candidate_id} has no proven local dispatch integration")
    runtime = _runtime_path()
    cli = runtime / "bin/sglang-local"
    site = runtime / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    if not site.is_dir() or site.is_symlink():
        raise RuntimeError(f"selected runtime site-packages is missing or unsafe: {site}")
    if cli.is_symlink() or not cli.is_file():
        raise RuntimeError(f"selected local launcher is missing or unsafe: {cli}")
    adapter = _adapter_evidence()
    common = manifest["common_deployment_accommodations"]
    args = [str(cli), "serve", "--model-path", str(SNAPSHOT), "--model-variant", common["model_variant"], "--num-gpus", "1", "--performance-mode", "memory", "--attention-backend", common["attention_backend"], "--layerwise-offload-components", *common["layerwise_offload_components"], "--pin-cpu-memory", "false", "--vae-cpu-offload", "true", "--dit-offload-prefetch-size", str(common["dit_offload_prefetch_size"]), "--dit-layerwise-resident-layers", str(common["dit_layerwise_resident_layers"]), "--enable-torch-compile", "false", "--warmup-mode", "off", "--port", "30010", "--lora-path", str(ADAPTER.parent), "--lora-weight-name", ADAPTER.name, "--lora-nickname", candidate_id, "--lora-scale", "1.0", "--lora-alpha", str(json.loads(BASELINE.read_text(encoding="utf-8"))["adapter"]["alpha"]), "--lora-merge-mode", "auto"]
    cache = runtime / ".cache"
    env = {"MINIACC_STAGE3_RUNNER": "1", "SGLANG_DIFFUSION_STAGE_LOGGING": "1", "SGLANG_STAGE3_EXPECTED_BACKEND": common["attention_backend"], "SGLANG_STAGE3_EXPECTED_FORWARDS": str(common["expected_denoiser_forwards"]), "SGLANG_STAGE3_NO_FALLBACK": "1", "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(site), "VIRTUAL_ENV": str(runtime), "PATH": f"{runtime / 'bin'}:{SYSTEM_PATH}", "XDG_CACHE_HOME": str(cache), "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"), "TRITON_CACHE_DIR": str(cache / "triton"), "CUDA_CACHE_PATH": str(cache / "cuda"), "TORCHINDUCTOR_CACHE_DIR": str(cache / "sgl_diffusion/torch_compile_cache/inductor"), "SGLANG_DIFFUSION_CACHE_ROOT": str(DATA_CACHE)}
    args.extend(["--enable-breakable-cuda-graph", "false"])
    env.update({"SGLANG_CACHE_DIT_ENABLED": "false", "MINIACC_STAGE3_SOL": "0",
                "MINIACC_STAGE3_VIDEO_FFN_MERGE_RATIO": "0"})
    delta = "baseline"
    source_evidence = None
    sidecar_evidence = None
    expected_backend = common["attention_backend"]
    workload = {
        "task": common["task"],
        "width": common["geometry"]["width"],
        "height": common["geometry"]["height"],
        "fps": common["geometry"]["fps"],
        "frame_count": common["geometry"]["frames"],
        "num_inference_steps": common["num_inference_steps"],
        "flow_shift": common["flow_shift"],
        "audio_flow_shift": common["audio_flow_shift"],
        "seed": common["seed"],
        "batch_size": common["batch_size"],
        "audio": common["audio"],
        "native_conditioning": common["native_conditioning"],
    }
    if candidate_id == "compiler_fusion":
        args[args.index("false", args.index("--enable-torch-compile"))] = "true"
        delta = "enable_torch_compile=true"
    if candidate_id == "adaln_sidecar_post_adapter":
        sidecar_evidence = _adaln_sidecar_evidence(site, adapter)
        args.extend(["--minimax-h3-adaln-cache-path", sidecar_evidence["path"]])
        delta = "post-adapter-bound native AdaLN sidecar only; compile=false, online-cache=false"
    if candidate_id == "kitchen_int8":
        source_evidence = _kitchen_source_evidence(site)
        args.extend(["--quantization", "kitchen_int8"])
        env.update({"SGLANG_KITCHEN_INT8_MAX_ROWS": "8192",
                    "SGLANG_KITCHEN_INT8_MIN_SPLIT_N": "8192"})
        delta = "post-native-adapter Kitchen INT8 ConvRot256; native CUDA only; no other feature"
    if candidate_id in {"sol_sparse_prefix3", "video_ffn_merge_unmerge", "cache_dit"}:
        source_evidence = _custom_feature_source_evidence(site, candidate_id)
        if candidate_id == "sol_sparse_prefix3":
            env["MINIACC_STAGE3_SOL"] = "1"
            delta = "BF16 Sol block64 beta1.2815515655446004; main forwards0-2 dense,3 sparse; refiners dense"
        elif candidate_id == "video_ffn_merge_unmerge":
            env["MINIACC_STAGE3_VIDEO_FFN_MERGE_RATIO"] = "0.25"
            delta = "25% target-video packed-adjacent merge/MLP/unmerge only; bounded workspace"
        else:
            env["SGLANG_CACHE_DIT_ENABLED"] = "true"
            delta = "Cache-DiT1.3 Pattern3 static[1,0,1,0],warmup1,four forwards,no TaylorSeer; CPU retained buffers"
    if purpose not in ("loader-fit", "serve", "speed", "quality", "boundary-a"):
        raise ValueError(f"invalid runner purpose: {purpose}")
    assessment_split = _assessment_split_provenance(purpose)
    return {"schema_version": 1, "candidate_id": candidate_id, "purpose": purpose, "argv": args, "env": env, "working_directory": str(ROOT), "adapter": adapter, "prompt_records": [] if purpose in ("loader-fit", "serve") else _prompt_records(purpose), "feature_delta": delta, "feature_sources": source_evidence, "adaln_sidecar": sidecar_evidence, "workload": workload, "assessment_split": assessment_split, "dispatch_contract": {"expected_backend": expected_backend, "fallback_forbidden": True, "expected_denoiser_forwards": common["expected_denoiser_forwards"], "backend_observation_required": True, "perf_report_required": True}, "resource_contract": manifest["resource_contract"], "model_path_convention": "snapshot root passed to native CLI; --model-variant fl2va selects snapshot/FL2VA", "cache_contract": {"data_root": str(DATA_CACHE), "required_free_bytes": int(MERGE_CACHE_EXPECTED_BYTES * MERGE_CACHE_HEADROOM), "identity": "exact base snapshot revision/inventory plus adapter SHA-256, path, scale, alpha, ordered merge, dtype/layout"}}


def _kitchen_source_evidence(site: Path) -> list[dict[str, Any]]:
    """Bind the narrowly admitted post-adapter lifecycle to installed bytes."""
    path = ROOT / "stage3/redo-20260915/quantization-parent-gate/current-sources.json"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("Kitchen has no reviewed source receipt")
    entries = json.loads(path.read_text(encoding="utf-8"))
    prefix = "sglang/multimodal_gen/runtime/"
    expected = {prefix + item for item in (
        "pipelines_core/__init__.py", "managers/gpu_worker.py",
        "server_args/server_args.py", "pipelines_core/lora/kitchen_int8_lifecycle.py",
        "loader/transformer_load_utils.py", "loader/component_loaders/transformer_loader.py",
        "loader/fsdp_load.py",
    )}
    if len(entries) != len(expected) or {e["path"] for e in entries} != expected:
        raise RuntimeError("Kitchen source receipt is incomplete")
    dependencies_path = path.with_name("dependencies.json")
    package_path = ROOT / "stage3/redo-20260915/kitchen-package-reuse.json"
    for receipt in (dependencies_path, package_path):
        if not receipt.is_file() or receipt.is_symlink():
            raise RuntimeError("Kitchen dependency receipt is missing or unsafe")
    dependencies = json.loads(dependencies_path.read_text())["files"]
    required_dependencies = {prefix + name for name in (
        "layers/quantization/kitchen_int8.py", "layers/linear.py",
        "layers/lora/linear.py", "pipelines_core/lora/pipeline.py",
        "pipelines_core/lora/lora_merge_cache.py", "models/dits/minimax_h3.py",
    )}
    if len(dependencies) != len(required_dependencies) or {e["path"] for e in dependencies} != required_dependencies:
        raise RuntimeError("Kitchen behavior dependency receipt is incomplete")
    package = json.loads(package_path.read_text())
    package_entries = package["files"]
    required_package = {
        "comfy_kitchen/__init__.py", "comfy_kitchen/registry.py",
        "comfy_kitchen/backends/cuda/_C.abi3.so",
        "comfy_kitchen-0.2.33.dist-info/METADATA",
    }
    package_names = {e["path"] for e in package_entries}
    if package.get("version") != "0.2.33" or len(package_entries) != len(package_names) or not required_package <= package_names:
        raise RuntimeError("Kitchen package receipt is incomplete")
    bound = entries + dependencies + package_entries
    for entry in bound:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Kitchen source receipt contains an unsafe path")
        source = site / relative
        if not source.is_file() or source.is_symlink() or _sha256(source) != entry["sha256"]:
            raise RuntimeError(f"Kitchen installed source mismatch: {entry['path']}")
    return bound


def _custom_feature_source_evidence(site: Path, candidate_id: str) -> list[dict[str, Any]]:
    root = ROOT / "stage3/redo-20260915/composed-features"
    entries = json.loads((root / "current-sources.json").read_text())
    prefix = "sglang/multimodal_gen/runtime/"
    expected = {prefix + name for name in (
        "models/dits/minimax_h3.py", "cache/cache_dit_integration.py",
        "utils/stage3_token_cache.py", "layers/attention/backends/triton_h3_block_sparse.py",
        "pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py",
        "pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py",
    )}
    if len(entries) != len(expected) or {e["path"] for e in entries} != expected:
        raise RuntimeError("custom feature source receipt is incomplete")
    if candidate_id == "sol_sparse_prefix3":
        admission = json.loads((root / "sol-operator-admission.json").read_text())
        kernel = next(e for e in entries if e["path"].endswith("/triton_h3_block_sparse.py"))
        if admission["architecture"] != "SM89" or admission["kernel_sha256"] != kernel["sha256"]:
            raise RuntimeError("Sol lacks current SM89 operator admission")
        for gate in admission["gates"]:
            path = ROOT / gate["path"]
            result = json.loads(path.read_text())
            if _sha256(path) != gate["sha256"] or result.get("status") != "passed" or not result.get("guard_finalized"):
                raise RuntimeError("Sol operator gate changed or was not finalized")
    if candidate_id == "cache_dit":
        package = json.loads((root / "cache-package.json").read_text())
        if package.get("version") != "1.3.0" or not package.get("files"):
            raise RuntimeError("Cache-DiT package is not admitted")
        entries += package["files"]
    for entry in entries:
        relative = Path(entry["path"])
        source = site / relative
        if relative.is_absolute() or ".." in relative.parts or not source.is_file() or source.is_symlink() or _sha256(source) != entry["sha256"]:
            raise RuntimeError(f"custom feature source mismatch: {relative}")
    return entries


def _assessment_split_provenance(purpose: str) -> dict[str, Any] | None:
    """Attach the current speed/quality split manifest provenance.

    Only the local speed purpose claims split provenance: quality is measured
    by the remote cluster under the same amendment, and local results must
    never be presented as quality evidence.
    """
    if purpose != "speed":
        return None
    split = json.loads((ROOT / "stage3/assessment-split-20260915.json").read_text(encoding="utf-8"))
    local_speed = split.get("local_speed", {})
    if (
        local_speed.get("key") != "vbench-0195"
        or local_speed.get("measured_requests_per_configuration") != 1
        or local_speed.get("full_native_av") is not True
        or local_speed.get("setup_compile_warmup_excluded") is not True
    ):
        raise RuntimeError(f"assessment-split manifest no longer matches the local speed contract: {local_speed}")
    remote = split.get("remote_quality", {})
    if remote.get("local_speed_dependency") is not False or remote.get("remote_timings_are_local_speed") is not False:
        raise RuntimeError(f"assessment-split manifest remote quality contract drifted: {remote}")
    return {
        "manifest": "stage3/assessment-split-20260915.json",
        "supersedes": split.get("supersedes"),
        "owner_instruction": split.get("owner_instruction"),
        "local_speed": local_speed,
        "remote_quality": {
            key: remote.get(key)
            for key in ("host", "keys", "local_speed_dependency", "remote_timings_are_local_speed", "bitwise_cross_hardware_equivalence_claim")
        },
        "provenance": "local speed only; the four-key quality assessment is owned by the remote cluster and is separate evidence",
    }


def _process_start_ticks(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    try:
        fields = (proc_root / str(pid) / "stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (OSError, ValueError, IndexError):
        return None


def _process_snapshot(pid: int, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    try:
        stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8").split()
        cmd = (proc_root / str(pid) / "cmdline").read_bytes().decode(errors="replace").replace("\0", " ").strip()
        return {"pid": pid, "ppid": int(stat[3]), "pgid": int(stat[4]), "start_time_ticks": int(stat[21]), "cmdline": cmd}
    except (OSError, ValueError, IndexError):
        return {"pid": pid, "missing": True}


def _owned_process_tree(root_pid: int, proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    rows = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        row = _process_snapshot(int(entry.name), proc_root)
        if not row.get("missing"):
            rows.append(row)
    by_parent: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_parent.setdefault(row["ppid"], []).append(row)
    selected: list[dict[str, Any]] = []
    pending = [root_pid]
    while pending:
        current = pending.pop()
        row = next((item for item in rows if item["pid"] == current), None)
        if row is None or row in selected:
            continue
        selected.append(row)
        pending.extend(item["pid"] for item in by_parent.get(current, []))
    return selected


def _runtime_exec_chain(config: dict[str, Any], runtime: Path) -> dict[str, str] | None:
    """Resolve the exact reviewed shell -> Python wrapper -> native CLI chain."""
    wrapper = Path(config["argv"][0])
    python_wrapper = runtime / "bin/python3"
    native_cli = runtime / "bin/sglang"
    if wrapper != runtime / "bin/sglang-local":
        return None
    try:
        outer_exec = shlex.split(wrapper.read_text(encoding="utf-8").splitlines()[-1])
        python_exec = shlex.split(python_wrapper.read_text(encoding="utf-8").splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return None
    if outer_exec != ["exec", str(python_wrapper), str(native_cli), "$@"]:
        return None
    if len(python_exec) != 3 or python_exec[0] != "exec" or python_exec[2] != "$@":
        return None
    selected_python = Path(python_exec[1])
    if not selected_python.is_absolute():
        return None
    return {
        "wrapper": str(wrapper),
        "python_wrapper": str(python_wrapper),
        "native_cli": str(native_cli),
        "selected_python": str(selected_python.resolve()),
        "shell": str(Path("/bin/sh").resolve()),
    }


_EXEC_WINDOW_READ_ATTEMPTS = 5
_EXEC_WINDOW_READ_SLEEP_SECONDS = 0.01


def _read_proc_cmdline(process_dir: Path) -> list[str]:
    """Read exact NUL argv, riding out the empty-cmdline execve window.

    Between begin_new_exec and installing the new stack, /proc/<pid>/cmdline
    reads empty while the executable link already resolves, so a single read
    can observe a live exec as an empty argv (reproduced at 127/200 immediate
    post-launch reads on 2026-09-15).  Retry briefly so the launch classifier
    sees the real command; a persistently empty cmdline still returns [] and
    fails closed downstream.
    """
    for attempt in range(_EXEC_WINDOW_READ_ATTEMPTS):
        raw = (process_dir / "cmdline").read_bytes().split(b"\0")
        if raw and raw[-1] == b"":
            raw.pop()
        if raw:
            return [item.decode(errors="replace") for item in raw]
        if attempt + 1 < _EXEC_WINDOW_READ_ATTEMPTS:
            time.sleep(_EXEC_WINDOW_READ_SLEEP_SECONDS)
    return []


def _process_launch_observation(
    pid: int,
    config: dict[str, Any],
    runtime: Path,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Read exact NUL argv, executable, PID start and PGID from one /proc view."""
    process_dir = proc_root / str(pid)
    if not process_dir.is_dir():
        return {"pid": pid, "state": "missing", "error": "process directory missing"}
    try:
        raw_stat = (process_dir / "stat").read_text(encoding="utf-8")
        command_end = raw_stat.rfind(")")
        if command_end < 0:
            raise ValueError("process stat command field is malformed")
        stat_after_command = raw_stat[command_end + 1 :].split()
        argv = _read_proc_cmdline(process_dir)
        executable = str((process_dir / "exe").resolve(strict=True))
        observation: dict[str, Any] = {
            "pid": pid,
            "start_time_ticks": int(stat_after_command[19]),
            "pgid": int(stat_after_command[2]),
            "argv": argv,
            "executable": executable,
            "state": "unrecognized",
        }
    except (OSError, ValueError, IndexError) as error:
        return {
            "pid": pid,
            "state": "read_error",
            "error": f"{type(error).__name__}: {error}",
        }

    chain = _runtime_exec_chain(config, runtime)
    if chain is None:
        observation["classification_error"] = "reviewed runtime exec chain is invalid"
        return observation
    request_tail = config["argv"][1:]
    if (
        executable == chain["shell"]
        and argv == ["/bin/sh", chain["wrapper"], *request_tail]
    ):
        observation["state"] = "wrapper"
    elif (
        executable == chain["shell"]
        and argv
        == ["/bin/sh", chain["python_wrapper"], chain["native_cli"], *request_tail]
    ):
        observation["state"] = "python_wrapper"
    elif (
        executable == chain["selected_python"]
        and argv
        == [chain["selected_python"], chain["native_cli"], *request_tail]
    ):
        observation["state"] = "native"
    return observation


def _owned_process(pid: int, config: dict[str, Any], proc_root: Path = Path("/proc")) -> bool:
    """Accept only an exact reviewed launch-chain identity, never a substring."""
    return _process_launch_observation(
        pid, config, _runtime_path(), proc_root
    ).get("state") in {"wrapper", "python_wrapper", "native"}


def _effective_environment(pid: int) -> dict[str, Any]:
    """Read the native process environment, not only the parent launch dict."""
    try:
        values = {}
        for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
            if b"=" in item:
                key, value = item.split(b"=", 1)
                if key.decode() in {
                    "PATH", "PYTHONPATH", "PYTHONHOME", "XDG_CACHE_HOME",
                    "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
                    "TORCHINDUCTOR_CACHE_DIR", "SGLANG_DIFFUSION_CACHE_ROOT",
                    "MINIACC_BOUNDARY_A_MODE", "MINIACC_BOUNDARY_A_ATTEMPT_ID",
                    "MINIACC_BOUNDARY_A_ARMED_MONOTONIC", "MINIACC_BOUNDARY_A_MARKER",
                    "MINIACC_BOUNDARY_A_RECEIPT",
                    "MINIACC_BOUNDARY_A_FINALIZATION_RECEIPT",
                    "MINIACC_BOUNDARY_A_EXECUTOR_SHA256",
                }:
                    values[key.decode()] = value.decode(errors="replace")
        return values
    except OSError as error:
        return {"error": f"{type(error).__name__}: {error}"}


def _validate_effective_environment(values: dict[str, Any], runtime: Path) -> None:
    if values.get("SGLANG_DIFFUSION_CACHE_ROOT") != str(DATA_CACHE):
        raise RuntimeError("native child did not retain dedicated SGLANG_DIFFUSION_CACHE_ROOT")
    for key in ("XDG_CACHE_HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR"):
        value = values.get(key, "")
        if not value.startswith(str(runtime / ".cache")):
            raise RuntimeError(f"native child cache escaped executable runtime: {key}={value!r}")
    if not values.get("PATH", "").startswith(str(runtime / "bin")):
        raise RuntimeError("native child PATH does not start with selected runtime bin")
    site = runtime / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    if values.get("PYTHONPATH") != str(site) or "PYTHONHOME" in values:
        raise RuntimeError("native child Python environment is not hermetic")


def _launch_command_state(pid: int, config: dict[str, Any]) -> str:
    """Compatibility projection of the exact process launch observation."""
    return str(
        _process_launch_observation(pid, config, _runtime_path()).get(
            "state", "read_error"
        )
    )


def _effective_environment_classification(values: dict[str, Any]) -> str:
    if not values:
        return "empty"
    if "error" in values:
        return "read_error"
    cache_root = values.get("SGLANG_DIFFUSION_CACHE_ROOT")
    if cache_root is None:
        return "missing_cache_root"
    if cache_root != str(DATA_CACHE):
        return "cache_root_mismatch"
    return "invalid_environment"


def _wait_effective_environment(
    process: subprocess.Popen[str],
    config: dict[str, Any],
    runtime: Path,
    start_ticks: int | None,
    *,
    deadline: float,
    output: Path,
    clock=time.monotonic,
    sleeper=time.sleep,
    readiness_seconds: float = 5.0,
    resource_probe=None,
    resources_path: Path | None = None,
) -> dict[str, Any]:
    """Await an exact identity-bound environment under deadlines and resource floors."""
    started = clock()
    readiness_deadline = min(deadline, started + readiness_seconds)
    expected_pgid = process.pid
    attempts: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "pid": process.pid,
        "expected_start_time_ticks": start_ticks,
        "expected_pgid": expected_pgid,
        "started_monotonic": started,
        "absolute_campaign_deadline": deadline,
        "readiness_deadline": readiness_deadline,
        "attempts": attempts,
        "status": "waiting",
    }

    def persist(status: str) -> None:
        evidence["status"] = status
        evidence["finished_monotonic"] = clock()
        atomic_write_json(output / "effective-env-readiness.json", evidence)

    def sample_resources(attempt: dict[str, Any], position: str) -> None:
        if resource_probe is None:
            return
        try:
            record = resource_probe("launch_readiness", process.pid)
        except BaseException as error:
            record = {
                "phase": "launch_readiness",
                "pid": process.pid,
                "probe_error": f"{type(error).__name__}: {error}",
            }
            if resources_path is not None:
                append_jsonl(resources_path, record)
            attempt.setdefault("resource_observations", []).append(
                {"position": position, "record": record}
            )
            persist("resource_probe_error")
            raise
        if resources_path is not None:
            append_jsonl(resources_path, record)
        attempt.setdefault("resource_observations", []).append(
            {"position": position, "record": record}
        )
        if record.get("violation"):
            attempt["classification"] = "resource_violation"
            persist("resource_violation")
            raise RuntimeError(str(record["violation"]))

    while True:
        now = clock()
        returncode = process.poll()
        before = _process_launch_observation(process.pid, config, runtime)
        attempt: dict[str, Any] = {
            "observed_monotonic": now,
            "returncode": returncode,
            "observed_start_time_ticks": before.get("start_time_ticks"),
            "command_state": before.get("state"),
            "pre_observation": {
                "observed_monotonic": now,
                "returncode": returncode,
                **before,
            },
        }
        attempts.append(attempt)
        if now >= deadline:
            attempt["classification"] = "campaign_deadline_exhausted"
            persist("campaign_deadline_exhausted")
            raise TimeoutError(
                "absolute campaign deadline exhausted during effective environment readiness"
            )
        if now >= readiness_deadline and attempts[:-1]:
            attempts.pop()
            persist("persistent_unreadiness")
            raise RuntimeError("effective environment readiness failed closed after bounded window")
        if returncode is not None or before.get("state") == "missing":
            attempt["classification"] = "process_disappeared"
            persist("process_disappeared")
            raise RuntimeError(
                "native child process disappeared during effective environment readiness; "
                f"rc={returncode!r}"
            )
        if before.get("start_time_ticks") != start_ticks:
            attempt["classification"] = "pid_reused"
            persist("pid_reused")
            raise RuntimeError(
                "native child PID identity changed during effective environment readiness; "
                f"expected_start={start_ticks!r} "
                f"observed_start={before.get('start_time_ticks')!r}"
            )
        if before.get("pgid") != expected_pgid:
            attempt["classification"] = "pgid_changed"
            persist("pgid_changed")
            raise RuntimeError("native child process group changed during effective environment readiness")
        if before.get("state") == "read_error":
            attempt["classification"] = "process_observation_error"
            persist("process_observation_error")
            raise RuntimeError(
                "native child identity observation failed during effective environment readiness: "
                f"{before.get('error')}"
            )

        sample_resources(attempt, "before_environment")
        after_pre_resources = clock()
        if after_pre_resources >= deadline:
            attempt["classification"] = "campaign_deadline_exhausted"
            persist("campaign_deadline_exhausted")
            raise TimeoutError(
                "absolute campaign deadline exhausted after pre-environment resource observation"
            )
        if after_pre_resources >= readiness_deadline:
            attempt["classification"] = "readiness_deadline_exhausted"
            persist("readiness_deadline_exhausted")
            raise TimeoutError(
                "effective environment readiness deadline exhausted after pre-environment resource observation"
            )
        values = _effective_environment(process.pid)
        attempt["environment"] = values
        atomic_write_json(output / "effective-env.json", values)

        after_environment = clock()
        after_returncode = process.poll()
        after = _process_launch_observation(process.pid, config, runtime)
        attempt["post_observation"] = {
            "observed_monotonic": after_environment,
            "returncode": after_returncode,
            **after,
        }
        if after_environment >= deadline:
            attempt["classification"] = "campaign_deadline_exhausted"
            persist("campaign_deadline_exhausted")
            raise TimeoutError(
                "absolute campaign deadline exhausted during effective environment readiness"
            )
        if after_environment >= readiness_deadline:
            attempt["classification"] = "readiness_deadline_exhausted"
            persist("readiness_deadline_exhausted")
            raise TimeoutError(
                "effective environment readiness deadline exhausted after environment observation"
            )
        if after_returncode is not None or after.get("state") == "missing":
            attempt["classification"] = "process_disappeared"
            persist("process_disappeared")
            raise RuntimeError(
                "native child process disappeared after environment observation; "
                f"rc={after_returncode!r}"
            )
        if after.get("start_time_ticks") != start_ticks:
            attempt["classification"] = "pid_reused"
            persist("pid_reused")
            raise RuntimeError(
                "native child PID identity changed after environment observation; "
                f"expected_start={start_ticks!r} "
                f"observed_start={after.get('start_time_ticks')!r}"
            )
        if after.get("pgid") != expected_pgid:
            attempt["classification"] = "pgid_changed"
            persist("pgid_changed")
            raise RuntimeError("native child process group changed after environment observation")
        if after.get("state") == "read_error":
            attempt["classification"] = "process_observation_error"
            persist("process_observation_error")
            raise RuntimeError(
                "native child identity observation failed after environment observation: "
                f"{after.get('error')}"
            )

        reviewed_transitions = {
            "wrapper": {"wrapper", "python_wrapper", "native"},
            "python_wrapper": {"python_wrapper", "native"},
            "native": {"native"},
        }
        before_state = str(before.get("state"))
        after_state = str(after.get("state"))
        if before_state in reviewed_transitions and after_state not in reviewed_transitions[before_state]:
            attempt["classification"] = "command_changed"
            persist("command_changed")
            raise RuntimeError("native command changed after environment observation")
        if before_state not in reviewed_transitions and after_state == "native":
            attempt["classification"] = "command_changed"
            persist("command_changed")
            raise RuntimeError("native command changed from an unrecognized launch identity")

        sample_resources(attempt, "after_environment")
        after_resources = clock()
        if after_resources >= deadline:
            attempt["classification"] = "campaign_deadline_exhausted"
            persist("campaign_deadline_exhausted")
            raise TimeoutError(
                "absolute campaign deadline exhausted during effective environment readiness"
            )
        if after_resources >= readiness_deadline:
            attempt["classification"] = "readiness_deadline_exhausted"
            persist("readiness_deadline_exhausted")
            raise TimeoutError(
                "effective environment readiness deadline exhausted after resource observation"
            )
        final_returncode = process.poll()
        final = _process_launch_observation(process.pid, config, runtime)
        final_observed = clock()
        attempt["pre_publication_observation"] = {
            "observed_monotonic": final_observed,
            "returncode": final_returncode,
            **final,
        }
        if final_observed >= deadline:
            attempt["classification"] = "campaign_deadline_exhausted"
            persist("campaign_deadline_exhausted")
            raise TimeoutError(
                "absolute campaign deadline exhausted after final identity observation"
            )
        if final_observed >= readiness_deadline:
            attempt["classification"] = "readiness_deadline_exhausted"
            persist("readiness_deadline_exhausted")
            raise TimeoutError(
                "effective environment readiness deadline exhausted after final identity observation"
            )
        if final_returncode is not None or final.get("state") == "missing":
            attempt["classification"] = "process_disappeared"
            persist("process_disappeared")
            raise RuntimeError(
                "native child process disappeared before readiness publication; "
                f"rc={final_returncode!r}"
            )
        if final.get("start_time_ticks") != start_ticks:
            attempt["classification"] = "pid_reused"
            persist("pid_reused")
            raise RuntimeError("native child PID identity changed before readiness publication")
        if final.get("pgid") != expected_pgid:
            attempt["classification"] = "pgid_changed"
            persist("pgid_changed")
            raise RuntimeError("native child process group changed before readiness publication")
        final_state = str(final.get("state"))
        if final_state == "read_error":
            attempt["classification"] = "process_observation_error"
            persist("process_observation_error")
            raise RuntimeError(
                "native child identity observation failed before readiness publication: "
                f"{final.get('error')}"
            )
        if (
            after_state in reviewed_transitions
            and final_state not in reviewed_transitions[after_state]
        ):
            attempt["classification"] = "command_changed"
            persist("command_changed")
            raise RuntimeError("native command changed before readiness publication")
        if after_state not in reviewed_transitions and final_state == "native":
            attempt["classification"] = "command_changed"
            persist("command_changed")
            raise RuntimeError("native command changed from an unrecognized launch identity")

        try:
            _validate_effective_environment(values, runtime)
        except RuntimeError as error:
            attempt["classification"] = _effective_environment_classification(values)
            attempt["validation_error"] = f"{type(error).__name__}: {error}"
        else:
            if final_state == "native":
                attempt["classification"] = "valid"
                persist("ready")
                return values
            if final_state in {"wrapper", "python_wrapper"}:
                attempt["classification"] = "wrapper_waiting_for_exec"
                attempt["validation_error"] = (
                    "RuntimeError: reviewed wrapper chain has not normalized to native exec target"
                )
            else:
                attempt["classification"] = "unrecognized_command"
                attempt["validation_error"] = (
                    "RuntimeError: child command is outside the exact reviewed launch chain"
                )
        persist("waiting")
        now = clock()
        if now >= deadline:
            persist("campaign_deadline_exhausted")
            raise TimeoutError(
                "absolute campaign deadline exhausted during effective environment readiness"
            )
        if now >= readiness_deadline:
            persist("persistent_unreadiness")
            raise RuntimeError(
                "effective environment readiness failed closed after bounded window; "
                f"last={attempt.get('classification')}: {attempt.get('validation_error')}"
            )
        sleeper(min(0.1, readiness_deadline - now, deadline - now))


def _process_memory(pid: int | None) -> dict[str, Any]:
    """Capture native loader PSS, mapped RSS and pin observations."""
    result: dict[str, Any] = {"pid": pid, "pss_kib": None, "rss_kib": None, "private_kib": None, "shared_kib": None, "vm_pin_kib": None, "mlocked_kib": None, "error": None}
    if pid is None:
        return result
    try:
        rollup = (Path(f"/proc/{pid}/smaps_rollup")).read_text(encoding="utf-8")
        values = {}
        for line in rollup.splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0].rstrip(":") in {"Pss", "Rss", "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty"}:
                values[fields[0].rstrip(":")] = int(fields[1])
        result.update({"pss_kib": values.get("Pss"), "rss_kib": values.get("Rss"), "private_kib": values.get("Private_Clean", 0) + values.get("Private_Dirty", 0), "shared_kib": values.get("Shared_Clean", 0) + values.get("Shared_Dirty", 0)})
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmPin:"):
                result["vm_pin_kib"] = int(line.split()[1])
                break
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("Mlocked:"):
                result["mlocked_kib"] = int(line.split()[1])
                break
    except (OSError, ValueError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def _validate_data_cache(config: dict[str, Any], *, create: bool = False) -> dict[str, Any]:
    """Admission-check the data-only cache without touching executable caches."""
    path = Path(config["cache_contract"]["data_root"])
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise RuntimeError(f"unsafe data cache path: {path}")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError(f"unsafe data cache parent: {parent}")
    required = int(config["cache_contract"]["required_free_bytes"])
    usage = shutil.disk_usage(parent)
    result = {"path": str(path), "parent": str(parent), "free_bytes": usage.free, "required_free_bytes": required, "capacity_pass": usage.free >= required, "created": False}
    if not result["capacity_pass"]:
        raise RuntimeError(f"data cache capacity blocked: {usage.free} < {required} bytes")
    if create:
        path.mkdir(mode=0o755, exist_ok=True)
        if path.is_symlink():
            raise RuntimeError(f"data cache became symlinked: {path}")
        result["created"] = True
    return result


def _cache_provenance(config: dict[str, Any], capacity: dict[str, Any]) -> dict[str, Any]:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    inventory = ROOT / "stage3/source-file-inventory.json"
    return {"schema_version": 1, "cache_root": str(DATA_CACHE), "capacity": capacity, "base_model_revision": baseline["model"]["revision"], "base_snapshot": str(SNAPSHOT), "base_inventory_sha256": _sha256(inventory), "adapter_path": str(ADAPTER), "adapter_sha256": config["adapter"]["actual_sha256"], "adapter_scale": 1.0, "adapter_alpha": baseline["adapter"]["alpha"], "ordered_merge": [{"path": str(ADAPTER), "strength": 1.0, "alpha": baseline["adapter"]["alpha"]}], "dtype": "native BF16 merged weights", "layout": "native safetensors tensor shape/layout; no quantized cache reuse", "runtime_cache_key": "4c13e852b953d64a (native LoraMergeCache key observed in loader log)", "runtime": {"name": "SGLang", "version": "0.5.19", "model_variant": "fl2va"}, "reuse_policy": "only exact base inventory, adapter digest, scale/alpha, ordered merge, dtype/layout may adopt cache"}


def _resource_record(phase: str, pid: int | None = None):
    snapshot = query_resources()
    tree = _owned_process_tree(pid) if pid is not None else []
    if pid is not None:
        known = {row["pid"] for row in tree}
        tree.extend(row for row in _group_snapshot(pid) if row["pid"] not in known)
    per_process = [{**row, "memory": _process_memory(row["pid"])} for row in tree]
    pss = [row["memory"].get("pss_kib") for row in per_process if row["memory"].get("pss_kib") is not None]
    return {"timestamp_epoch": time.time(), "phase": phase, "gpu_free_mib": dict(snapshot.gpu_free_mib) if snapshot.gpu_free_mib is not None else None, "host_mem_available_bytes": snapshot.host_available_bytes, "gpu_query_returncode": snapshot.gpu_query_returncode, "gpu_query_stderr": snapshot.gpu_query_stderr, "gpu_query_parse_error": snapshot.gpu_query_parse_error, "host_query_error": snapshot.host_query_error, "violation": resource_violation(snapshot), "owned_process_tree": per_process, "memory_scope": "per-PID observations; pss_sum_kib is a process-tree sum with shared-page double-count caveat; host Mlocked is host-wide", "pss_sum_kib": sum(pss) if pss else None, "process_memory": _process_memory(pid)}


def _group_snapshot(pgid: int) -> list[dict[str, Any]]:
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        row = _process_snapshot(int(entry.name))
        if not row.get("missing") and row.get("pgid") == pgid:
            rows.append(row)
    return rows


def _cleanup_owned_process(process: subprocess.Popen[str], config: dict[str, Any], output: Path, start_ticks: int | None) -> dict[str, Any]:
    """Terminate only the launch session and retain auditable reap evidence."""
    pgid = process.pid
    before = _group_snapshot(pgid)
    root_owned = _owned_process(process.pid, config) and _process_start_ticks(process.pid) == start_ticks
    group_owned = root_owned
    receipt = {"pid": process.pid, "pgid": pgid, "start_time_ticks": start_ticks, "owned_root_verified": root_owned, "owned_group_verified": group_owned, "group_before": before, "signal": None, "wait_returncode": None, "forced_kill": False}
    if before and not group_owned:
        # The abort guard may already have terminated the root. Reap/observe
        # without sending another signal: an empty argv or vanished root is
        # not authority to signal surviving members (or a reused group).
        receipt["reconciliation_only"] = True
        deadline = time.monotonic() + 10
        while True:
            receipt["wait_returncode"] = process.poll()
            receipt["group_after"] = _group_snapshot(pgid)
            if not receipt["group_after"] or time.monotonic() >= deadline:
                break
            time.sleep(0.2)
        receipt["clean"] = (
            not receipt["group_after"] and receipt["wait_returncode"] is not None
        )
        if not receipt["clean"]:
            receipt["error"] = f"refusing to stop unverified process group {pgid}"
        atomic_write_json(output / "cleanup-receipt.json", receipt)
        if not receipt["clean"]:
            raise RuntimeError(receipt["error"])
        return receipt
    if before:
        os.killpg(pgid, signal.SIGTERM)
        receipt["signal"] = "SIGTERM"
    try:
        receipt["wait_returncode"] = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if _group_snapshot(pgid):
            os.killpg(pgid, signal.SIGKILL)
            receipt["signal"] = "SIGKILL"
            receipt["forced_kill"] = True
        receipt["wait_returncode"] = process.wait(timeout=30)
    deadline = time.monotonic() + 10
    while _group_snapshot(pgid) and time.monotonic() < deadline:
        time.sleep(0.2)
    receipt["group_after"] = _group_snapshot(pgid)
    receipt["clean"] = not receipt["group_after"]
    atomic_write_json(output / "cleanup-receipt.json", receipt)
    return receipt


def _prepare_launch(
    config: dict[str, Any],
    output: Path,
    initial_phase: str = "loader_fit_before_launch",
    deadline: float | None = None,
) -> tuple[subprocess.Popen[str], Any, Path, dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=False)
    capacity = _validate_data_cache(config, create=True)
    provenance = _cache_provenance(config, capacity)
    provenance_path = DATA_CACHE / "cache-provenance.json"
    if provenance_path.is_file():
        prior = json.loads(provenance_path.read_text(encoding="utf-8"))
        for field in ("base_model_revision", "base_inventory_sha256", "adapter_sha256", "adapter_scale", "adapter_alpha", "ordered_merge", "dtype", "layout", "runtime_cache_key"):
            if prior.get(field) != provenance.get(field):
                raise RuntimeError(f"refusing incompatible existing data cache provenance field: {field}")
    atomic_write_json(provenance_path, provenance)
    atomic_write_json(output / "cache-capacity.json", capacity)
    (output / "runner-config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"):
        env.pop(key, None)
    env.update(config["env"])
    env["PYTHONNOUSERSITE"] = "1"
    resources = output / "resources.jsonl"
    first = _resource_record(initial_phase)
    append_jsonl(resources, first)
    if first.get("violation"):
        raise RuntimeError(f"loader-fit resource admission blocked: {first['violation']}")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("absolute campaign deadline exhausted before native process spawn")

    log = (output / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        config["argv"], cwd=ROOT, env=env, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True, text=True,
    )
    start_ticks = _process_start_ticks(process.pid)
    try:
        pgid = os.getpgid(process.pid)
        (output / "process.json").write_text(
            json.dumps(
                {
                    "pid": process.pid,
                    "argv": config["argv"],
                    "candidate_id": config["candidate_id"],
                    "mode": "loader-fit",
                    "start_time_ticks": start_ticks,
                    "pgid": pgid,
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        _wait_effective_environment(
            process,
            config,
            _runtime_path(),
            start_ticks,
            deadline=(deadline if deadline is not None else time.monotonic() + 5.0),
            output=output,
            resource_probe=_resource_record,
            resources_path=resources,
        )
    except BaseException as launch_error:
        cleanup_error: BaseException | None = None
        try:
            _cleanup_owned_process(process, config, output, start_ticks)
        except BaseException as error:
            cleanup_error = error
            try:
                atomic_write_json(
                    output / "cleanup-error.json",
                    {
                        "launch_error": f"{type(launch_error).__name__}: {launch_error}",
                        "cleanup_error": f"{type(error).__name__}: {error}",
                    },
                )
            except BaseException:
                pass
        finally:
            try:
                log.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if cleanup_error is not None:
            raise launch_error from cleanup_error
        raise
    return process, start_ticks, log, capacity


class MonitorTerminationError(RuntimeError):
    """The resource monitor could not be settled before finalization."""


class OwnedProcessIdentityError(RuntimeError):
    """The exact owned native process identity disappeared or changed."""


class _CampaignAbortGuard:
    """Fail closed using the launch-time PID/start/PGID identity."""

    def __init__(
        self,
        process: subprocess.Popen[str],
        config: dict[str, Any],
        start_ticks: int | None,
        pgid: int,
    ) -> None:
        self.process = process
        self.config = config
        self.start_ticks = start_ticks
        self.pgid = pgid
        self.state: dict[str, Any] = {
            "violation": None,
            "error": None,
            "abort_receipt": None,
        }
        self._lock = threading.Lock()

    def _abort_owned_group(self, reason: str) -> dict[str, Any]:
        current_ticks = _process_start_ticks(self.process.pid)
        owned = _owned_process(self.process.pid, self.config)
        root_identity_verified = bool(
            owned
            and self.start_ticks is not None
            and current_ticks == self.start_ticks
            and self.pgid == self.process.pid
        )
        group = _group_snapshot(self.pgid)
        surviving_group_verified = bool(
            self.pgid == self.process.pid
            and group
            and all(row.get("pgid") == self.pgid for row in group)
            and any(
                "sglang" in row.get("cmdline", "")
                or "sgl_diffusion" in row.get("cmdline", "")
                for row in group
            )
        )
        identity_verified = root_identity_verified or surviving_group_verified
        receipt = {
            "pid": self.process.pid,
            "pgid": self.pgid,
            "captured_start_time_ticks": self.start_ticks,
            "observed_start_time_ticks": current_ticks,
            "identity_verified": identity_verified,
            "root_identity_verified": root_identity_verified,
            "surviving_group_verified": surviving_group_verified,
            "group_before": group,
            "reason": reason,
            "signal": None,
        }
        if identity_verified and group:
            try:
                os.killpg(self.pgid, signal.SIGTERM)
                receipt["signal"] = "SIGTERM"
            except OSError as error:
                receipt["signal_error"] = f"{type(error).__name__}: {error}"
        return receipt

    def fail(self, reason: str, *, violation: bool = False) -> None:
        with self._lock:
            key = "violation" if violation else "error"
            if self.state[key] is None:
                self.state[key] = reason
            if self.state["abort_receipt"] is None:
                self.state["abort_receipt"] = self._abort_owned_group(reason)

    def check(self, where: str) -> None:
        reason = self.state["error"] or self.state["violation"]
        if reason:
            raise RuntimeError(f"campaign aborted at {where}: {reason}")
        if not _owned_process(self.process.pid, self.config):
            returncode = self.process.poll()
            reason = (
                "owned native process disappeared during request "
                f"at {where}; rc={returncode!r}"
            )
            self.fail(reason)
            raise OwnedProcessIdentityError(reason)


def _remaining(deadline: float, clock, label: str) -> float:
    remaining = float(deadline) - float(clock())
    if remaining <= 0:
        raise TimeoutError(f"{label} deadline expired")
    return remaining


def _check_abort(abort_guard: _CampaignAbortGuard | None, where: str) -> None:
    if abort_guard is not None:
        abort_guard.check(where)


def _sleep_until(
    seconds: float,
    deadline: float,
    *,
    abort_guard: _CampaignAbortGuard | None = None,
    clock=time.monotonic,
) -> None:
    _check_abort(abort_guard, "before sleep")
    duration = min(float(seconds), _remaining(deadline, clock, "campaign"))
    time.sleep(duration)
    _check_abort(abort_guard, "after sleep")
    _remaining(deadline, clock, "campaign")


class HttpTransportTimeoutError(TimeoutError):
    """A connect/response timeout while the absolute campaign still had time."""


def _raise_http_timeout(
    error: TimeoutError,
    method: str,
    url: str,
    effective_timeout: float,
    deadline: float | None,
    clock,
) -> None:
    if deadline is not None:
        try:
            _remaining(deadline, clock, f"{method} {url}")
        except TimeoutError as deadline_error:
            raise deadline_error from error
    raise HttpTransportTimeoutError(
        f"{method} {url} transport timed out with bounded timeout "
        f"{effective_timeout}: {error}"
    ) from error


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
    *,
    deadline: float | None = None,
    abort_guard: _CampaignAbortGuard | None = None,
    clock=time.monotonic,
) -> dict[str, Any]:
    _check_abort(abort_guard, f"before {method} {url}")
    effective_timeout = float(timeout)
    if deadline is not None:
        effective_timeout = min(
            effective_timeout, _remaining(deadline, clock, f"{method} {url}")
        )
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        response = urllib.request.urlopen(request, timeout=effective_timeout)
    except TimeoutError as error:
        _raise_http_timeout(
            error, method, url, effective_timeout, deadline, clock
        )
    with response:
        _check_abort(abort_guard, f"after opening {method} {url}")
        try:
            raw = response.read()
        except TimeoutError as error:
            _raise_http_timeout(
                error, method, url, effective_timeout, deadline, clock
            )
    _check_abort(abort_guard, f"after reading {method} {url}")
    if deadline is not None:
        _remaining(deadline, clock, f"{method} {url}")
    return json.loads(raw)


def _wait_health(
    base: str,
    process: subprocess.Popen[str],
    config: dict[str, Any],
    deadline: float,
    output: Path,
    *,
    abort_guard: _CampaignAbortGuard | None = None,
    clock=time.monotonic,
) -> None:
    del output
    while True:
        _check_abort(abort_guard, "health check")
        _remaining(deadline, clock, "native server HTTP readiness")
        if not _owned_process(process.pid, config):
            returncode = process.poll()
            raise RuntimeError(
                f"native server exited before HTTP readiness; rc={returncode!r}"
            )
        try:
            _http_json(
                "GET",
                base + "/health",
                timeout=10,
                deadline=deadline,
                abort_guard=abort_guard,
                clock=clock,
            )
            _check_abort(abort_guard, "health response")
            return
        except (OSError, ValueError, urllib.error.URLError):
            _sleep_until(
                1, deadline, abort_guard=abort_guard, clock=clock
            )


def _video_body(prompt: str, config: dict[str, Any]) -> dict[str, Any]:
    """Build the request solely from the validated, persisted workload contract."""
    workload = config.get("workload")
    if not isinstance(workload, dict):
        raise RuntimeError("runner config is missing its frozen native workload")
    required = ("task", "width", "height", "fps", "frame_count", "num_inference_steps", "flow_shift", "audio_flow_shift", "seed", "batch_size", "audio", "native_conditioning")
    if any(key not in workload for key in required):
        raise RuntimeError("runner config has an incomplete frozen native workload")
    if workload["batch_size"] != 1 or workload["fps"] != 24 or workload["frame_count"] != 124:
        raise RuntimeError("runner config workload is outside the accepted native geometry")
    if workload["width"] != 1344 or workload["height"] != 768 or workload["seed"] != 20260909:
        raise RuntimeError("runner config workload is outside the accepted native identity")
    # H3 resolves temporal dimensions from target.duration_seconds and rejects
    # explicit transport fps/num_frames.  5 s rounds and aligns to 124 frames.
    return {
        "model": "MiniMaxAI/MiniMax-H3",
        "prompt": prompt,
        "seconds": 5,
        "task": workload["task"],
        "conditions": [],
        "target": {
            "short_edge": workload["height"],
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
        },
        "num_outputs_per_prompt": workload["batch_size"],
        "num_inference_steps": workload["num_inference_steps"],
        "flow_shift": workload["flow_shift"],
        "audio_flow_shift": workload["audio_flow_shift"],
        "seed": workload["seed"],
        "output_mode": "decoded_files",
    }


_FROZEN_WORKLOAD = {
    "task": "t2va",
    "width": 1344,
    "height": 768,
    "fps": 24,
    "frame_count": 124,
    "num_inference_steps": 5,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "seed": 20260909,
    "batch_size": 1,
    "native_conditioning": True,
}


def _assert_frozen_workload(config: dict[str, Any]) -> None:
    workload = config.get("workload")
    if not isinstance(workload, dict):
        raise RuntimeError("frozen native workload is missing")
    mismatches = {
        key: {"expected": expected, "actual": workload.get(key)}
        for key, expected in _FROZEN_WORKLOAD.items()
        if workload.get(key) != expected
    }
    audio = workload.get("audio")
    if audio != {
        "required": True,
        "sample_rate_hz": 32000,
        "channels": 2,
        "generation": "joint",
    }:
        mismatches["audio"] = {"expected": "joint stereo 32000 Hz", "actual": audio}
    if workload.get("cached_embeddings", False) is not False:
        mismatches["cached_embeddings"] = {
            "expected": False,
            "actual": workload.get("cached_embeddings"),
        }
    if workload.get("remote_conditioning", False) is not False:
        mismatches["remote_conditioning"] = {
            "expected": False,
            "actual": workload.get("remote_conditioning"),
        }
    hidden = {
        key: value
        for key, value in workload.items()
        if key in {
            "prompt_embeddings",
            "condition_embeddings",
            "remote_conditioner",
            "transformer_weights_path",
        }
        and value not in (None, False, [], {})
    }
    if hidden:
        mismatches["hidden_substitution_fields"] = hidden
    if mismatches:
        raise RuntimeError(f"frozen native request contract mismatch: {mismatches}")


def _reject_payload_substitutions(body: dict[str, Any]) -> None:
    forbidden_names = {
        "cached_embeddings",
        "prompt_embeddings",
        "remote_conditioning",
        "remote_conditioner",
        "condition_embeddings",
        "transformer_weights_path",
    }
    forbidden = {
        key: value
        for key, value in body.items()
        if key in forbidden_names and value not in (None, False, [], {})
    }
    if forbidden or body.get("model") != "MiniMaxAI/MiniMax-H3":
        raise RuntimeError(
            "actual submitted payload contains a forbidden model/conditioning "
            f"substitution: model={body.get('model')!r}, fields={forbidden}"
        )


def _cpu_native_request_validation(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute only in the isolated selected-runtime CPU subprocess."""
    import inspect
    import msgspec
    from types import SimpleNamespace

    from sglang.multimodal_gen.configs.sample import minimax_h3 as sampling_module
    from sglang.multimodal_gen.runtime.entrypoints.openai import protocol as protocol_module
    from sglang.multimodal_gen.runtime.pipelines_core import schedule_batch as schedule_module
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import request_validation as validation_module
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import resolved_plan as plan_module
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import timestep_preparation as timestep_module

    body = payload["body"]
    workload = payload["workload"]
    _reject_payload_substitutions(body)
    request = protocol_module.VideoGenerationsRequest(**body)
    if not isinstance(request, protocol_module.VideoGenerationsRequest):
        raise RuntimeError("selected VideoGenerationsRequest was not constructed")
    # Mirror video_api's transport projection: it synthesizes 120/24 from
    # seconds=5, and the H3 lowerer must remove those transport timing fields
    # before canonical target duration resolves to 124 aligned frames.
    generic = {
        "prompt": request.prompt,
        "seed": request.seed,
        "num_frames": int(request.seconds * 24),
        "fps": 24,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "guidance_scale_2": request.guidance_scale_2,
        "true_cfg_scale": request.true_cfg_scale,
        "negative_prompt": request.negative_prompt,
        "flow_shift": request.flow_shift,
        "audio_flow_shift": (request.model_extra or {}).get("audio_flow_shift"),
        "output_mode": (request.model_extra or {}).get("output_mode"),
        "perf_dump_path": request.perf_dump_path,
    }
    lowered = sampling_module.MiniMaxH3SamplingParams.lower_video_request_kwargs(
        request, generic
    )
    sampling_params = sampling_module.MiniMaxH3SamplingParams(**lowered)
    native_req = schedule_module.Req(sampling_params=sampling_params)
    canonical = validation_module.minimax_h3_validate_canonical_request(
        task=lowered.get("task"),
        prompt=lowered.get("prompt"),
        conditions=lowered.get("conditions"),
        target=lowered.get("target"),
        flow_shift=lowered.get("flow_shift"),
        audio_flow_shift=lowered.get("audio_flow_shift"),
        seed=lowered.get("seed"),
    )
    plan = plan_module.minimax_h3_resolve_plan(canonical)
    plan_value = msgspec.to_builtins(plan)
    shape = plan_value["shape"]
    forbidden_names = {
        "cached_embeddings",
        "prompt_embeddings",
        "remote_conditioning",
        "remote_conditioner",
        "condition_embeddings",
        "transformer_weights_path",
    }
    forbidden = {
        key: value
        for key, value in body.items()
        if key in forbidden_names and value not in (None, False, [], {})
    }
    # Exercise the installed native timestep preparation. This binds the
    # submitted schedule count to both modality sigma arrays and forward slots.
    batch = SimpleNamespace(
        extra={}, num_inference_steps=lowered.get("num_inference_steps")
    )
    timestep_stage = timestep_module.MiniMaxH3TimestepPreparationStage.__new__(
        timestep_module.MiniMaxH3TimestepPreparationStage
    )
    timestep_stage.sigma_shift_scales = None
    timestep_stage._generate_sigmas_from_plan(batch, plan)
    timestep_stage._publish_native_timestep_state(batch)
    sigmas = {
        name: [float(value) for value in values]
        for name, values in batch.extra["minimax_h3_sigmas"].items()
    }
    timesteps = [float(value) for value in batch.timesteps.tolist()]
    forward_slots = len(sigmas["video"]) - 1
    branches = plan_value.get("branches", [])
    checks = {
        "request_instance": type(request) is protocol_module.VideoGenerationsRequest,
        "task": plan.task == workload["task"],
        "geometry": shape.get("width") == workload["width"]
        and shape.get("height") == workload["height"],
        "temporal": shape.get("frame_count") == workload["frame_count"]
        and shape.get("fps") == workload["fps"],
        "seed": canonical.get("seed") == workload["seed"],
        "schedule": lowered.get("num_inference_steps")
        == workload["num_inference_steps"]
        and len(sigmas["video"]) == workload["num_inference_steps"]
        and len(sigmas["audio"]) == workload["num_inference_steps"]
        and forward_slots == workload["expected_denoiser_forwards"]
        and len(timesteps) == workload["expected_denoiser_forwards"],
        "shifts": plan.flow_shift == workload["flow_shift"]
        and plan.audio_flow_shift == workload["audio_flow_shift"],
        "native_conditioning": canonical.get("conditions") == []
        and workload["native_conditioning"] is True,
        "joint_audio": workload["audio"]
        == {
            "required": True,
            "sample_rate_hz": 32000,
            "channels": 2,
            "generation": "joint",
        }
        and bool(branches),
        "no_substitution": not forbidden
        and body.get("model") == "MiniMaxAI/MiniMax-H3",
        "transport_temporal_absent": request.fps is None
        and request.num_frames is None
        and "fps" not in lowered
        and "num_frames" not in lowered,
    }
    diagnostic_marker = payload.get("diagnostic_marker")
    if diagnostic_marker is not None:
        checks["diagnostic_marker_propagation"] = (
            request.perf_dump_path == diagnostic_marker
            and lowered.get("perf_dump_path") == diagnostic_marker
        )
        checks["native_req_marker_propagation"] = (
            type(native_req) is schedule_module.Req
            and native_req.perf_dump_path == diagnostic_marker
        )
    checks["all"] = all(checks.values())
    modules = {
        "protocol": protocol_module,
        "sampling_params": sampling_module,
        "schedule_batch": schedule_module,
        "request_validation": validation_module,
        "resolved_plan": plan_module,
        "timestep_preparation": timestep_module,
    }
    result = {
        "checks": checks,
        "resolved": {
            "task": plan.task,
            "shape": {
                "width": shape.get("width"),
                "height": shape.get("height"),
                "frame_count": shape.get("frame_count"),
                "fps": shape.get("fps"),
            },
            "seed": plan.seed,
            "num_inference_steps": lowered.get("num_inference_steps"),
            "expected_denoiser_forwards": forward_slots,
            "flow_shift": plan.flow_shift,
            "audio_flow_shift": plan.audio_flow_shift,
            "native_sigmas": sigmas,
            "native_timesteps": timesteps,
            "branches": branches,
        },
        "lowered": lowered,
        "native_req": {
            "type": type(native_req).__name__,
            "perf_dump_path": native_req.perf_dump_path,
        },
        "canonical": canonical,
        "forbidden_substitutions": forbidden,
        "module_origins": {
            name: str(Path(inspect.getfile(module)).resolve())
            for name, module in modules.items()
        },
        "subprocess": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python_executable": sys.executable,
        },
    }
    if not checks["all"]:
        raise RuntimeError(f"native H3 canonical resolution mismatch: {checks}")
    return result


def _run_native_validation_payload(
    payload: dict[str, Any], *, timeout: float = 120.0
) -> dict[str, Any]:
    """Run a supplied request contract through the isolated CPU seam."""
    runtime = _runtime_path()
    site = runtime / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"):
        env.pop(key, None)
    command = [str(runtime / "bin/python"), str(Path(__file__).resolve()), "--cpu-validate-request"]
    with tempfile.TemporaryDirectory(prefix="miniacc-stage3-cpu-validation-") as cache:
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(site),
                "VIRTUAL_ENV": str(runtime),
                "PATH": f"{runtime / 'bin'}:{SYSTEM_PATH}",
                "XDG_CACHE_HOME": cache,
                "TORCH_EXTENSIONS_DIR": str(Path(cache) / "torch_extensions"),
                "TRITON_CACHE_DIR": str(Path(cache) / "triton"),
                "CUDA_CACHE_PATH": str(Path(cache) / "cuda"),
                "TORCHINDUCTOR_CACHE_DIR": str(Path(cache) / "inductor"),
                "HF_HOME": str(Path(cache) / "huggingface"),
            }
        )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated native request validation failed "
            f"rc={completed.returncode}; stderr={completed.stderr!r}; "
            f"stdout={completed.stdout!r}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"isolated native request validation emitted invalid JSON: {completed.stdout!r}"
        ) from error
    selected_site = site.resolve()
    origins = result.get("module_origins", {})
    if not origins or any(
        not Path(origin).resolve().is_relative_to(selected_site)
        for origin in origins.values()
    ):
        raise RuntimeError(f"native validation module origin escaped selected runtime: {origins}")
    result["subprocess"].update(
        {
            "returncode": completed.returncode,
            "stderr": completed.stderr,
            "runtime": str(runtime),
        }
    )
    return result


def _validate_native_request_subprocess(
    config: dict[str, Any],
    prompt: str,
    *,
    timeout: float = 120.0,
    diagnostic_marker: str | None = None,
) -> dict[str, Any]:
    """Validate the actual frozen request without importing SGLang here."""
    _assert_frozen_workload(config)
    body = _video_body(prompt, config)
    if diagnostic_marker is not None:
        body["perf_dump_path"] = diagnostic_marker
    _reject_payload_substitutions(body)
    payload = {
        "body": body,
        "workload": {
            **config["workload"],
            "expected_denoiser_forwards": config["dispatch_contract"][
                "expected_denoiser_forwards"
            ],
        },
    }
    if diagnostic_marker is not None:
        payload["diagnostic_marker"] = diagnostic_marker
    return _run_native_validation_payload(payload, timeout=timeout)


def _media_validation(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "valid": False,
    }
    if not path.is_file():
        return result
    result["sha256"] = _sha256(path)
    probe = subprocess.run(
        [
            "/usr/bin/ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    result["ffprobe_returncode"] = probe.returncode
    if probe.returncode:
        result["ffprobe_stderr"] = probe.stderr[-4000:]
        return result
    try:
        info = json.loads(probe.stdout)
    except json.JSONDecodeError as error:
        result["ffprobe_stderr"] = f"invalid ffprobe JSON: {error}; {probe.stderr[-4000:]}"
        return result
    streams = info.get("streams", [])
    audio_stream = next(
        (row for row in streams if row.get("codec_type") == "audio"), None
    )
    video_stream = next(
        (row for row in streams if row.get("codec_type") == "video"), None
    )
    result["audio_valid"] = bool(
        audio_stream
        and str(audio_stream.get("sample_rate")) == "32000"
        and int(audio_stream.get("channels") or 0) == 2
    )
    frame_rate = (video_stream or {}).get("r_frame_rate") or (
        video_stream or {}
    ).get("avg_frame_rate")
    try:
        decoded_frame_count = int((video_stream or {}).get("nb_read_frames"))
    except (TypeError, ValueError):
        decoded_frame_count = None
    result["decoded_video_frame_count"] = decoded_frame_count
    result["metadata_video_frame_count"] = (video_stream or {}).get("nb_frames")
    result["video_valid"] = bool(
        video_stream
        and video_stream.get("width") == 1344
        and video_stream.get("height") == 768
        and frame_rate == "24/1"
        and decoded_frame_count == 124
    )
    decode = subprocess.run(
        [
            "/usr/bin/ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    result["full_decode_returncode"] = decode.returncode
    result["full_decode_stderr"] = decode.stderr[-4000:]
    result["valid"] = bool(
        result["audio_valid"]
        and result["video_valid"]
        and decode.returncode == 0
    )
    return result


def _set_response_timeout(response: Any, timeout: float) -> None:
    """Tighten a urllib socket timeout where the response exposes one."""
    candidates = [response]
    for attributes in (("fp",), ("fp", "raw"), ("fp", "raw", "_sock")):
        value = response
        try:
            for attribute in attributes:
                value = getattr(value, attribute)
        except AttributeError:
            continue
        candidates.append(value)
    for candidate in reversed(candidates):
        setter = getattr(candidate, "settimeout", None)
        if callable(setter):
            setter(timeout)
            return


def _materialize_content(
    url: str,
    path: Path,
    *,
    deadline: float,
    abort_guard: _CampaignAbortGuard | None,
    clock=time.monotonic,
    chunk_size: int = 1024 * 1024,
) -> None:
    """Stream to a retained partial and atomically publish before timing closes."""
    _check_abort(abort_guard, "before content open")
    open_timeout = min(180.0, _remaining(deadline, clock, "content"))
    temp = path.with_suffix(path.suffix + ".partial")
    temp.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=open_timeout) as response:
        with temp.open("wb") as handle:
            while True:
                _check_abort(abort_guard, "before content chunk read")
                remaining = _remaining(deadline, clock, "content")
                _set_response_timeout(response, remaining)
                chunk = response.read(chunk_size)
                _check_abort(abort_guard, "after content chunk read")
                _remaining(deadline, clock, "content")
                if not chunk:
                    break
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    _check_abort(abort_guard, "before atomic content save")
    _remaining(deadline, clock, "content")
    os.replace(temp, path)
    _check_abort(abort_guard, "after atomic content save")
    _remaining(deadline, clock, "content")


def _request_record(
    config: dict[str, Any],
    prompt: dict[str, Any],
    output: Path,
    base: str,
    purpose: str,
    *,
    deadline: float | None = None,
    abort_guard: _CampaignAbortGuard | None = None,
    clock=time.monotonic,
    native_contract: dict[str, Any] | None = None,
    deadline_seconds: float | None = None,
) -> dict[str, Any]:
    workload = config["workload"]
    job_id = f"{prompt['prompt_id']}-s{workload['seed']}"
    target = output / "media" / f"{job_id}.mp4"
    if deadline is None:
        deadline = clock() + (3600.0 if deadline_seconds is None else deadline_seconds)
    record: dict[str, Any] = {
        "job_id": job_id,
        "prompt_id": prompt["prompt_id"],
        "purpose": purpose,
        "prompt_en": prompt["prompt_en"],
        "seed": workload["seed"],
        "output_path": str(target),
        "cached_embeddings": False,
        "remote_conditioning": False,
        "native_conditioning": workload["native_conditioning"],
        "expected_denoiser_forwards": config["dispatch_contract"][
            "expected_denoiser_forwards"
        ],
        "generation_status": "pending",
    }
    path = output / "requests" / f"{job_id}.json"
    atomic_write_json(path, record)

    def submit() -> dict[str, Any]:
        # The first caller clock has already been read by execute_caller_request.
        request_body = _video_body(prompt["prompt_en"], config)
        _reject_payload_substitutions(request_body)
        request_body["perf_dump_path"] = str(
            (output / "perf" / f"{job_id}.json").resolve()
        )
        record["request_body"] = request_body
        return _http_json(
            "POST",
            base + "/v1/videos",
            request_body,
            timeout=30,
            deadline=deadline,
            abort_guard=abort_guard,
            clock=clock,
        )

    def poll(request_id: str) -> dict[str, Any]:
        state = _http_json(
            "GET",
            base + "/v1/videos/" + request_id,
            timeout=30,
            deadline=deadline,
            abort_guard=abort_guard,
            clock=clock,
        )
        if state.get("status") not in ("completed", "failed"):
            _sleep_until(
                1, deadline, abort_guard=abort_guard, clock=clock
            )
        return state

    def materialize(request_id: str, materialized_path: Path) -> None:
        _materialize_content(
            base + "/v1/videos/" + request_id + "/content",
            materialized_path,
            deadline=deadline,
            abort_guard=abort_guard,
            clock=clock,
        )

    failure_phase = "request_setup"
    try:
        _check_abort(abort_guard, "before isolated native request validation")
        if native_contract is None:
            native_contract = _validate_native_request_subprocess(
                config,
                prompt["prompt_en"],
                timeout=min(120.0, _remaining(deadline, clock, "native validation")),
            )
        _check_abort(abort_guard, "after isolated native request validation")
        _remaining(deadline, clock, "native request")
        record["native_request_contract"] = native_contract
        atomic_write_json(path, record)

        failure_phase = "transport"
        execute_caller_request(
            record,
            target,
            submit=submit,
            poll=poll,
            materialize=materialize,
            clock=clock,
            on_update=lambda value: atomic_write_json(path, value),
            deadline_seconds=_remaining(deadline, clock, "native request"),
        )
        _check_abort(abort_guard, "after measured caller request")
        _remaining(deadline, clock, "native request")

        # Timing has closed. Retain failures in these external checks just as
        # durably as transport failures, without extending the caller interval.
        failure_phase = "output_validation"
        validate_caller_timing(record)
        record["media"] = _media_validation(target)
        if record.get("status") != "completed" or record["media"].get("valid") is not True:
            raise RuntimeError("completed request failed full AV validation")
        record["generation_status"] = "success"
        atomic_write_json(path, record)
    except BaseException as error:
        record["generation_status"] = (
            "output_validation_failure"
            if failure_phase == "output_validation"
            else "generation_failure"
        )
        record["failure_phase"] = failure_phase
        record["error"] = f"{type(error).__name__}: {error}"
        atomic_write_json(path, record)
        raise
    return record


def _dispatch_evidence(config: dict[str, Any], output: Path, request_id: str, perf_path: Path | None = None) -> dict[str, Any]:
    """Require request-correlated native perf data, not synthetic observer rows."""
    text = (output / "server.log").read_text(encoding="utf-8", errors="replace")
    expected = config["dispatch_contract"]["expected_backend"].lower()
    observed = re.findall(r"Using ([a-z0-9_]+) attention backend", text, re.I)
    fallback = [item for item in observed if item.lower() != expected]
    if perf_path is None:
        perf_path = output / "perf" / f"{request_id}.json"
    report: dict[str, Any] | None = None
    report_error: str | None = None
    try:
        report = json.loads(perf_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        report_error = f"{type(error).__name__}: {error}"
    steps = report.get("denoise_steps_ms") if isinstance(report, dict) else None
    steps = steps if isinstance(steps, list) else []
    valid_steps = all(
        isinstance(item, dict)
        and item.get("step") == index
        and isinstance(item.get("duration_ms"), (int, float))
        and not isinstance(item.get("duration_ms"), bool)
        and math.isfinite(float(item["duration_ms"]))
        and item["duration_ms"] >= 0
        for index, item in enumerate(steps)
    )
    request_match = (
        isinstance(report, dict) and report.get("request_id") == request_id
    )
    selected_backend = (
        expected if expected in {item.lower() for item in observed} else None
    )
    forward_count = len(steps) if request_match and valid_steps else 0
    expected_forwards = config["dispatch_contract"]["expected_denoiser_forwards"]
    passed = bool(
        selected_backend == expected
        and not fallback
        and request_match
        and valid_steps
        and forward_count == expected_forwards
    )
    feature_proof = None
    if config.get("candidate_id") == "kitchen_int8":
        receipts = re.findall(
            r"kitchen_int8_quantization_receipt selected_layers=(\d+) processed_layers=(\d+) "
            r"affected_layers=(\d+) unaffected_layers=(\d+)", text
        )
        feature_ok = bool(receipts and "stage3_kitchen_backend=cuda fallback_disabled=true" in text)
        if receipts:
            selected, processed, affected, unaffected = map(int, receipts[-1])
            feature_ok = feature_ok and selected > 0 and selected == processed and affected + unaffected == selected
        feature_proof = {"kind": "native_post_adapter_int8", "receipts": receipts,
                         "cuda_only": "stage3_kitchen_backend=cuda fallback_disabled=true" in text,
                         "passed": bool(feature_ok)}
        passed = passed and feature_ok
    candidate_id = config.get("candidate_id")
    if candidate_id in {"sol_sparse_prefix3", "video_ffn_merge_unmerge", "cache_dit"}:
        events = []
        for line in text.splitlines():
            start = line.find("{")
            if start < 0:
                continue
            try:
                event = json.loads(line[start:])
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("request_id") != request_id:
                continue
            if candidate_id == "sol_sparse_prefix3" and "stage3_sol_forward " in line:
                events.append(event)
            elif candidate_id == "video_ffn_merge_unmerge" and event.get("event") == "stage3_video_ffn_merge_request":
                events.append(event)
            elif candidate_id == "cache_dit" and event.get("event") == "stage3_cache_dit_request":
                events.append(event)
        if candidate_id == "sol_sparse_prefix3":
            feature_ok = (len(events) == 4
                          and [e.get("forward_index") for e in events] == [0, 1, 2, 3]
                          and [e.get("actual_delta") for e in events] == [0, 0, 0, 50])
        elif candidate_id == "video_ffn_merge_unmerge":
            expected_rows = {"native_forwards": 4, "target_video_rows": 37296,
                             "packed_rows": 37760, "mlp_invocations": 200,
                             "mlp_input_rows_before_merge": 7552000,
                             "mlp_input_rows": 5687200}
            feature_ok = len(events) == 1 and all(events[0].get(k) == v for k, v in expected_rows.items())
        else:
            event = events[0] if len(events) == 1 else {}
            counters = event.get("counters", {})
            feature_ok = (len(events) == 1 and event.get("native_forwards") == 4
                          and event.get("scheduler_points") == 5
                          and counters.get("cached_steps") == [1, 3]
                          and all(isinstance(counters.get(k), int) and counters[k] > 0
                                  for k in ("cached_forward_count", "executed_forward_count", "transformer_executed_count")))
        feature_proof = {"kind": candidate_id, "events": events, "passed": bool(feature_ok)}
        passed = passed and feature_ok
    return {
        "request_id": request_id,
        "feature_proof": feature_proof,
        "selected_backend": selected_backend,
        "observed_backends": observed,
        "fallback_detected": bool(fallback),
        "perf_report_path": str(perf_path),
        "perf_report_error": report_error,
        "perf_report": report,
        "request_identity_match": request_match,
        "forward_count": forward_count,
        "forward_records": steps,
        "status": "passed" if passed else "blocked_missing_native_evidence",
    }


class BoundaryADiagnosticError(RuntimeError):
    """A fail-closed diagnostic outcome, separate from an intentional stop."""

    def __init__(self, classification: str, detail: str):
        super().__init__(f"{classification}: {detail}")
        self.classification = classification


def _boundary_a_patch_contract() -> dict[str, Any]:
    path = ROOT / "scripts/install_sglang_boundary_a_diagnostic.py"
    spec = importlib.util.spec_from_file_location("miniacc_boundary_a_installer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Boundary A installer contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "target": module.TARGET,
        "source_sha256": module.patched_sha256(),
        "target_module": module.TARGET_STAGE_MODULE,
        "target_qualname": module.TARGET_STAGE_QUALNAME,
    }


def _arm_boundary_a_config(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Bind one fresh baseline-only diagnostic attempt before any launch."""
    if config.get("purpose") != "boundary-a":
        raise RuntimeError("Boundary A activation requires purpose=boundary-a")
    if config.get("candidate_id") != "baseline":
        raise RuntimeError("Boundary A diagnostic is restricted to the baseline candidate")
    contract = _boundary_a_patch_contract()
    runtime = _runtime_path()
    site = runtime / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    source = site / contract["target"]
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"Boundary A executor source is missing or unsafe: {source}")
    actual_source = _sha256(source)
    if actual_source != contract["source_sha256"]:
        raise RuntimeError(
            "Boundary A diagnostic is not installed at the exact reviewed source; "
            f"expected={contract['source_sha256']} actual={actual_source}"
        )
    armed = json.loads(json.dumps(config))
    attempt_id = secrets.token_hex(16)
    armed_monotonic = time.monotonic()
    root = output.resolve()
    marker = root / f"boundary-a-{attempt_id}.marker"
    receipt = root / "boundary-a-snapshot.json"
    finalization_receipt = root / "boundary-a-native-finalization.json"
    if any(path.exists() or path.is_symlink() for path in (marker, receipt, finalization_receipt)):
        raise RuntimeError("Boundary A diagnostic artifact path is stale")
    armed["diagnostic"] = {
        "attempt_id": attempt_id,
        "attempt_armed_monotonic": armed_monotonic,
        "marker": str(marker),
        "receipt": str(receipt),
        "finalization_receipt": str(finalization_receipt),
        "source_sha256": actual_source,
        "target_module": contract["target_module"],
        "target_qualname": contract["target_qualname"],
        "single_request_only": True,
        "excluded_from_speed_and_quality": True,
    }
    armed["env"].update(
        {
            "MINIACC_BOUNDARY_A_MODE": "1",
            "MINIACC_BOUNDARY_A_ATTEMPT_ID": attempt_id,
            "MINIACC_BOUNDARY_A_ARMED_MONOTONIC": repr(armed_monotonic),
            "MINIACC_BOUNDARY_A_MARKER": str(marker),
            "MINIACC_BOUNDARY_A_RECEIPT": str(receipt),
            "MINIACC_BOUNDARY_A_FINALIZATION_RECEIPT": str(finalization_receipt),
            "MINIACC_BOUNDARY_A_EXECUTOR_SHA256": actual_source,
        }
    )
    _validate_boundary_a_effective_arming(armed, armed["env"])
    return armed


def _validate_boundary_a_effective_arming(
    config: dict[str, Any], effective: dict[str, Any]
) -> None:
    diagnostic = config.get("diagnostic")
    if config.get("candidate_id") != "baseline" or config.get("purpose") != "boundary-a":
        raise BoundaryADiagnosticError("arming_failure", "diagnostic is not baseline Boundary A")
    if not isinstance(diagnostic, dict):
        raise BoundaryADiagnosticError("arming_failure", "diagnostic contract is missing")
    expected = {
        "MINIACC_BOUNDARY_A_MODE": "1",
        "MINIACC_BOUNDARY_A_ATTEMPT_ID": diagnostic.get("attempt_id"),
        "MINIACC_BOUNDARY_A_ARMED_MONOTONIC": repr(diagnostic.get("attempt_armed_monotonic")),
        "MINIACC_BOUNDARY_A_MARKER": diagnostic.get("marker"),
        "MINIACC_BOUNDARY_A_RECEIPT": diagnostic.get("receipt"),
        "MINIACC_BOUNDARY_A_FINALIZATION_RECEIPT": diagnostic.get("finalization_receipt"),
        "MINIACC_BOUNDARY_A_EXECUTOR_SHA256": diagnostic.get("source_sha256"),
    }
    mismatches = {
        key: {"expected": value, "actual": effective.get(key)}
        for key, value in expected.items()
        if not isinstance(value, str) or not value or effective.get(key) != value
    }
    try:
        armed = float(effective.get("MINIACC_BOUNDARY_A_ARMED_MONOTONIC", ""))
    except (TypeError, ValueError):
        armed = math.nan
    paths = [diagnostic.get(key) for key in ("marker", "receipt", "finalization_receipt")]
    if (
        not math.isfinite(armed)
        or not isinstance(diagnostic.get("attempt_id"), str)
        or len(diagnostic["attempt_id"]) != 32
        or not isinstance(diagnostic.get("source_sha256"), str)
        or len(diagnostic["source_sha256"]) != 64
        or any(not isinstance(path, str) or not Path(path).is_absolute() for path in paths)
        or len(set(paths)) != 3
    ):
        mismatches["contract_shape"] = "malformed"
    if mismatches:
        raise BoundaryADiagnosticError(
            "arming_failure", f"effective diagnostic arming mismatch: {mismatches}"
        )


def _verify_boundary_a_effective_arming(
    config: dict[str, Any], process: subprocess.Popen[str], abort_guard: _CampaignAbortGuard
) -> dict[str, Any]:
    _check_abort(abort_guard, "before effective Boundary A arming read")
    effective = _effective_environment(process.pid)
    _check_abort(abort_guard, "after effective Boundary A arming read")
    _validate_boundary_a_effective_arming(config, effective)
    return effective


def _read_boundary_a_receipt(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise BoundaryADiagnosticError("unsafe_receipt", str(path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BoundaryADiagnosticError(
            "invalid_receipt", f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise BoundaryADiagnosticError("invalid_receipt", "receipt is not an object")
    return value


def _strict_int(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _validate_scalar_metadata(value: Any, path: str = "payload_metadata") -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError(f"Boundary A {path} contains non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_scalar_metadata(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise RuntimeError(f"Boundary A {path} contains a non-string key")
        shape = value.get("shape")
        if shape is not None and (
            not isinstance(shape, list)
            or any(not _strict_int(item) for item in shape)
        ):
            raise RuntimeError(f"Boundary A {path} contains an invalid shape")
        for key, item in value.items():
            _validate_scalar_metadata(item, f"{path}.{key}")
        return
    raise RuntimeError(f"Boundary A {path} contains non-scalar metadata")


def _expected_boundary_a_identities(expected: dict[str, Any]) -> dict[int, int | None]:
    identities = expected.get("owned_process_identities")
    if identities is not None:
        return identities
    return {int(expected["pid"]): expected["process_start_time_ticks"]}


def _validate_boundary_a_receipt(
    receipt: dict[str, Any], expected: dict[str, Any]
) -> None:
    contract = _boundary_a_patch_contract()
    exact = {
        "schema_version": 1,
        "diagnostic": "stage3_boundary_a",
        "attempt_id": expected["attempt_id"],
        "request_id": expected["request_id"],
        "attempt_armed_monotonic": expected.get("attempt_armed_monotonic"),
    }
    mismatches = {
        key: {"expected": value, "actual": receipt.get(key)}
        for key, value in exact.items()
        if receipt.get(key) != value
    }
    identities = _expected_boundary_a_identities(expected)
    receipt_pid = receipt.get("pid")
    receipt_ticks = receipt.get("process_start_time_ticks")
    if (
        not _strict_int(receipt_pid, minimum=1)
        or not _strict_int(receipt_ticks, minimum=1)
        or receipt_pid not in identities
        or identities.get(receipt_pid) != receipt_ticks
    ):
        mismatches["process_identity"] = {
            "expected": identities,
            "actual": {str(receipt_pid): receipt_ticks},
        }
    source = receipt.get("source")
    if not isinstance(source, dict) or (
        source.get("expected_pipeline_executor_sha256") != expected["source_sha256"]
        or source.get("actual_pipeline_executor_sha256") != expected["source_sha256"]
        or source.get("observation_error") not in (None, "")
        or not isinstance(source.get("path"), str)
        or not Path(source["path"]).is_absolute()
    ):
        raise RuntimeError(f"Boundary A receipt source identity mismatch: {source}")
    target = receipt.get("target")
    if not isinstance(target, dict) or target != {
        "module": contract["target_module"],
        "qualname": contract["target_qualname"],
        "stage_name": contract["target_qualname"],
        "stage_index": target.get("stage_index") if isinstance(target, dict) else None,
        "boundary": "before PipelineExecutor.before_stage",
    } or not _strict_int(target.get("stage_index")):
        raise RuntimeError(f"Boundary A receipt target identity mismatch: {target}")
    armed = receipt.get("attempt_armed_monotonic")
    snapshot_time = receipt.get("timestamp_monotonic")
    if (
        not _finite_number(armed)
        or not _finite_number(snapshot_time)
        or not _finite_number(receipt.get("timestamp_epoch"))
        or float(snapshot_time) < float(armed)
    ):
        raise RuntimeError("Boundary A receipt timestamps are invalid or misordered")
    host = receipt.get("host_mem_available_bytes")
    if not _strict_int(host):
        raise RuntimeError("Boundary A receipt MemAvailable is not an integer")
    allocator = receipt.get("allocator")
    if not isinstance(allocator, dict) or allocator.get("scope") != "process_allocator_device_0_not_whole_device_free":
        raise RuntimeError("Boundary A allocator scope is invalid")
    initialized = allocator.get("cuda_already_initialized")
    allocated = allocator.get("memory_allocated_bytes_device_0")
    reserved = allocator.get("memory_reserved_bytes_device_0")
    if type(initialized) is not bool:
        raise RuntimeError("Boundary A allocator initialized flag is not bool")
    if initialized:
        if not _strict_int(allocated) or not _strict_int(reserved) or reserved < allocated:
            raise RuntimeError("Boundary A allocator counters are invalid")
    elif allocated is not None or reserved is not None:
        raise RuntimeError("Boundary A unavailable allocator counters must be null")
    payload_metadata = receipt.get("payload_metadata")
    if not isinstance(payload_metadata, dict) or "extra" not in payload_metadata:
        raise RuntimeError("Boundary A payload metadata is incomplete")
    _validate_scalar_metadata(payload_metadata)
    prefetch = receipt.get("prefetch")
    if not isinstance(prefetch, dict) or (
        not isinstance(prefetch.get("registered_use_keys"), list)
        or any(
            not isinstance(item, list)
            or len(item) != 3
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            or item[2] is not None and not isinstance(item[2], str)
            for item in prefetch.get("registered_use_keys", [])
        )
        or not isinstance(prefetch.get("seen_components"), list)
        or any(not isinstance(item, str) for item in prefetch.get("seen_components", []))
        or type(prefetch.get("transformer_seen")) is not bool
        or prefetch.get("meaning") != "registration/preparation observation; not a residency byte claim"
    ):
        raise RuntimeError("Boundary A prefetch metadata is invalid")
    _validate_scalar_metadata(prefetch, "prefetch")
    if mismatches:
        raise RuntimeError(f"Boundary A receipt identity mismatch: {mismatches}")


def _validate_boundary_a_finalization_receipt(
    receipt: dict[str, Any], snapshot: dict[str, Any], expected: dict[str, Any]
) -> None:
    identities = _expected_boundary_a_identities(expected)
    pid = receipt.get("pid")
    ticks = receipt.get("process_start_time_ticks")
    exact = {
        "schema_version": 1,
        "diagnostic": "stage3_boundary_a",
        "status": "native_finalization_succeeded_after_stop",
        "attempt_id": expected["attempt_id"],
        "request_id": expected["request_id"],
        "snapshot_timestamp_monotonic": snapshot["timestamp_monotonic"],
        "native_finalization": "finish_component_residency_request_returned",
    }
    if any(receipt.get(key) != value for key, value in exact.items()):
        raise RuntimeError("Boundary A native finalization receipt identity mismatch")
    if (
        not _strict_int(pid, minimum=1)
        or not _strict_int(ticks, minimum=1)
        or identities.get(pid) != ticks
    ):
        raise RuntimeError("Boundary A native finalization process identity mismatch")
    source = receipt.get("source")
    if not isinstance(source, dict) or source.get("actual_pipeline_executor_sha256") != expected["source_sha256"]:
        raise RuntimeError("Boundary A native finalization source identity mismatch")
    final_time = receipt.get("timestamp_monotonic")
    if (
        not _finite_number(final_time)
        or not _finite_number(receipt.get("timestamp_epoch"))
        or float(final_time) < float(snapshot["timestamp_monotonic"])
        or float(receipt["timestamp_epoch"]) < float(snapshot["timestamp_epoch"])
    ):
        raise RuntimeError("Boundary A native finalization timestamps are invalid")


def _classify_boundary_a_terminal(
    state: dict[str, Any],
    receipt: dict[str, Any] | None,
    expected: dict[str, Any],
    *,
    finalization_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = state.get("status")
    if status == "completed":
        raise BoundaryADiagnosticError(
            "unexpected_success",
            "dedicated diagnostic request completed; no output may be consumed",
        )
    if status != "failed":
        raise BoundaryADiagnosticError(
            "ordinary_request_error", f"unexpected terminal state: {state}"
        )
    if receipt is None:
        raise BoundaryADiagnosticError(
            "missing_hook_or_receipt",
            f"request failed without an identity-bound Boundary A receipt: {state}",
        )
    receipt_status = receipt.get("status")
    if receipt_status in {
        "arming_failed_before_denoising_before_stage",
        "capture_failed_before_denoising_before_stage",
    }:
        source = receipt.get("source")
        observed_source = (
            source.get("actual_pipeline_executor_sha256")
            if isinstance(source, dict) else None
        )
        source_observation_succeeded = (
            isinstance(source, dict)
            and source.get("observation_error") in (None, "")
            and isinstance(observed_source, str)
            and bool(observed_source)
        )
        if source_observation_succeeded and (
            observed_source != source.get("expected_pipeline_executor_sha256")
        ):
            classification = "source_mismatch"
        elif isinstance(source, dict) and source.get("observation_error"):
            classification = "source_observation_failure"
        elif receipt_status.startswith("arming_failed"):
            classification = "arming_failure"
        else:
            classification = "capture_failure"
        raise BoundaryADiagnosticError(
            classification,
            f"{receipt.get('capture_error', 'unknown failure')}; source={source}",
        )
    if receipt_status != "stopped_before_denoising_before_stage":
        raise BoundaryADiagnosticError(
            "invalid_receipt", f"unexpected receipt status: {receipt_status!r}"
        )
    minimum_host = expected.get("min_host_mem_available_bytes", 17179869184)
    captured_host = receipt.get("host_mem_available_bytes")
    if _strict_int(captured_host) and captured_host < minimum_host:
        raise BoundaryADiagnosticError(
            "resource_breach",
            "captured host MemAvailable "
            f"{captured_host} is below {minimum_host} bytes",
        )
    try:
        _validate_boundary_a_receipt(receipt, expected)
    except (KeyError, RuntimeError) as error:
        raise BoundaryADiagnosticError("invalid_receipt", str(error)) from error
    if finalization_receipt is None:
        raise BoundaryADiagnosticError(
            "native_finalization_missing_or_failed",
            "native finish_component_residency_request success proof is missing; "
            f"terminal_state={state}",
        )
    try:
        _validate_boundary_a_finalization_receipt(
            finalization_receipt, receipt, expected
        )
    except RuntimeError as error:
        raise BoundaryADiagnosticError(
            "native_finalization_missing_or_failed", str(error)
        ) from error
    return {
        "classification": "intentional_diagnostic_stop",
        "server_state": state,
        "receipt": receipt,
        "native_finalization_receipt": finalization_receipt,
    }


def _classify_boundary_a_exception(error: BaseException) -> str:
    if isinstance(error, BoundaryADiagnosticError):
        return error.classification
    if isinstance(error, HttpTransportTimeoutError):
        return "transport_failure"
    if isinstance(error, TimeoutError):
        return "deadline_failure"
    if isinstance(error, OwnedProcessIdentityError):
        return "ownership_failure"
    text = f"{type(error).__name__}: {error}".lower()
    if "owned native process disappeared" in text or "process identity" in text:
        return "ownership_failure"
    if "sampler failure" in text or "telemetry" in text or "resource probe" in text:
        return "telemetry_or_monitor_failure"
    if isinstance(error, (urllib.error.URLError, OSError)):
        return "transport_failure"
    return "ordinary_request_error"


def _execute_boundary_a_request(
    config: dict[str, Any],
    prompt: dict[str, Any],
    output: Path,
    base: str,
    *,
    deadline: float,
    abort_guard: _CampaignAbortGuard,
    process: subprocess.Popen[str],
    start_ticks: int | None,
    clock=time.monotonic,
) -> dict[str, Any]:
    """Submit exactly one diagnostic request; never materialize, retry, or score."""
    diagnostic = config["diagnostic"]
    path = output / "boundary-a-request.json"
    record: dict[str, Any] = {
        "status": "submitting",
        "classification": None,
        "attempt_id": diagnostic["attempt_id"],
        "prompt": prompt,
        "excluded_from_speed_and_quality": True,
        "request_id": None,
        "content_materialization_forbidden": True,
        "retry_forbidden": True,
    }
    atomic_write_json(path, record)
    try:
        effective_arming = _verify_boundary_a_effective_arming(
            config, process, abort_guard
        )
        record["effective_arming"] = effective_arming
        atomic_write_json(path, record)
        _check_abort(abort_guard, "before Boundary A diagnostic POST")
        _remaining(deadline, clock, "Boundary A diagnostic")
        body = _video_body(prompt["prompt_en"], config)
        _reject_payload_substitutions(body)
        body["perf_dump_path"] = diagnostic["marker"]
        response = _http_json(
            "POST", base + "/v1/videos", body, timeout=30,
            deadline=deadline, abort_guard=abort_guard, clock=clock,
        )
        request_id = response.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise BoundaryADiagnosticError(
                "ordinary_request_error", "diagnostic POST omitted request id"
            )
        record["request_id"] = request_id
        record["status"] = "awaiting_terminal_stop"
        atomic_write_json(path, record)
        while True:
            _check_abort(abort_guard, "before Boundary A terminal poll")
            _remaining(deadline, clock, "Boundary A diagnostic")
            state = _http_json(
                "GET", base + "/v1/videos/" + request_id, timeout=30,
                deadline=deadline, abort_guard=abort_guard, clock=clock,
            )
            if state.get("status") in {"completed", "failed"}:
                break
            _sleep_until(1, deadline, abort_guard=abort_guard, clock=clock)
        receipt = _read_boundary_a_receipt(Path(diagnostic["receipt"]))
        finalization_receipt = _read_boundary_a_receipt(
            Path(diagnostic["finalization_receipt"])
        )
        owned = {
            row["pid"]: row.get("start_time_ticks")
            for row in _owned_process_tree(process.pid)
        }
        if process.pid not in owned:
            owned[process.pid] = start_ticks
        result = _classify_boundary_a_terminal(
            state,
            receipt,
            {
                "attempt_id": diagnostic["attempt_id"],
                "attempt_armed_monotonic": diagnostic["attempt_armed_monotonic"],
                "request_id": request_id,
                "owned_process_identities": owned,
                "source_sha256": diagnostic["source_sha256"],
                "min_host_mem_available_bytes": config["resource_contract"][
                    "min_host_mem_available_bytes"
                ],
            },
            finalization_receipt=finalization_receipt,
        )
        record.update(result)
        record["status"] = "intentional_diagnostic_stop_observed"
        atomic_write_json(path, record)
        return result
    except BaseException as error:
        classification = _classify_boundary_a_exception(error)
        record["status"] = "failed_closed"
        record["classification"] = classification
        record["error"] = f"{type(error).__name__}: {error}"
        atomic_write_json(path, record)
        raise


def _boundary_a_completion_status(
    result: dict[str, Any],
    sampler_state: dict[str, Any],
    cleanup: dict[str, Any],
) -> dict[str, Any]:
    """Classify without conflating native, monitor, and host teardown evidence."""
    failures: list[dict[str, Any]] = []
    request_class = result.get("classification")
    if request_class not in (None, "intentional_diagnostic_stop"):
        failures.append({"class": request_class, "error": result.get("error")})
    if sampler_state.get("violation"):
        failures.append({"class": "resource_breach", "error": sampler_state["violation"]})
    if sampler_state.get("error"):
        monitor_class = (
            "ownership_failure"
            if "owned native process disappeared" in sampler_state["error"].lower()
            else "telemetry_or_monitor_failure"
        )
        failures.append({"class": monitor_class, "error": sampler_state["error"]})
    if not cleanup.get("clean"):
        failures.append({
            "class": "cleanup_failure",
            "error": cleanup.get("error", "owned process cleanup was not clean"),
        })
    result["failures"] = failures
    if failures:
        priorities = (
            "resource_breach", "ownership_failure", "telemetry_or_monitor_failure",
            request_class, "cleanup_failure",
        )
        primary = next(
            item for name in priorities if name
            for item in failures if item["class"] == name
        )
        result.update(
            status="blocked_or_failed",
            failure_class=primary["class"],
            completion_error=primary["error"],
        )
    elif request_class == "intentional_diagnostic_stop":
        result.update(status="diagnostic_stopped", failure_class=None)
    else:
        result.update(status="blocked_or_failed", failure_class="ordinary_request_error")
    return result


def _execute_warmup(
    config: dict[str, Any],
    prompt: dict[str, Any],
    output: Path,
    base: str,
    *,
    deadline: float,
    abort_guard: _CampaignAbortGuard,
    clock=time.monotonic,
) -> dict[str, Any]:
    """Execute and persist the excluded warmup under the campaign deadline."""
    target = output / "media" / "_warmup.mp4"
    record: dict[str, Any] = {
        "prompt": prompt,
        "request_id": None,
        "started_monotonic": clock(),
        "ended_monotonic": None,
        "output": str(target),
        "excluded_from_speed_and_quality": True,
        "status": "started",
    }
    record_path = output / "warmup-excluded.json"
    atomic_write_json(record_path, record)
    try:
        _check_abort(abort_guard, "before excluded warmup payload")
        body = _video_body(prompt["prompt_en"], config)
        response = _http_json(
            "POST",
            base + "/v1/videos",
            body,
            timeout=30,
            deadline=deadline,
            abort_guard=abort_guard,
            clock=clock,
        )
        request_id = response.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError("warmup response omitted video id")
        record["request_id"] = request_id
        atomic_write_json(record_path, record)
        while True:
            _check_abort(abort_guard, "before excluded warmup poll")
            _remaining(deadline, clock, "excluded native warmup")
            state = _http_json(
                "GET",
                base + "/v1/videos/" + request_id,
                timeout=30,
                deadline=deadline,
                abort_guard=abort_guard,
                clock=clock,
            )
            _check_abort(abort_guard, "after excluded warmup poll")
            if state.get("status") in ("completed", "failed"):
                break
            _sleep_until(
                1, deadline, abort_guard=abort_guard, clock=clock
            )
        if state.get("status") != "completed":
            raise RuntimeError(f"excluded native warmup failed: {state}")
        _materialize_content(
            base + "/v1/videos/" + request_id + "/content",
            target,
            deadline=deadline,
            abort_guard=abort_guard,
            clock=clock,
        )
        record["status"] = "completed"
        record["server_result"] = state
        return record
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["ended_monotonic"] = clock()
        atomic_write_json(record_path, record)


def _stop_monitor_before_finalization(
    sampler: threading.Thread,
    stop_sampler: threading.Event,
    abort_guard: Any,
    *,
    join_timeout: float = 5.0,
) -> None:
    """Settle monitoring before final telemetry, cleanup, or normal artifacts."""
    stop_sampler.set()
    sampler.join(timeout=join_timeout)
    if sampler.is_alive():
        abort_guard.fail("resource sampler did not stop before finalization")
        sampler.join(timeout=join_timeout)
    if sampler.is_alive():
        raise MonitorTerminationError(
            f"resource sampler thread {sampler.name!r} remains unsettled; "
            "normal final sampling and cleanup are forbidden"
        )


def _campaign_completion_status(
    result: dict[str, Any],
    sampler_state: dict[str, Any],
    cleanup: dict[str, Any],
) -> dict[str, Any]:
    """Apply final resource, media, and owned-exit gates in one place."""
    if result.get("status") != "complete":
        return result
    requests = result.get("requests", [])
    invalid = any(row.get("generation_status") != "success" for row in requests)
    reason = (
        sampler_state.get("error")
        or sampler_state.get("violation")
        or ("one or more started requests failed output validation" if invalid else None)
        or (None if cleanup.get("clean") else "owned process cleanup was not clean")
    )
    if reason:
        result["status"] = "blocked_or_failed"
        result["error"] = reason
    return result


def execute_request_campaign(config: dict[str, Any], output: Path, timeout_seconds: float = 900.0) -> int:
    """Run a guarded request session; Boundary A reuses the same outer lifecycle."""
    diagnostic_mode = config.get("purpose") == "boundary-a"
    if diagnostic_mode:
        config = _arm_boundary_a_config(config, output)
    campaign_deadline = time.monotonic() + timeout_seconds
    process, start_ticks, log, capacity = _prepare_launch(
        config, output, "request_before_launch", campaign_deadline
    )
    output.joinpath("requests").mkdir()
    output.joinpath("media").mkdir()
    resources = output / "resources.jsonl"
    stop_sampler = threading.Event()
    abort_guard = _CampaignAbortGuard(process, config, start_ticks, process.pid)

    def sample() -> None:
        while not stop_sampler.wait(1.0):
            try:
                if not _owned_process(process.pid, config):
                    abort_guard.fail(
                        "owned native process disappeared during resource sampling; "
                        f"rc={process.poll()!r}"
                    )
                    stop_sampler.set()
                    return
                current = _resource_record("request_runtime", process.pid)
                append_jsonl(resources, current)
                if current["violation"]:
                    abort_guard.fail(current["violation"], violation=True)
                    stop_sampler.set()
                    return
            except BaseException as error:
                abort_guard.fail(
                    f"sampler failure: {type(error).__name__}: {error}"
                )
                stop_sampler.set()
                return
    sampler = threading.Thread(target=sample, name="stage3-resource-sampler", daemon=True)
    sampler.start()
    cleanup = None
    records: list[dict[str, Any]] = []
    try:
        base = "http://127.0.0.1:30010"
        _wait_health(
            base,
            process,
            config,
            campaign_deadline,
            output,
            abort_guard=abort_guard,
        )
        abort_guard.check("after health readiness")
        native_contract = _validate_native_request_subprocess(
            config,
            config["prompt_records"][0]["prompt_en"],
            timeout=min(
                120.0,
                _remaining(campaign_deadline, time.monotonic, "native validation"),
            ),
            diagnostic_marker=(
                config["diagnostic"]["marker"] if diagnostic_mode else None
            ),
        )
        abort_guard.check("after isolated native validation")
        _remaining(campaign_deadline, time.monotonic, "native request campaign")
        if diagnostic_mode:
            diagnostic = _execute_boundary_a_request(
                config,
                config["prompt_records"][0],
                output,
                base,
                deadline=campaign_deadline,
                abort_guard=abort_guard,
                process=process,
                start_ticks=start_ticks,
            )
            receipt_resource = _resource_record(
                "boundary_a_receipt_observed", process.pid
            )
            append_jsonl(resources, receipt_resource)
            if receipt_resource.get("violation"):
                abort_guard.fail(receipt_resource["violation"], violation=True)
            result = {
                **diagnostic,
                "status": "pending_finalization",
                "candidate_id": config["candidate_id"],
                "purpose": config["purpose"],
                "requests": [],
                "warmup_excluded": True,
                "native_request_contract": native_contract,
                "receipt_resource": receipt_resource,
                "cache_capacity": capacity,
            }
        else:
            warmup = dict(config["prompt_records"][0])
            warmup["prompt_id"] = "warmup-excluded"
            _execute_warmup(
                config,
                warmup,
                output,
                base,
                deadline=campaign_deadline,
                abort_guard=abort_guard,
            )
            abort_guard.check("after excluded warmup")
            for prompt in config["prompt_records"]:
                abort_guard.check("before measured request")
                _remaining(campaign_deadline, time.monotonic, "native request campaign")
                try:
                    record = _request_record(
                        config,
                        prompt,
                        output,
                        base,
                        config["purpose"],
                        deadline=campaign_deadline,
                        abort_guard=abort_guard,
                        native_contract=native_contract,
                    )
                except BaseException:
                    request_path = output / "requests" / (
                        f"{prompt['prompt_id']}-s{config['workload']['seed']}.json"
                    )
                    if request_path.is_file():
                        records.append(json.loads(request_path.read_text(encoding="utf-8")))
                    raise
                record["dispatch_proof"] = _dispatch_evidence(
                    config,
                    output,
                    record["server_request_id"],
                    Path(record["request_body"]["perf_dump_path"]),
                )
                atomic_write_json(
                    output / "requests" / f"{record['job_id']}.json", record
                )
                records.append(record)
                if record["dispatch_proof"]["status"] != "passed":
                    raise RuntimeError(
                        "native dispatch/forward evidence incomplete: "
                        f"{record['dispatch_proof']}"
                    )
            result = {
                "status": "complete",
                "candidate_id": config["candidate_id"],
                "purpose": config["purpose"],
                "requests": records,
                "warmup_excluded": True,
                "assessment_split": config.get("assessment_split"),
                "native_request_contract": native_contract,
                "cache_capacity": capacity,
            }
    except Exception as error:
        result = {
            "status": "blocked_or_failed",
            "candidate_id": config["candidate_id"],
            "purpose": config["purpose"],
            "requests": records,
            "error": f"{type(error).__name__}: {error}",
            "assessment_split": config.get("assessment_split"),
            "cache_capacity": capacity,
        }
        if diagnostic_mode:
            result["classification"] = _classify_boundary_a_exception(error)
    finally:
        # No final telemetry, normal cleanup, or success artifact may race the
        # monitor. If it cannot settle, only the identity-checked abort in the
        # guard is permitted and the unsettled state is persisted as a blocker.
        try:
            _stop_monitor_before_finalization(
                sampler, stop_sampler, abort_guard, join_timeout=5.0
            )
            monitor_settled = True
        except MonitorTerminationError as error:
            monitor_settled = False
            abort_guard.state["error"] = (
                abort_guard.state["error"] or f"{type(error).__name__}: {error}"
            )
        if monitor_settled:
            try:
                final_resource = _resource_record(
                    "request_before_cleanup", process.pid
                )
                append_jsonl(resources, final_resource)
                if final_resource["violation"]:
                    abort_guard.state["violation"] = (
                        abort_guard.state["violation"]
                        or final_resource["violation"]
                    )
            except BaseException as error:
                final_resource = {
                    "phase": "request_before_cleanup",
                    "error": f"{type(error).__name__}: {error}",
                }
                abort_guard.state["error"] = (
                    abort_guard.state["error"] or final_resource["error"]
                )
            try:
                cleanup = _cleanup_owned_process(
                    process, config, output, start_ticks
                )
            except BaseException as error:
                cleanup = {
                    "clean": False,
                    "error": f"{type(error).__name__}: {error}",
                }
                abort_guard.state["error"] = (
                    abort_guard.state["error"] or cleanup["error"]
                )
        else:
            final_resource = {
                "phase": "request_before_cleanup",
                "status": "not_attempted_monitor_unsettled",
            }
            cleanup = {
                "clean": False,
                "status": "normal_cleanup_not_attempted_monitor_unsettled",
                "abort_receipt": abort_guard.state.get("abort_receipt"),
            }
        log.close()
        result["monitor_settled_before_finalization"] = monitor_settled
        result["final_resource"] = final_resource
        result["sampler"] = dict(abort_guard.state)
        result["cleanup"] = cleanup
        if diagnostic_mode:
            _boundary_a_completion_status(result, abort_guard.state, cleanup)
        else:
            _campaign_completion_status(result, abort_guard.state, cleanup)
    result_path = output / (
        "boundary-a-result.json" if diagnostic_mode else "campaign.json"
    )
    atomic_write_json(result_path, result)
    if not diagnostic_mode:
        with (ROOT / "BENCHMARKS.md").open("a", encoding="utf-8") as benchmark_log:
            benchmark_log.write("\n- **Stage 3 native request campaign attempt**: " + json.dumps(result, sort_keys=True) + "\n")
            benchmark_log.flush()
            os.fsync(benchmark_log.fileno())
    expected_status = "diagnostic_stopped" if diagnostic_mode else "complete"
    return 0 if result["status"] == expected_status else 1


def execute_loader_fit(config: dict[str, Any], output: Path, timeout_seconds: float = 900.0) -> int:
    """Launch native server with true warmup-off config, then retain full cleanup evidence."""
    process, start_ticks, log, capacity = _prepare_launch(config, output)
    resources = output / "resources.jsonl"
    started = time.monotonic()
    ready = False
    reason = "timeout"
    cleanup = None
    try:
        while time.monotonic() - started < timeout_seconds:
            if not _owned_process(process.pid, config):
                reason = "owned_process_disappeared"
                break
            current = _resource_record("loader_fit_load", process.pid)
            append_jsonl(resources, current)
            if current["violation"]:
                reason = current["violation"]
                break
            text = (output / "server.log").read_text(encoding="utf-8", errors="replace")
            observed_backends = re.findall(r"Using ([a-z0-9_]+) attention backend", text, re.I)
            if any(item.lower() != config["dispatch_contract"]["expected_backend"].lower() for item in observed_backends):
                reason = f"backend fallback or mismatch observed: {observed_backends}"
                break
            if re.search(r"(Application startup complete|server is ready|Uvicorn running)", text, re.I):
                if config["dispatch_contract"]["expected_backend"].lower() not in {item.lower() for item in observed_backends}:
                    reason = "required native attention backend observation missing"
                    break
                ready = True
                reason = "server_ready_after_native_load"
                break
            time.sleep(1)
    finally:
        cleanup = _cleanup_owned_process(process, config, output, start_ticks)
        log.close()
    result = {"candidate_id": config["candidate_id"], "mode": "loader-fit", "pid": process.pid, "ready": ready, "reason": reason, "elapsed_seconds": time.monotonic() - started, "argv": config["argv"], "dispatch_contract": config["dispatch_contract"], "cleanup": cleanup, "cache_capacity": capacity}
    atomic_write_json(output / "loader-fit.json", result)
    with (ROOT / "BENCHMARKS.md").open("a", encoding="utf-8") as benchmark_log:
        benchmark_log.write("\n- **Stage 3 loader-fit admission attempt**: " + json.dumps(result, sort_keys=True) + "\n")
        benchmark_log.flush()
        os.fsync(benchmark_log.fileno())
    return 0 if ready else 1


def main(argv=None) -> int:
    selected_argv = list(sys.argv[1:] if argv is None else argv)
    if selected_argv == ["--cpu-validate-request"]:
        payload = json.loads(sys.stdin.read())
        print(json.dumps(_cpu_native_request_validation(payload), allow_nan=False))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default="baseline")
    parser.add_argument("--purpose", choices=("loader-fit", "serve", "speed", "quality", "boundary-a"), default="loader-fit")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "stage3/local-run")
    parser.add_argument("--execute-loader-fit", action="store_true")
    parser.add_argument("--execute-request", action="store_true", help="execute guarded HTTP request path; requires an independently reviewed campaign authorization")
    parser.add_argument("--execute-boundary-a-diagnostic", action="store_true", help="execute one fail-closed Boundary A request; requires separate parent installation and execution authorization")
    args = parser.parse_args(argv)
    manifest = _load_manifest(args.manifest)
    config = build_local_config(manifest, args.candidate_id, args.purpose)
    execution_modes = sum(
        bool(item)
        for item in (
            args.execute_loader_fit,
            args.execute_request,
            args.execute_boundary_a_diagnostic,
        )
    )
    if execution_modes == 0:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    if execution_modes != 1:
        raise SystemExit("choose one execution mode")
    if args.execute_loader_fit:
        if args.purpose != "loader-fit":
            raise SystemExit("loader-fit execution requires --purpose loader-fit")
        return execute_loader_fit(config, args.output)
    if args.execute_boundary_a_diagnostic:
        if args.purpose != "boundary-a":
            raise SystemExit("Boundary A execution requires --purpose boundary-a")
        return execute_request_campaign(config, args.output)
    if args.purpose not in ("speed", "quality"):
        raise SystemExit("request execution requires --purpose speed or quality")
    return execute_request_campaign(config, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
