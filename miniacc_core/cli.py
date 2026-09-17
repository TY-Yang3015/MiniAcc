"""Thin command adapters for preparation, registry export and owned probes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex

from . import data
from .artifacts import ArtifactStore
from .config import (
    COMFY_TURBO_ADAPTER,
    COMFY_TURBO_AUDIO_VAE,
    COMFY_TURBO_CLIP,
    COMFY_TURBO_DIFFUSION,
    COMFY_TURBO_VIDEO_VAE,
    CandidateConfig,
    HostConfig,
    RunConfig,
    load_hardware_config,
    WorkloadConfig,
)
from .evaluation import MediaValidationError, MediaValidator
from .interface import ApplicationInterface
from .models import ModelManagerFactory
from .probe import (
    OwnedProcessError,
    ResourceGuard,
    _exclusive_path,
    _write_report,
    comfy_server_command,
    create_run_dir,
    resource_snapshot,
)
from .registry import build_candidate_registry
from .runtime import ResourceBudget, ResourceGuardViolation, RuntimeOwner

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/stage1/vbench_dev_manifest.json"
DEFAULT_OUTPUT = ROOT / "artifacts/local_probe"
DEFAULT_COMFY_OUTPUT = ROOT / ".local/comfy-output"
DEFAULT_MODEL_ROOT = ROOT / ".local/models"


def preparation_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage-1 preparation and audit planning"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("preflight")
    inventory.add_argument("--output", type=Path)
    prepare = commands.add_parser("prepare-prompts")
    prepare.add_argument(
        "--metadata", type=Path, default=ROOT / "data/stage1/vbench_full_info.json"
    )
    prepare.add_argument(
        "--audit", type=Path, default=ROOT / "data/stage1/source_audit.json"
    )
    prepare.add_argument("--output", type=Path)
    plan = commands.add_parser("candidate-plan")
    plan.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    plan.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = data.preflight()
        elif args.command == "prepare-prompts":
            audit = json.loads(args.audit.read_text(encoding="utf-8"))
            source = next(
                item for item in audit["sources"] if item["id"] == "vbench_metadata"
            )
            result = data.build_prompt_manifest(args.metadata.read_bytes(), source)
        else:
            result = build_candidate_registry(args.model_root)
        data.write_json(args.output, result)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.exit(1, f"Stage-1 preparation failed: {error}\n")


def _candidate_from_args(args):
    if args.family == "comfyui-turbo":
        defaults = {
            "diffusion_name": COMFY_TURBO_DIFFUSION,
            "clip_name": COMFY_TURBO_CLIP,
            "video_vae_name": COMFY_TURBO_VIDEO_VAE,
            "audio_vae_name": COMFY_TURBO_AUDIO_VAE,
        }
        adapter_name = args.adapter_name or COMFY_TURBO_ADAPTER
    else:
        base = CandidateConfig()
        defaults = {
            "diffusion_name": base.diffusion_name,
            "clip_name": base.clip_name,
            "video_vae_name": base.video_vae_name,
            "audio_vae_name": base.audio_vae_name,
        }
        adapter_name = args.adapter_name
    return CandidateConfig(
        family=args.family,
        diffusion_name=args.diffusion_name or defaults["diffusion_name"],
        clip_name=args.clip_name or defaults["clip_name"],
        video_vae_name=args.video_vae_name or defaults["video_vae_name"],
        audio_vae_name=args.audio_vae_name or defaults["audio_vae_name"],
        adapter_name=adapter_name,
        adapter_scale=args.adapter_scale,
        adapter_alpha=args.adapter_alpha,
        file_backed_dit=args.file_backed_dit,
        steps=args.steps if args.steps is not None else (4 if adapter_name else 49),
        video_shift=(
            args.video_shift
            if args.video_shift is not None
            else (6.0 if adapter_name else 12.0)
        ),
        audio_shift=args.audio_shift if args.audio_shift is not None else 3.0,
    )


def _add_candidate_arguments(parser):
    parser.add_argument(
        "--family",
        choices=sorted(ModelManagerFactory.KNOWN_FAMILIES),
        default="comfyui-base",
    )
    parser.add_argument("--diffusion-name")
    parser.add_argument("--clip-name")
    parser.add_argument("--video-vae-name")
    parser.add_argument("--audio-vae-name")
    parser.add_argument("--adapter-name")
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--adapter-alpha", type=float)
    parser.add_argument(
        "--file-backed-dit",
        action="store_true",
        help="Map full BF16 H3 weights for the original reference or four-step controls",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--video-shift", type=float)
    parser.add_argument("--audio-shift", type=float)


def probe_main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded local ComfyUI probe")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--job-id", help="Exact frozen job ID; default is the first job")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--comfy-output", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_OUTPUT / "runs")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument(
        "--python",
        default=str(Path.home() / ".cache/miniacc/89a54668-runtime/bin/python"),
    )
    parser.add_argument(
        "--comfy-main", type=Path, default=ROOT / ".local/ComfyUI/main.py"
    )
    parser.add_argument("--bootstrap", type=Path, default=ROOT / "miniacc_bootstrap.py")
    parser.add_argument(
        "--extra-model-paths", type=Path, default=ROOT / ".local/extra_model_paths.yaml"
    )
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--allocator-gib", type=float)
    parser.add_argument("--hardware-config", type=Path,
                        help="Consumed JSON-compatible YAML hardware profile")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--vae-device", choices=("cpu", "gpu"), default="gpu")
    _add_candidate_arguments(parser)
    args = parser.parse_args(argv)

    try:
        candidate = _candidate_from_args(args)
        workload = WorkloadConfig()
        hardware = load_hardware_config(args.hardware_config) if args.hardware_config else None
        allocator_gib = args.allocator_gib if args.allocator_gib is not None else (hardware.allocator_gib if hardware else 16.0)
        runtime_profile = hardware.runtime_profile if hardware else "default"
        min_vram_reserve_mib = max(2048, int((hardware.vram_headroom_gib if hardware else 2.0) * 1024))
        manifest = data.PromptDataModule(args.manifest)
        job = manifest.job(args.job_id)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "failure_class": "candidate_setup_failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                indent=2,
            )
        )
        return 1

    report = None
    raw_log = None
    run_dir = None
    try:
        run_config = RunConfig(
            args.manifest, args.run_root, args.output, args.comfy_output
        )
        host = HostConfig(
            allocator_gib=allocator_gib,
            timeout_seconds=args.timeout,
            cpu_threads=args.cpu_threads,
            vae_device=args.vae_device,
            min_vram_reserve_mib=min_vram_reserve_mib,
            runtime_profile=runtime_profile,
        )
        run_dir = create_run_dir(run_config.run_root, args.run_dir).resolve()
        report = _exclusive_path(
            (run_config.output or run_dir / "result.json").resolve()
        )
        media_root = _exclusive_path(
            (run_config.comfy_output or run_dir / "media").resolve()
        )
        raw_log = _exclusive_path(
            (args.server_log or run_dir / "server.raw.log").resolve()
        )
        sample_log = _exclusive_path(run_dir / "resources.jsonl")
        checkpoint = _exclusive_path(run_dir / "av-latent.safetensors")
        history_path = _exclusive_path(run_dir / "history.json")
        internal = tuple(
            run_dir / name
            for name in (
                "candidate.json",
                "application-submissions.json",
                "submitted-graph.json",
            )
        )
        paths = (
            report,
            media_root,
            raw_log,
            sample_log,
            checkpoint,
            history_path,
            *internal,
        )
        for index, left in enumerate(paths):
            for right in paths[index + 1 :]:
                if (
                    left == right
                    or left.is_relative_to(right)
                    or right.is_relative_to(left)
                ):
                    raise ValueError(
                        "report, media, raw-log and internal artifact paths conflict"
                    )
        required = (
            args.python,
            args.comfy_main,
            args.bootstrap,
            args.extra_model_paths,
        )
        if not all(Path(path).is_file() for path in required):
            raise FileNotFoundError(
                "Python, ComfyUI source/config, or bootstrap path is missing"
            )

        command = comfy_server_command(
            args.python,
            args.comfy_main,
            output_root=media_root,
            extra_model_paths=args.extra_model_paths,
            bootstrap=args.bootstrap,
            port=args.port,
            allocator_gib=host.allocator_gib,
            cpu_threads=host.cpu_threads,
            vae_device=host.vae_device,
            file_backed_dit=candidate.file_backed_dit,
            file_backed_text=(candidate.file_backed_dit and candidate.clip_name == COMFY_TURBO_CLIP),
            hardware_config=hardware,
        )
        env = os.environ.copy()
        env.update(
            {
                "MINIACC_RUN_ID": run_dir.name,
                "MINIACC_JOB_ID": job["id"],
                "MINIACC_LATENT_CHECKPOINT": str(checkpoint),
            }
        )
        artifact_store = ArtifactStore(run_dir)
        artifact_store.write_json(
            "candidate.json",
            {
                "candidate": {
                    "family": candidate.family,
                    "diffusion_name": candidate.diffusion_name,
                    "clip_name": candidate.clip_name,
                    "video_vae_name": candidate.video_vae_name,
                    "audio_vae_name": candidate.audio_vae_name,
                    "adapter_name": candidate.adapter_name,
                    "adapter_scale": candidate.adapter_scale,
                    "file_backed_dit": candidate.file_backed_dit,
                    "steps": candidate.steps,
                    "video_shift": candidate.video_shift,
                    "audio_shift": candidate.audio_shift,
                    "capabilities": sorted(candidate.capabilities),
                },
                "workload": {
                    "width": workload.width,
                    "height": workload.height,
                    "frames": workload.frames,
                    "fps": workload.fps,
                    "batch_size": workload.batch_size,
                    "audio_rate": workload.audio_rate,
                    "audio_channels": workload.audio_channels,
                    "seeds": list(workload.seeds),
                },
                "server_command": command,
                "manifest": str(args.manifest),
                "hardware_config_path": str(args.hardware_config) if args.hardware_config else None,
                "hardware_config": (vars(hardware) if hardware else None),
            },
        )
        budget = ResourceBudget(
            gpu_index=host.gpu_index,
            min_vram_reserve_mib=host.min_vram_reserve_mib,
            min_host_headroom_bytes=host.min_host_headroom_bytes,
            min_project_free_bytes=host.min_project_free_bytes,
            min_executable_free_bytes=host.min_executable_free_bytes,
        )
        guard = ResourceGuard(sample_log=sample_log, budget=budget)
        manager = ModelManagerFactory().create(
            candidate, workload=workload, base_url=f"http://127.0.0.1:{args.port}"
        )
        application = ApplicationInterface(
            data=manifest,
            manager=manager,
            runtime=RuntimeOwner(
                guard=guard, timeout_seconds=host.timeout_seconds, host=host
            ),
            artifacts=artifact_store,
            evaluator=MediaValidator(workload.media_expectation(), hash_media=False),
        )
        guard.check()
        completed = application.run(
            command,
            raw_log,
            output_root=media_root,
            jobs=(job,),
            timeout=host.timeout_seconds,
            env=env,
            readiness_log=raw_log,
            expected_run_id=run_dir.name,
        )
        if not isinstance(completed, list) or len(completed) != 1:
            raise RuntimeError("application returned an unexpected completed result")
        result = completed[0]
        process_start = application.runtime.process_started_at
        completion = result.pop("completion_monotonic", None)
        result["cold_e2e_seconds"] = (
            completion - process_start
            if completion is not None and process_start is not None else None
        )
        result["process_to_cleanup_seconds"] = application.runtime.process_to_cleanup_seconds
        result["timing_note"] = (
            "cold_e2e_seconds: before server spawn through observed history completion/output materialization; "
            "elapsed_seconds: request-to-history; process_to_cleanup_seconds additionally includes "
            "output validation and owned teardown. Process-cold, not OS-cache-cold; no warm measurement."
        )
        result.update(
            {
                "run_id": run_dir.name,
                "raw_log": str(raw_log),
                "server_command": command,
                "candidate": {
                    "family": candidate.family,
                    "diffusion": candidate.diffusion_name,
                    "adapter": candidate.adapter_name,
                    "file_backed_dit": candidate.file_backed_dit,
                    "graph_sha256": result.get("graph_sha256"),
                },
            }
        )
        _write_report(report, result)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        message = str(error)
        if isinstance(error, MediaValidationError):
            failure_class = "output_validation_failure"
        elif "OutOfMemory" in message or "execution failed" in message:
            failure_class = "inference_failure"
        elif isinstance(error, ResourceGuardViolation):
            failure_class = "resource_guard_failure"
        elif isinstance(error, OwnedProcessError):
            failure_class = "bounded_termination"
        elif isinstance(error, TimeoutError):
            failure_class = "timeout"
        elif "admission rejected" in message:
            failure_class = "resource_admission_failure"
        else:
            failure_class = "runtime_setup_failure"
        failure = {
            "job": job,
            "run_id": run_dir.name if run_dir is not None else None,
            "raw_log": str(raw_log) if raw_log is not None else None,
            "status": "failed",
            "failure_class": failure_class,
            "error_type": type(error).__name__,
            "error": message,
        }
        try:
            failure["resource_at_failure"] = resource_snapshot()
        except Exception:
            failure["resource_at_failure"] = None
        try:
            if report is not None:
                _write_report(report, failure)
        except (FileExistsError, OSError, TypeError):
            pass
        print(json.dumps(failure, indent=2))
        return 1
