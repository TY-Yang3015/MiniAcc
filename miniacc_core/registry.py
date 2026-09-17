"""Source-backed Stage 1 candidate registry and native run contracts.

This module records eligibility and blockers; it does not download, import, or
execute any model backend.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shlex

from .config import (
    COMFY_NVFP4_CLIP,
    COMFY_TURBO_ADAPTER,
    CandidateConfig,
    WorkloadConfig,
)


REGISTRY_VERSION = "stage1-four-families-v1"
ARTIFACT_FILES = (
    "candidate.json",
    "submitted-graph.json",
    "application-submissions.json",
    "server.raw.log",
    "resources.jsonl",
    "history.json",
    "av-latent.safetensors",
    "result.json",
)


def _artifact_contract(candidate_id: str, *, implemented: bool = True) -> dict:
    return {
        "root_template": f".local/artifacts/stage1-native/{candidate_id}/<unique-run-dir>",
        "status": "implemented" if implemented else "planned_not_implemented",
        "create_only": True,
        "files": list(ARTIFACT_FILES) if implemented else [],
        "telemetry": {
            "resource_samples": "resources.jsonl; sampled extrema, not instantaneous true peaks",
            "forward_markers": "server.raw.log; count explicit start/end/error markers only",
            "timing": [
                "startup",
                "compilation",
                "input_encoding",
                "denoising",
                "decode",
                "mux",
                "e2e",
            ],
            "media_validation": (
                "ffprobe and full ffmpeg decode through MediaValidator"
                if implemented
                else "planned; no runner is implemented"
            ),
            "provenance": (
                "candidate.json contains exact command, workload, source and asset observations"
                if implemented
                else "planned; no runner is implemented"
            ),
        },
    }


def _comfy_command(
    candidate_id: str,
    adapter: str,
    steps: int,
    video_shift: float,
    candidate: CandidateConfig | None = None,
) -> str:
    candidate = candidate or CandidateConfig(
        family="comfyui-turbo",
        adapter_name=adapter,
        steps=steps,
        video_shift=video_shift,
        audio_shift=3.0,
    )
    flags = [
        "python3",
        "local_probe.py",
        "--family",
        "comfyui-turbo",
        "--diffusion-name",
        candidate.diffusion_name,
        "--clip-name",
        candidate.clip_name,
        "--video-vae-name",
        candidate.video_vae_name,
        "--audio-vae-name",
        candidate.audio_vae_name,
        "--adapter-name",
        adapter,
        "--adapter-scale",
        str(candidate.adapter_scale),
        "--steps",
        str(steps),
        "--video-shift",
        f"{video_shift:g}",
        "--audio-shift",
        f"{candidate.audio_shift:g}",
        *( ["--file-backed-dit"] if candidate.file_backed_dit else [] ),
        "--vae-device",
        "gpu",
        "--allocator-gib",
        "16",
        "--timeout",
        "3600",
        "--manifest",
        "data/stage1/vbench_dev_manifest.json",
        "--run-root",
        f".local/artifacts/stage1-native/{candidate_id}/runs",
    ]
    return " ".join(shlex.quote(value) for value in flags)


def _sglang_command(adapter_args: str) -> str:
    return (
        "${MINIACC_SGLANG_PYTHON:?set executable-home runtime} -m sglang.launch_server "
        "--model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va "
        "--attention-backend fa --quantization kitchen_int8 "
        "--layerwise-offload-components dit,text_encoder "
        "--dit-offload-prefetch-size 1 --dit-layerwise-resident-layers 0 "
        f"{adapter_args}"
    )


def _base_candidate(
    candidate_id: str,
    family: str,
    status: str,
    blockers: list[str],
    *,
    config: dict,
    invocation: dict,
    sources: list[dict],
) -> dict:
    return {
        "id": candidate_id,
        "family": family,
        "eligibility": status,
        "blockers": blockers,
        "config": config,
        "invocation": invocation,
        "artifact_contract": _artifact_contract(
            candidate_id,
            implemented=family.startswith("comfyui")
            and invocation.get("command") is not None,
        ),
        "sources": sources,
    }


def build_candidate_registry(model_root: Path) -> dict:
    """Build an immutable-in-inputs status snapshot from the public audit."""
    workload = WorkloadConfig()
    common = {
        "task": "t2va",
        "geometry": workload.media_expectation(),
        "batch_size": workload.batch_size,
        "seeds": list(workload.seeds),
        "prompt_policy": "data/stage1/vbench_dev_manifest.json exact prompt_en bytes",
        "resource_policy": {
            "allocator_gib": 16,
            "deadline_seconds": 3600,
            "min_free_vram_mib": 2048,
            "min_host_headroom_bytes": 8 * 1024**3,
        },
    }
    candidates = []

    for steps, suffix in ((4, "T4"), (8, "T8")):
        adapter = (
            f"minimax_h3_fl2v_turbo_{steps}step_v1.0_768p_comfyui_bf16.safetensors"
        )
        candidate_config = CandidateConfig(
            family="comfyui-turbo",
            diffusion_name="minimax_h3_fl2va_bf16.safetensors",
            adapter_name=adapter,
            steps=steps,
            video_shift=6.0,
            audio_shift=3.0,
        )
        from .models import CandidateCatalog

        observation = CandidateCatalog(model_root).inspect(candidate_config)
        candidates.append(
            _base_candidate(
                f"S1-FILT-06-CUI-{suffix}",
                "comfyui-turbo",
                "planned_missing_assets",
                [
                    "official BF16 Comfy FL2VA base and BF16 text encoder are not present in the local asset root",
                    "BF16-text control remains planned; adapter/base compatibility and native residency are unverified",
                ],
                config={
                    **common,
                    "backend": "ComfyUI",
                    "base": candidate_config.diffusion_name,
                    "clip": candidate_config.clip_name,
                    "video_vae": candidate_config.video_vae_name,
                    "audio_vae": candidate_config.audio_vae_name,
                    "adapter": adapter,
                    "steps": steps,
                    "video_shift": 6.0,
                    "audio_shift": 3.0,
                    "attention": "Comfy default; record actual dispatch",
                    "encoder_precision": "bf16",
                    "residency": "planned; BF16 DiT and encoder assets absent locally",
                    "source_tree": {
                        "revision": "a98869194787969724c7425d95d0ed73ce9202af",
                        "sha256": "12e846693057db59893aea75cae8801289a123238021c1d4c46eeb601f45f41d",
                    },
                    "local_asset_observation": observation.__dict__,
                },
                invocation={
                    "command": _comfy_command(
                        f"S1-FILT-06-CUI-{suffix}",
                        adapter,
                        steps,
                        6.0,
                        candidate_config,
                    ),
                    "expected_forward_count": steps,
                    "setup_required": True,
                },
                sources=[
                    {
                        "name": "ModelTC Minimax-H3-Turbo",
                        "revision": "02e26d591f7a04d5d1a074c9566d5dd4f22f6225",
                    },
                    {
                        "name": "LightX2V Comfy adapter",
                        "variant": f"768p {steps}-step; source audit pin",
                    },
                ],
            )
        )

    nvfp4_candidate = CandidateConfig(
        family="comfyui-turbo",
        diffusion_name="minimax_h3_fl2va_bf16.safetensors",
        clip_name=COMFY_NVFP4_CLIP,
        adapter_name=COMFY_TURBO_ADAPTER,
        steps=4,
        video_shift=6.0,
        audio_shift=3.0,
        file_backed_dit=True,
    )
    nvfp4_observation = CandidateCatalog(model_root).inspect(nvfp4_candidate)
    candidates.append(
        _base_candidate(
            "S1-STRUCT-04-CUI-T4-NVFP4-TEXT",
            "comfyui-turbo",
            "source_ready_setup_required",
            [
                "full BF16 FL2VA DiT is not present in the local asset root",
                "NVFP4 text path is only the RUN-05 CPU-emulated control; native compatibility and residency are unverified",
                "this is not a BF16/Q0/source-exact teacher or validated Turbo result",
            ],
            config={
                **common,
                "backend": "ComfyUI",
                "base": nvfp4_candidate.diffusion_name,
                "clip": nvfp4_candidate.clip_name,
                "video_vae": nvfp4_candidate.video_vae_name,
                "audio_vae": nvfp4_candidate.audio_vae_name,
                "adapter": nvfp4_candidate.adapter_name,
                "steps": 4,
                "video_shift": 6.0,
                "audio_shift": 3.0,
                "encoder_precision": "nvfp4_awq_cpu_emulated_control",
                "residency": "text encoder CPU-emulated as in RUN-05; DiT/VAE placement must be measured",
                "file_backed_dit": True,
                "pinned_memory": "disabled by owned bootstrap for this opt-in policy",
                "source_tree": {
                    "revision": "a98869194787969724c7425d95d0ed73ce9202af",
                    "sha256": "12e846693057db59893aea75cae8801289a123238021c1d4c46eeb601f45f41d",
                },
                "local_asset_observation": nvfp4_observation.__dict__,
            },
            invocation={
                "command": _comfy_command(
                    "S1-STRUCT-04-CUI-T4-NVFP4-TEXT",
                    nvfp4_candidate.adapter_name,
                    4,
                    6.0,
                    nvfp4_candidate,
                ),
                "expected_forward_count": 4,
                "setup_required": True,
                "control_variant": "existing RUN-05 NVFP4 CPU-emulated text encoder; distinct from BF16-text controls",
            },
            sources=[
                {
                    "name": "Comfy-Org MiniMax-H3 tree",
                    "revision": "a98869194787969724c7425d95d0ed73ce9202af",
                    "sha256": "12e846693057db59893aea75cae8801289a123238021c1d4c46eeb601f45f41d",
                },
                {
                    "name": "LightX2V Comfy adapter",
                    "variant": "768p 4-step; source audit pin",
                },
                {
                    "name": "RUN-05 encoder control",
                    "provenance": "existing local NVFP4 CPU-emulated path; not BF16/Q0",
                },
            ],
        )
    )

    candidates.append(
        _base_candidate(
            "S1-AUDIT-CUI-PRUNED-TURBO",
            "comfyui-turbo",
            "blocked_unverified_export",
            [
                "Winnougan header metadata exposes a graph, not merge/rebuild provenance",
                "required merge adapter -> rebuild AdaLN -> quantize sequence is not verified",
                "RUN-01 dynamic LoRA on pruned W4 is explicitly excluded",
            ],
            config={
                **common,
                "backend": "ComfyUI",
                "base": "Winnougan pruned_bf16 / ConvRot candidate",
                "adapter": "Turbo v1.2 header graph lead",
                "header_artifact": ".local/artifacts/stage1-family-registry/public-header-audit/report.json",
                "header_tensor_bytes_inspected": 0,
            },
            invocation={"command": None, "blocked": True},
            sources=[
                {
                    "name": "Winnougan FLA-Turbo header",
                    "revision": "534baff6424db916e8150f3515303fed8faa56db",
                },
            ],
        )
    )

    for candidate_id, adapter_args, steps, lora_note in (
        (
            "S1-FILT-06-SGL-LX4",
            "--lora-path <LightX2V-4step-file> --lora-weight-name minimax_h3_fl2v_turbo_4step_v0.1.safetensors --lora-scale 1.0 --lora-alpha 8 --lora-merge-mode auto",
            5,
            "LightX2V: request 5 scheduler points / 4 denoiser calls",
        ),
        (
            "S1-FILT-06-SGL-LARRY8",
            "--lora-path <Larry-v4-file> --lora-weight-name minimax_h3_turbo_v4_step600_ema.safetensors --lora-scale 1.0 --lora-merge-mode auto",
            9,
            "Larry: request 9 scheduler points / 8 denoiser calls",
        ),
    ):
        candidates.append(
            _base_candidate(
                candidate_id,
                "sglang-h3",
                "source_ready_setup_required",
                [
                    "sglang diffusion package is not installed in the pinned executable runtime",
                    "model and adapter assets are not present locally",
                    "16-GiB MiniAcc cap and resident behavior remain unmeasured",
                ],
                config={
                    **common,
                    "backend": "SGLang diffusion",
                    "model_variant": "fl2va",
                    "attention": "fa first; Sage/Sol/hybrid are separate variants",
                    "quantization": "kitchen_int8 online; do not substitute external INT8",
                    "flow_shift": 12.0,
                    "audio_shift": 3.0,
                    "num_inference_steps": steps,
                    "adapter_contract": lora_note,
                    "setup_required": [
                        "sglang[diffusion]",
                        "source-backed model and adapter assets",
                    ],
                },
                invocation={
                    "command": _sglang_command(adapter_args),
                    "request": {
                        "width": 1344,
                        "height": 768,
                        "num_frames": 124,
                        "fps": 24,
                        "audio_sample_rate": 32000,
                        "audio_channels": 2,
                        "seed": "manifest seed",
                        "prompt_fields": "exact frozen prompt; no rewrite",
                    },
                },
                sources=[
                    {
                        "name": "SGLang MiniMax H3 documentation",
                        "url": "https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3.md",
                    },
                    {
                        "name": "SGLang attention backend documentation",
                        "url": "https://docs.sglang.io/docs/sglang-diffusion/attention_backends.md",
                    },
                ],
            )
        )

    candidates.append(
        _base_candidate(
            "S1-FILT-06-FHD-4",
            "fast-h3-dense",
            "weight_ready_setup_required",
            [
                "FastVideo runtime is not installed",
                "compatible compressed export is absent from the audited public card",
                "RTX 4090 native runtime and residency are untested",
            ],
            config={
                **common,
                "backend": "FastVideo",
                "model": "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree",
                "checkpoint_step": 1000,
                "attention": "FLASH_ATTN dense",
                "scheduler_points": 5,
                "denoiser_forwards": 4,
                "video_shift": 12.0,
                "audio_shift": 3.0,
                "compressed_export": "required; not source-verified",
            },
            invocation={
                "command": "FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN python3 examples/inference/basic/basic_minimax_h3_t2v.py --model_path FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree --steps 5",
                "setup_required": True,
            },
            sources=[
                {
                    "name": "FastH3 Dense model card",
                    "revision": "f58c2eb023afcad8379de99bebe14c43b23b49b8",
                },
                {
                    "name": "FastVideo",
                    "revision": "a943220c115228ade5d57b3bab9a6a87fd600a10",
                },
            ],
        )
    )

    candidates.append(
        _base_candidate(
            "S1-FILT-06-FVS-4",
            "fast-h3-vsa",
            "source_ready_runtime_sm_validation",
            [
                "actual SM80/SM89 model-forward and native AV validation are pending",
                "optional sm100a CUDA VSA kernel is not applicable on SM80/SM89",
                "dense fallback is not permitted; only the trained VSA checkpoint with tile64 Triton is valid",
            ],
            config={
                **common,
                "backend": "FastVideo",
                "model": "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
                "checkpoint_step": 1300,
                "attention": "VIDEO_SPARSE_ATTN_H3",
                "vsa_tile_size": 64,
                "vsa_sparsity": 0.9,
                "vsa_kernel": "triton",
                "scheduler_points": 5,
                "denoiser_forwards": 4,
                "video_shift": 12.0,
                "audio_shift": 3.0,
            },
            invocation={"command": "python examples/inference/basic/basic_fasth3.py --model-path FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree --steps 5 --num-gpus 1 --vsa-tile-size 64 --vsa-kernel triton --vsa-sparsity 0.9", "setup_required": True, "blocked": False},
            sources=[
                {
                    "name": "FastH3 VSA model card",
                    "revision": "5ea076f35b84da4c3c82217112fa733d8eea2ae1",
                },
                {
                    "name": "FastVideo VSA kernel README",
                    "revision": "48a047c05ff4138f20cfa33351499c6ec5945f5d",
                    "architecture": "tile64 Triton fallback on other CUDA GPUs; optional sm100a route is SM100/SM103 only",
                },
            ],
        )
    )

    return {
        "schema_version": 1,
        "registry": REGISTRY_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "candidate_status_and_proposed_invocations_not_experiment_evidence",
        "immutable_inputs": [
            "execution_plan.pdf",
            "data/stage1/vbench_dev_manifest.json",
            "data/stage1/source_audit.json",
        ],
        "audit_evidence": [
            {
                "artifact": ".local/artifacts/stage1-family-registry/public-header-audit/comfy-org-tree-015.json",
                "source": "Comfy-Org MiniMax-H3 tree",
                "revision": "a98869194787969724c7425d95d0ed73ce9202af",
                "sha256": "12e846693057db59893aea75cae8801289a123238021c1d4c46eeb601f45f41d",
                "finding": "source tree names qwen3vl_32b_minimax_h3_bf16.safetensors, minimax_h3_fl2va_bf16.safetensors, minimax_h3_video_vae_fp16.safetensors and minimax_h3_audio_vae_fp32.safetensors",
            },
            {
                "artifact": ".local/artifacts/stage1-family-registry/public-header-audit/report.json",
                "source": "Winnougan MMH3-FLA-TurboV1.2-int8-convrot-simple.safetensors",
                "header_bytes": 172704,
                "header_sha256": "2a4c961e900f694859c9dff54872282dbbec33b313b631ce61a4878b26e7b9b4",
                "tensor_bytes_inspected": 0,
                "finding": "embedded pruned_bf16 -> Turbo LoRA -> ModelSave graph; merge/rebuild-AdaLN/quantize provenance remains unverified",
            },
        ],
        "families_in_scope": [
            "comfyui-turbo",
            "sglang-h3",
            "fast-h3-dense",
            "fast-h3-vsa",
        ],
        "candidates": candidates,
        "global_gates": {
            "q0": "original BF16 reference is absent; no quality screening may start",
            "compressed_turbo": "merge adapter -> rebuild AdaLN -> quantize -> verify; header evidence is not proof",
            "vsa_sm89": "requires explicit user disposition or faithful implementation",
            "one_gpu_job": True,
        },
    }
