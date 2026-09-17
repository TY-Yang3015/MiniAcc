"""Immutable contracts shared by preparation, runtimes, and evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
import json

DEVELOPMENT_DIMENSIONS: Final = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
)
STRATA: Final = ("people", "motion_objects", "scenes", "complex")
SEEDS: Final = (20260909, 20260910)
SELECTION_NAMESPACE: Final = "miniacc-vbench-dev-v1"

# The uncompressed, full BF16 route is the only Turbo base admitted by the
# implemented Comfy adapter.  Compressed exports require a separately verified
# merge/rebuild/quantize artifact and therefore cannot enter this path.
COMFY_TURBO_DIFFUSION: Final = "minimax_h3_fl2va_bf16.safetensors"
COMFY_TURBO_CLIP: Final = "qwen3vl_32b_minimax_h3_bf16.safetensors"
COMFY_NVFP4_CLIP: Final = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
COMFY_TURBO_ADAPTER: Final = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
COMFY_TURBO_ADAPTER_8STEP: Final = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
COMFY_TURBO_VIDEO_VAE: Final = "minimax_h3_video_vae_fp16.safetensors"
COMFY_TURBO_AUDIO_VAE: Final = "minimax_h3_audio_vae_fp32.safetensors"
HISTORICAL_BASE_DIFFUSION: Final = (
    "minimax_h3_fl2va_pruned-w4a8_convrot_pruned.safetensors"
)


@dataclass(frozen=True)
class WorkloadConfig:
    width: int = 1344
    height: int = 768
    frames: int = 124
    fps: int = 24
    batch_size: int = 1
    audio_rate: int = 32000
    audio_channels: int = 2
    seeds: tuple[int, ...] = SEEDS

    def media_expectation(self) -> dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "fps": self.fps,
            "audio_channels": self.audio_channels,
            "audio_rate": self.audio_rate,
        }


@dataclass(frozen=True)
class CandidateConfig:
    """Runtime candidate and its explicit asset/schedule provenance."""

    family: str = "comfyui-base"
    diffusion_name: str | None = None
    clip_name: str | None = None
    video_vae_name: str = COMFY_TURBO_VIDEO_VAE
    audio_vae_name: str = COMFY_TURBO_AUDIO_VAE
    adapter_name: str | None = None
    adapter_scale: float = 1.0
    adapter_alpha: float | None = None
    steps: int = 49
    video_shift: float = 12.0
    audio_shift: float = 3.0
    file_backed_dit: bool = False
    capabilities: frozenset[str] = frozenset()

    def __post_init__(self):
        # A family-specific default must not accidentally inherit the historical
        # pruned BASE.  Explicitly supplied compressed names still fail closed.
        if self.family == "comfyui-turbo":
            if self.diffusion_name is None:
                object.__setattr__(self, "diffusion_name", COMFY_TURBO_DIFFUSION)
            if self.clip_name is None:
                object.__setattr__(self, "clip_name", COMFY_TURBO_CLIP)
        else:
            if self.diffusion_name is None:
                object.__setattr__(self, "diffusion_name", HISTORICAL_BASE_DIFFUSION)
            if self.clip_name is None:
                object.__setattr__(
                    self, "clip_name", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
                )
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.adapter_scale < 0:
            raise ValueError("adapter_scale must not be negative")
        if self.adapter_alpha is not None:
            raise ValueError("adapter_alpha is unsupported by the Comfy adapter")
        if self.file_backed_dit:
            turbo_control = (
                self.family == "comfyui-turbo"
                and self.diffusion_name == COMFY_TURBO_DIFFUSION
                and self.clip_name in {COMFY_NVFP4_CLIP, COMFY_TURBO_CLIP}
                and self.adapter_name in {COMFY_TURBO_ADAPTER, COMFY_TURBO_ADAPTER_8STEP}
                and self.steps in {4, 8}
                and ((self.steps == 4 and self.adapter_name == COMFY_TURBO_ADAPTER)
                     or (self.steps == 8 and self.adapter_name == COMFY_TURBO_ADAPTER_8STEP))
                and self.video_shift == 6.0
                and self.audio_shift == 3.0
            )
            if not (self.is_bf16_reference or turbo_control):
                raise ValueError(
                    "file-backed DiT policy is restricted to the exact BF16 reference or four-step controls"
                )
        if self.family == "comfyui-base" and self.adapter_name:
            raise ValueError("comfyui-base cannot be combined with an adapter")
        if self.family == "comfyui-turbo" and self.adapter_name:
            if self.diffusion_name != COMFY_TURBO_DIFFUSION:
                raise ValueError(
                    "Turbo adapters require the source-backed full BF16 Comfy base; "
                    "compressed/pruned pairings are blocked"
                )
        if not self.capabilities:
            capabilities = {"t2va", "native_av", "comfy_graph"}
            if self.family == "comfyui-turbo":
                capabilities.add("turbo_lora")
            object.__setattr__(self, "capabilities", frozenset(capabilities))

    @property
    def is_bf16_reference(self) -> bool:
        """Original-weight/schedule reference, never the quantized BASE default."""
        return (
            self.family == "comfyui-base"
            and self.diffusion_name == COMFY_TURBO_DIFFUSION
            and self.clip_name == COMFY_TURBO_CLIP
            and self.video_vae_name == COMFY_TURBO_VIDEO_VAE
            and self.audio_vae_name == COMFY_TURBO_AUDIO_VAE
            and self.adapter_name is None
            and self.steps == 49
            and self.video_shift == 12.0
            and self.audio_shift == 3.0
        )


@dataclass(frozen=True)
class HardwareConfig:
    """Consumed per-hardware runtime and parallelism contract.

    Config files are intentionally data-only.  A distributed backend is an
    explicit gate; selecting it does not silently alter the single-GPU path.
    """

    name: str
    gpu_count: int
    per_gpu_memory_gib: float
    allocator_gib: float
    vram_headroom_gib: float
    vram_mode: str = "novram"
    parallel_backend: str = "none"
    world_size: int = 1
    runtime_profile: str = "default"
    host_headroom_gib: float = 8.0
    conditioning_mode: str = "concurrent"

    def __post_init__(self):
        if not self.name or self.gpu_count < 1 or self.world_size < 1:
            raise ValueError("hardware identity and counts must be positive")
        if (self.per_gpu_memory_gib <= 0 or self.allocator_gib <= 0
                or self.vram_headroom_gib <= 0 or self.host_headroom_gib <= 0):
            raise ValueError("hardware memory values must be positive")
        if self.conditioning_mode not in {"concurrent", "serialized_first_forward"}:
            raise ValueError("unknown conditioning mode")
        if self.allocator_gib + self.vram_headroom_gib > self.per_gpu_memory_gib:
            raise ValueError("allocator plus headroom must fit per-GPU memory")
        if self.vram_mode not in {"novram", "normal"}:
            raise ValueError("vram_mode must be novram or normal")
        if self.parallel_backend not in {"none", "independent_jobs", "fsdp_inference_experimental"}:
            raise ValueError("unknown parallel backend")
        if self.parallel_backend == "none" and self.world_size != 1:
            raise ValueError("world_size must be one when parallel backend is none")
        if self.parallel_backend == "independent_jobs" and self.world_size < 2:
            raise ValueError("independent_jobs requires at least two ranks")
        if self.parallel_backend != "none" and self.world_size > self.gpu_count:
            raise ValueError("world_size cannot exceed configured GPU count")
        if self.runtime_profile not in {"default", "a100-independent"}:
            raise ValueError("unknown runtime profile")


def load_hardware_config(path: Path) -> HardwareConfig:
    """Load a small JSON-compatible YAML hardware profile without new deps."""
    try:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid hardware config {path}: {error}") from error
    if not isinstance(values, dict):
        raise ValueError("hardware config must be an object")
    try:
        return HardwareConfig(**values)
    except TypeError as error:
        raise ValueError(f"invalid hardware config fields: {error}") from error


@dataclass(frozen=True)
class HostConfig:
    gpu_index: int = 0
    allocator_gib: float = 16.0
    timeout_seconds: float = 3600.0
    cpu_threads: int = 8
    vae_device: str = "cpu"
    min_vram_reserve_mib: int = 2048
    min_host_headroom_bytes: int = 8 * 1024**3
    min_project_free_bytes: int = 100 * 1024**3
    min_executable_free_bytes: int = 15 * 1024**3
    runtime_profile: str = "default"

    def __post_init__(self):
        if self.vae_device not in {"cpu", "gpu"}:
            raise ValueError("vae_device must be cpu or gpu")
        if self.allocator_gib <= 0 or self.timeout_seconds <= 0 or self.cpu_threads < 1:
            raise ValueError("host limits must be positive")
        if self.runtime_profile not in {"default", "a100-independent"}:
            raise ValueError("unknown runtime profile")


@dataclass(frozen=True)
class RunConfig:
    manifest: Path
    run_root: Path
    output: Path | None = None
    comfy_output: Path | None = None
