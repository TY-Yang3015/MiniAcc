"""Model-manager contracts and auditable ComfyUI adapters.

Managers are inert at construction. Unsupported families remain explicit rather
than being silently routed to the BASE graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import COMFY_TURBO_ADAPTER, COMFY_TURBO_ADAPTER_8STEP, COMFY_TURBO_DIFFUSION, CandidateConfig, WorkloadConfig


class UnsupportedCapabilityError(RuntimeError):
    """Raised when a requested family or option has no verified implementation."""


@dataclass(frozen=True)
class CandidateReadiness:
    family: str
    implementation_status: str
    available_assets: tuple[str, ...]
    missing_assets: tuple[str, ...]
    compatibility: str
    note: str


class CandidateCatalog:
    """Inspect candidate-specific local assets without importing a model runtime."""

    def __init__(self, model_root: Path):
        self.model_root = Path(model_root)

    def inspect(self, candidate: CandidateConfig | None = None) -> CandidateReadiness:
        candidate = candidate or CandidateConfig()
        if candidate.family not in ModelManagerFactory.KNOWN_FAMILIES:
            return CandidateReadiness(
                candidate.family,
                "unknown_family",
                (),
                (),
                "unsupported",
                "Unknown family is never routed to BASE.",
            )
        required = self._required_assets(candidate)
        if required is None:
            return CandidateReadiness(
                candidate.family,
                "runtime_not_installed_or_not_inspected",
                (),
                (f"runtime:{candidate.family}",),
                "unverified",
                "This family has an independent runtime/model store; Comfy BASE assets were not counted.",
            )
        available = tuple(
            f"{folder}/{name}"
            for folder, name in required
            if (self.model_root / folder / name).is_file()
        )
        missing = tuple(
            f"{folder}/{name}"
            for folder, name in required
            if not (self.model_root / folder / name).is_file()
        )
        if candidate.family == "comfyui-base":
            status = "assets_present" if not missing else "assets_missing"
            compatibility = "runtime_check_required"
            note = "Local assets are present; runtime loading and residency remain explicit checks."
        else:
            status = (
                "assets_present_compatibility_unverified"
                if not missing
                else "assets_missing"
            )
            compatibility = "unverified"
            note = "Local assets are present only for candidate-specific inspection; presence is not export or kernel compatibility proof."
        return CandidateReadiness(
            candidate.family, status, available, missing, compatibility, note
        )

    @staticmethod
    def _required_assets(candidate: CandidateConfig) -> list[tuple[str, str]] | None:
        if candidate.family in {"comfyui-base", "comfyui-turbo"}:
            required = [
                ("diffusion_models", candidate.diffusion_name),
                ("text_encoders", candidate.clip_name),
                ("vae", candidate.video_vae_name),
                ("vae", candidate.audio_vae_name),
            ]
            if candidate.family == "comfyui-turbo":
                required.append(
                    (
                        "loras",
                        candidate.adapter_name
                        or "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
                    )
                )
            return required
        return None


def _node(name: str, inputs: dict, class_type: str) -> dict:
    return {"inputs": inputs, "class_type": class_type}


def build_prompt_graph(
    prompt: str,
    seed: int,
    *,
    candidate: CandidateConfig | None = None,
    workload: WorkloadConfig | None = None,
    video_tile_size: int = 512,
    video_tile_overlap: int = 64,
    video_temporal_size: int = 64,
    video_temporal_overlap: int = 8,
) -> dict:
    """Build one source-backed Comfy graph without contacting ComfyUI."""
    candidate = candidate or CandidateConfig()
    workload = workload or WorkloadConfig()
    if candidate.adapter_name and candidate.family != "comfyui-turbo":
        raise ValueError("only comfyui-turbo may attach a Turbo adapter")
    graph = {
        "1": _node(
            "unet",
            {"unet_name": candidate.diffusion_name, "weight_dtype": "default"},
            "UNETLoader",
        ),
        "3": _node(
            "shift",
            {
                "model": ["1", 0],
                "shift_video": candidate.video_shift,
                "shift_audio": candidate.audio_shift,
            },
            "MiniMaxH3SigmaShift",
        ),
        "4": _node(
            "clip",
            {"clip_name": candidate.clip_name, "type": "minimax", "device": "default"},
            "CLIPLoader",
        ),
        "5": _node("video vae", {"vae_name": candidate.video_vae_name}, "VAELoader"),
        "6": _node("audio vae", {"vae_name": candidate.audio_vae_name}, "VAELoader"),
        "7": _node(
            "conditioning",
            {
                "clip": ["4", 0],
                "vae": ["5", 0],
                "prompt": prompt,
                "width": workload.width,
                "height": workload.height,
                "length": workload.frames,
            },
            "MiniMaxH3ImageToVideo",
        ),
        "8": _node(
            "guider", {"model": ["3", 0], "conditioning": ["7", 0]}, "BasicGuider"
        ),
        "9": _node("sampler", {"sampler_name": "euler"}, "KSamplerSelect"),
        "10": _node(
            "schedule",
            {
                "model": ["3", 0],
                "scheduler": "simple",
                "steps": candidate.steps,
                "denoise": 1.0,
            },
            "BasicScheduler",
        ),
        "11": _node("noise", {"noise_seed": seed}, "RandomNoise"),
        "12": _node(
            "sample",
            {
                "noise": ["11", 0],
                "guider": ["8", 0],
                "sampler": ["9", 0],
                "sigmas": ["10", 0],
                "latent_image": ["7", 1],
            },
            "SamplerCustomAdvanced",
        ),
        "13": _node(
            "video decode (tiled)",
            {
                "samples": ["12", 0],
                "vae": ["5", 0],
                "tile_size": video_tile_size,
                "overlap": video_tile_overlap,
                "temporal_size": video_temporal_size,
                "temporal_overlap": video_temporal_overlap,
            },
            "VAEDecodeTiled",
        ),
        "14": _node(
            "audio decode", {"samples": ["12", 0], "vae": ["6", 0]}, "VAEDecodeAudio"
        ),
        "15": _node(
            "mux",
            {"images": ["13", 0], "audio": ["14", 0], "fps": workload.fps},
            "CreateVideo",
        ),
        "16": _node(
            "save",
            {
                "video": ["15", 0],
                "filename_prefix": "video/MiniAcc_H3",
                "format": "auto",
                "codec": "auto",
            },
            "SaveVideo",
        ),
    }
    if candidate.adapter_name:
        graph["2"] = _node(
            "turbo adapter",
            {
                "model": ["1", 0],
                "lora_name": candidate.adapter_name,
                "strength_model": candidate.adapter_scale,
            },
            "LoraLoaderModelOnly",
        )
        graph["3"]["inputs"]["model"] = ["2", 0]
    return graph


def _remaining(deadline: float | None, maximum: float) -> float:
    if deadline is None:
        return maximum
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("absolute deadline expired")
    return min(maximum, remaining)


class ModelManager(ABC):
    """Explicit lifecycle; loading is never performed by ``__init__``."""

    candidate: CandidateConfig
    loaded: bool = False

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[str]: ...

    @abstractmethod
    def load_inference_model(self, candidate=None, *, deadline=None): ...

    @abstractmethod
    def infer(self, prompt: str, seed: int, *, deadline=None) -> dict: ...

    def fit(self, *args, **kwargs):
        raise UnsupportedCapabilityError("training is deferred and unavailable")

    def close(self):
        self.loaded = False


class ComfyBaseManager(ModelManager):
    """Submit a real Comfy graph; process ownership stays with RuntimeOwner."""

    def __init__(
        self, candidate=None, workload=None, base_url="http://127.0.0.1:8188", post=None
    ):
        self.candidate = candidate or CandidateConfig()
        self.workload = workload or WorkloadConfig()
        self.base_url = base_url
        self._post = post or post_json
        self.loaded = False
        self.readiness = "not_checked"

    @property
    def capabilities(self):
        return self.candidate.capabilities

    def _check_candidate(self, candidate):
        if candidate is not None and candidate != self.candidate:
            raise ValueError("manager candidate does not match requested candidate")

    def load_inference_model(self, candidate=None, *, deadline=None):
        self._check_candidate(candidate)
        try:
            self._get("/system_stats", deadline=deadline)
        except Exception as error:
            self.readiness = "unavailable"
            raise RuntimeError(f"ComfyUI runtime is not ready: {error}") from error
        self.readiness = "runtime_ready_lazy_weights"
        self.loaded = True
        return {"readiness": self.readiness, "resident": False}

    def infer(self, prompt, seed, *, deadline=None):
        if not self.loaded:
            raise RuntimeError("load_inference_model must precede infer")
        graph = build_prompt_graph(
            prompt, seed, candidate=self.candidate, workload=self.workload
        )
        response = self._post_request(
            f"{self.base_url.rstrip('/')}/prompt",
            {"client_id": "miniacc_application", "prompt": graph},
            deadline=deadline,
        )
        if not response.get("prompt_id"):
            raise RuntimeError(f"ComfyUI did not return prompt_id: {response}")
        graph_hash = hashlib.sha256(
            json.dumps(graph, sort_keys=True).encode()
        ).hexdigest()
        return {
            "prompt_id": response["prompt_id"],
            "graph": graph,
            "graph_sha256": graph_hash,
        }

    def _post_request(self, url, value, *, deadline=None):
        timeout = _remaining(deadline, 30.0)
        return self._post(url, value, timeout=timeout)

    def wait_until_ready(
        self,
        deadline,
        *,
        guard=None,
        owned=None,
        readiness_log=None,
        expected_run_id=None,
    ):
        while time.monotonic() < deadline:
            if (
                owned is not None
                and getattr(owned, "process", None) is not None
                and owned.process.poll() is not None
            ):
                raise RuntimeError(
                    f"ComfyUI server exited during startup with code {owned.process.poll()}"
                )
            if guard is not None:
                guard.check()
            try:
                self._get("/system_stats", deadline=deadline)
                if readiness_log is not None and not _has_readiness_marker(
                    readiness_log, expected_run_id
                ):
                    # HTTP can come up before the post-registration timing or
                    # checkpoint hook marker is flushed. Keep the same bounded
                    # wait rather than misclassifying that race as a failure.
                    time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
                    continue
                return
            except (HTTPError, URLError, TimeoutError):
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(
            "ComfyUI server did not become ready before the absolute deadline"
        )

    def wait_for_history(self, prompt_id, timeout, *, guard=None, deadline=None):
        end = (
            min(time.monotonic() + timeout, deadline)
            if deadline is not None
            else time.monotonic() + timeout
        )
        while time.monotonic() < end:
            if guard is not None:
                guard.check()
            try:
                with urlopen(
                    f"{self.base_url.rstrip('/')}/history/{prompt_id}",
                    timeout=_remaining(end, 30.0),
                ) as response:
                    history = json.loads(response.read())
                if prompt_id in history:
                    return history[prompt_id]
            except (HTTPError, URLError, TimeoutError):
                pass
            time.sleep(min(2.0, max(0.0, end - time.monotonic())))
        raise TimeoutError(
            f"ComfyUI history did not complete within {timeout:g}s: {prompt_id}"
        )

    def _get(self, path, *, deadline=None):
        with urlopen(
            self.base_url.rstrip("/") + path, timeout=_remaining(deadline, 5.0)
        ) as response:
            return json.loads(response.read())


class ComfyTurboManager(ComfyBaseManager):
    """Real Comfy LoRA graph adapter for the full BF16 control only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.candidate.adapter_name:
            raise ValueError("comfyui-turbo requires an explicit adapter_name")
        if self.candidate.diffusion_name != COMFY_TURBO_DIFFUSION:
            raise ValueError("compressed/pruned Turbo base pairings are blocked")
        allowed = {
            (COMFY_TURBO_ADAPTER, 4),
            (COMFY_TURBO_ADAPTER_8STEP, 8),
        }
        if (self.candidate.adapter_name, self.candidate.steps) not in allowed:
            raise ValueError("Turbo adapter and configured steps must match the released 4- or 8-step pair")


def _has_readiness_marker(path, expected_run_id):
    for line in Path(path).read_text(errors="replace").splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            event = json.JSONDecoder().raw_decode(line[start:])[0]
        except json.JSONDecodeError:
            continue
        if event.get("event") in {"miniacc_checkpoint_hook_ready", "miniacc_timing_hook_ready"} and (
            expected_run_id is None or event.get("run_id") == expected_run_id
        ):
            return True
    return False


class ModelManagerFactory:
    SUPPORTED = {"comfyui-base": ComfyBaseManager, "comfyui-turbo": ComfyTurboManager}
    KNOWN_FAMILIES = frozenset(
        {"comfyui-base", "comfyui-turbo", "sglang-h3", "fast-h3-dense", "fast-h3-vsa"}
    )

    def create(self, candidate=None, **kwargs):
        candidate = candidate or CandidateConfig()
        manager = self.SUPPORTED.get(candidate.family)
        if manager is None:
            if candidate.family not in self.KNOWN_FAMILIES:
                raise UnsupportedCapabilityError(
                    f"unknown model family: {candidate.family}"
                )
            raise UnsupportedCapabilityError(
                f"model family is audited but not implemented: {candidate.family}"
            )
        return manager(candidate=candidate, **kwargs)


def post_json(url, value, timeout=30.0):
    request = Request(
        url,
        data=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())
