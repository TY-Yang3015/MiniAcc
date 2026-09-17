"""Reviewed bootstrap for bounded MiniAcc ComfyUI runs.

This is the only code injected before ComfyUI. It configures a native PyTorch
allocator ceiling before Comfy imports can allocate CUDA memory and wraps the
pinned MiniMaxH3Model.forward method with raw JSON lifecycle markers. It also
retains finite-checked joint latents before decoding, without changing sampler output.
"""

from __future__ import annotations

import argparse
import inspect
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time


_LOG_LOCK = threading.Lock()


def _emit(event: str, **fields):
    record = {
        "event": event,
        "run_id": os.environ.get("MINIACC_RUN_ID", "unknown"),
        "job_id": os.environ.get("MINIACC_JOB_ID", "unknown"),
        "timestamp_monotonic": time.monotonic(),
        **fields,
    }
    with _LOG_LOCK:
        # Start a new line even when tqdm has left a progress bar unterminated.
        print("\n" + json.dumps(record, sort_keys=True, default=str), flush=True)


def _timestep_values(value):
    try:
        return value.detach().to(device="cpu").reshape(-1).tolist()
    except (AttributeError, RuntimeError, TypeError):
        return None


def derive_allocator_ceiling(requested_gib: float, free_bytes: int, headroom_bytes: int) -> float:
    """Bound a requested cap by current free memory while retaining headroom."""
    if requested_gib <= 0 or free_bytes <= headroom_bytes:
        raise RuntimeError("insufficient current free memory for requested profile headroom")
    return min(float(requested_gib), (free_bytes - headroom_bytes) / 1024**3)


def configure_allocator(ceiling_gib: float = 16.0, runtime_profile: str = "default"):
    """Set a native allocator ceiling with a profile-specific safety contract."""
    if runtime_profile not in {"default", "a100-independent", "a100-fsdp-experimental"}:
        raise RuntimeError(f"unknown runtime profile: {runtime_profile}")
    if ceiling_gib <= 0:
        raise RuntimeError("allocator ceiling must be positive")
    if runtime_profile == "default" and ceiling_gib > 18.0:
        raise RuntimeError("default allocator ceiling must be <=18 GiB")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native"
    try:
        import torch
    except Exception as error:
        raise RuntimeError(f"PyTorch import failed before allocator setup: {error}") from error
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "set_per_process_memory_fraction"):
        raise RuntimeError("CUDA allocator ceiling cannot be enforced on this runtime")
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("no CUDA device available for allocator setup")
    total = int(torch.cuda.get_device_properties(0).total_memory)
    # Older test doubles and supported CPU-adjacent launchers may not expose
    # mem_get_info; real CUDA profiles always use the current free reading.
    free, _ = torch.cuda.mem_get_info(0) if hasattr(torch.cuda, "mem_get_info") else (total, 0)
    headroom = 8 * 1024**3 if runtime_profile in {"a100-independent", "a100-fsdp-experimental"} else 2 * 1024**3
    derived_ceiling = derive_allocator_ceiling(ceiling_gib, free, headroom)
    fraction = (derived_ceiling * 1024**3) / total
    if fraction <= 0 or fraction >= 1:
        raise RuntimeError("allocator ceiling is not below device capacity")
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    getter = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    if getter is None or abs(float(getter(0)) - fraction) > 1e-6:
        raise RuntimeError("PyTorch did not report the requested allocator ceiling")
    backend_getter = getattr(torch.cuda, "get_allocator_backend", None)
    backend = backend_getter() if backend_getter is not None else "unknown"
    if backend != "native":
        raise RuntimeError(f"native CUDA allocator required; reported backend: {backend}")
    _emit("miniacc_allocator_configured", allocator_backend=backend,
          requested_ceiling_gib=ceiling_gib, ceiling_gib=derived_ceiling,
          ceiling_bytes=int(derived_ceiling * 1024**3), total_device_bytes=total,
          free_device_bytes=free, headroom_gib=headroom / 1024**3,
          runtime_profile=runtime_profile, fraction=fraction)
    return torch


def install_forward_markers():
    """Wrap exactly MiniMaxH3Model.forward without changing arguments/results."""
    from comfy.ldm.minimax.model import MiniMaxH3Model

    if getattr(MiniMaxH3Model.forward, "_miniacc_wrapped", False):
        return
    original = MiniMaxH3Model.forward

    def traced(self, *args, **kwargs):
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        _emit("miniacc_dit_forward_start", model="MiniMaxH3Model",
              model_input_timestep=_timestep_values(timestep))
        try:
            result = original(self, *args, **kwargs)
        except BaseException as error:
            _emit("miniacc_dit_forward_error", model="MiniMaxH3Model",
                  error_type=type(error).__name__, error=str(error)[:500])
            raise
        _emit("miniacc_dit_forward_end", model="MiniMaxH3Model",
              model_input_timestep=_timestep_values(timestep))
        return result

    traced._miniacc_wrapped = True
    MiniMaxH3Model.forward = traced


def save_av_latent(samples, path: Path):
    """Retain the sampler's exact AV output before either decoder can fail."""
    import torch
    from safetensors.torch import save

    if not samples.is_nested:
        raise RuntimeError("expected a joint H3 AV latent for checkpointing")
    streams = samples.unbind()
    if len(streams) != 2:
        raise RuntimeError("expected exactly video and audio latent streams")
    tensors = {name: value.detach().cpu().contiguous()
               for name, value in zip(("video", "audio"), streams)}
    finite = {name: bool(torch.isfinite(value).all().item()) for name, value in tensors.items()}
    payload = save(tensors, metadata={
        "run_id": os.environ.get("MINIACC_RUN_ID", "unknown"),
        "job_id": os.environ.get("MINIACC_JOB_ID", "unknown"),
        "role": "SamplerCustomAdvanced output[0], before AV decoding",
    })
    with path.open("xb") as stream:
        stream.write(payload)
    _emit("miniacc_av_latent_checkpoint", path=str(path),
          sha256=hashlib.sha256(payload).hexdigest(), finite=finite,
          tensor_shapes={name: list(value.shape) for name, value in tensors.items()})
    if not all(finite.values()):
        raise RuntimeError("non-finite denoised AV latent; retained for diagnosis")


def _install_observer(cls, method_name, event_name, callback):
    """Wrap a registered V3/V1 method while retaining exact output and aliases."""
    descriptor = inspect.getattr_static(cls, method_name, None)
    is_classmethod = isinstance(descriptor, classmethod)
    original = descriptor.__func__ if is_classmethod else descriptor
    if original is None or not callable(original):
        raise RuntimeError(f"registered {cls.__name__} has no callable {method_name}")
    if getattr(original, "_miniacc_observer", False):
        return original

    @wraps(original)
    def traced(receiver, *args, **kwargs):
        node_cls = receiver if is_classmethod else type(receiver)
        identity = f"{node_cls.__module__}.{node_cls.__name__}"
        _emit(event_name + "_start", node_class=identity)
        try:
            result = original(receiver, *args, **kwargs)
        except BaseException as error:
            _emit(event_name + "_error", node_class=identity,
                  error_type=type(error).__name__, error=str(error)[:500])
            raise
        _emit(event_name + "_end", node_class=identity)
        callback(result)
        return result

    traced._miniacc_observer = True
    setattr(cls, method_name, classmethod(traced) if is_classmethod else traced)
    # V3 nodes retain aliases such as sample/decode. Preserve an exact alias when
    # it pointed at the original function; do not alter unrelated methods.
    for alias in ("sample", "decode"):
        alias_descriptor = inspect.getattr_static(cls, alias, None)
        alias_original = (alias_descriptor.__func__ if isinstance(alias_descriptor, classmethod)
                          else alias_descriptor)
        if alias_original is original:
            setattr(cls, alias, classmethod(traced) if is_classmethod else traced)
    return traced


def install_latent_checkpoint(registered_class=None):
    """Install on the class Comfy actually registered, not an ordinary import."""
    checkpoint = os.environ.get("MINIACC_LATENT_CHECKPOINT")
    if not checkpoint:
        return None
    if registered_class is None:
        import nodes
        registered_class = nodes.NODE_CLASS_MAPPINGS.get("SamplerCustomAdvanced")
    if registered_class is None:
        raise RuntimeError("registered SamplerCustomAdvanced class is unavailable")

    def checkpoint_result(result):
        # NodeOutput[0] is intentionally retained exactly; the observer returns
        # the same object and therefore cannot perturb RNG state or graph output.
        save_av_latent(result[0]["samples"], Path(checkpoint))

    _install_observer(registered_class, "execute", "miniacc_sampler", checkpoint_result)
    return registered_class


def _install_timing_observer(cls, method_name, phase, node_id):
    """Observe one native node call without changing its arguments or result."""
    descriptor = inspect.getattr_static(cls, method_name, None)
    is_classmethod = isinstance(descriptor, classmethod)
    original = descriptor.__func__ if is_classmethod else descriptor
    if original is None or not callable(original):
        raise RuntimeError(f"timing target {cls.__name__}.{method_name} is not callable")
    installed = getattr(original, "_miniacc_timing_phases", set())
    if phase in installed:
        return original

    @wraps(original)
    def traced(receiver, *args, **kwargs):
        node_cls = receiver if is_classmethod else type(receiver)
        identity = f"{node_cls.__module__}.{node_cls.__name__}"
        fields = {"phase": phase, "node_id": node_id, "node_class": identity}
        _emit("miniacc_timing_start", **fields)
        try:
            result = original(receiver, *args, **kwargs)
        except BaseException as error:
            _emit("miniacc_timing_error", **fields,
                  error_type=type(error).__name__, error=str(error)[:500])
            raise
        _emit("miniacc_timing_end", **fields)
        return result

    traced._miniacc_timing_phases = set(installed) | {phase}
    setattr(cls, method_name, classmethod(traced) if is_classmethod else traced)
    return traced


def install_timing_observers():
    """Install optional phase observers on the registered native Comfy nodes."""
    import nodes

    targets = (
        # These loader calls are lazy in Comfy and therefore separately expose
        # first-request model preparation without claiming pure disk-load time.
        ("UNETLoader", "load_unet", "initial_model_loading", "UNETLoader"),
        ("CLIPLoader", "load_clip", "initial_model_loading", "CLIPLoader"),
        ("VAELoader", "load_vae", "initial_model_loading", "VAELoader"),
        ("LoraLoaderModelOnly", "load_lora_model_only", "initial_model_loading", "LoraLoaderModelOnly"),
        ("MiniMaxH3ImageToVideo", "execute", "prompt_conditioning", "MiniMaxH3ImageToVideo"),
        ("SamplerCustomAdvanced", "execute", "denoising", "SamplerCustomAdvanced"),
        ("VAEDecodeAudio", "execute", "audio_decode", "VAEDecodeAudio"),
        ("VAEDecodeTiled", "decode", "video_decode", "VAEDecodeTiled"),
        # CreateVideo only wraps decoded components; SaveVideo's native
        # Video.save_to call performs the actual mux and file materialization.
        ("CreateVideo", "execute", "video_assembly", "CreateVideo"),
        ("SaveVideo", "execute", "mux_save", "SaveVideo"),
    )
    installed = []
    for class_name, method_name, phase, node_id in targets:
        cls = nodes.NODE_CLASS_MAPPINGS.get(class_name)
        if cls is None:
            raise RuntimeError(f"registered timing node is unavailable: {class_name}")
        _install_timing_observer(cls, method_name, phase, node_id)
        installed.append({"node_id": node_id, "phase": phase, "method": method_name})
    _emit("miniacc_timing_hook_ready", targets=installed,
          instrumentation="observation_only_native_node_boundaries",
          mux_save_boundary="SaveVideo.execute -> Video.save_to")
    return installed


def install_registered_hooks():
    """Install optional debug checkpoint hooks and the separate timing observer."""
    import nodes

    if os.environ.get("MINIACC_TIMING") == "1":
        install_timing_observers()

    sampler = install_latent_checkpoint(nodes.NODE_CLASS_MAPPINGS.get("SamplerCustomAdvanced"))
    if sampler is None:
        return None

    for node_id, method_name, event_name in (
        ("VAEDecodeAudio", "execute", "miniacc_audio_decode"),
        ("VAEDecodeTiled", "decode", "miniacc_video_decode"),
    ):
        decoder = nodes.NODE_CLASS_MAPPINGS.get(node_id)
        if decoder is None:
            raise RuntimeError(f"registered decoder class is unavailable: {node_id}")
        _install_observer(decoder, method_name, event_name, lambda result: None)

    identity = f"{sampler.__module__}.{sampler.__name__}"
    _emit("miniacc_checkpoint_hook_ready", registered_class=identity,
          checkpoint_path=os.environ["MINIACC_LATENT_CHECKPOINT"])
    return sampler


def wrap_extra_node_initialization():
    """Install hooks only after Comfy's dynamic registration coroutine completes."""
    import nodes
    original = nodes.init_extra_nodes
    if getattr(original, "_miniacc_registration_wrapper", False):
        return

    @wraps(original)
    async def traced(*args, **kwargs):
        result = await original(*args, **kwargs)
        install_registered_hooks()
        return result

    traced._miniacc_registration_wrapper = True
    nodes.init_extra_nodes = traced


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-main", type=Path, required=True)
    parser.add_argument("--allocator-gib", type=float, default=16.0)
    parser.add_argument("--runtime-profile", choices=("default", "a100-independent", "a100-fsdp-experimental"), default="default")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--file-backed-dit",
        action="store_true",
        help="Enable the exact source-verified H3 file-backed DiT policy",
    )
    parser.add_argument(
        "--file-backed-text", action="store_true",
        help="Use mapped original-BF16 MiniMax H3 text weights on CPU",
    )
    args, comfy_args = parser.parse_known_args(argv)
    if args.cpu_threads < 1 or args.cpu_threads > 24:
        raise SystemExit("--cpu-threads must be between 1 and 24")
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    comfy_main = args.comfy_main.resolve()
    if not comfy_main.is_file():
        raise SystemExit(f"ComfyUI source entrypoint does not exist: {comfy_main}")
    sys.path.insert(0, str(comfy_main.parent))
    os.chdir(comfy_main.parent)
    # Comfy caches parsed arguments at import time. Enable parsing with only
    # Comfy's arguments before the forward hook imports model_management.
    if args.file_backed_dit or args.file_backed_text:
        # This is an owned-mode contract, not a general Comfy default.  The
        # flag is passed to Comfy itself and checked after its real parser runs.
        if "--disable-pinned-memory" not in comfy_args:
            comfy_args.append("--disable-pinned-memory")
        os.environ["MINIACC_FILE_BACKED_DIT"] = "1"
    sys.argv = [str(comfy_main), *comfy_args]
    import comfy.options
    comfy.options.enable_args_parsing()
    configure_allocator(args.allocator_gib, args.runtime_profile)
    if args.file_backed_dit:
        from miniacc_core.file_backed import install_file_backed_dit_loader
        import comfy.cli_args
        if not getattr(comfy.cli_args, "args", None).disable_pinned_memory:
            raise RuntimeError("Comfy did not consume --disable-pinned-memory")
        install_file_backed_dit_loader(comfy.cli_args.args)
        _emit(
            "miniacc_file_backed_dit_configured",
            disable_pinned_memory=True,
            dynamic_vram_disabled=bool(getattr(comfy.cli_args.args, "disable_dynamic_vram", False)),
        )
    if args.file_backed_text:
        from miniacc_core.file_backed import install_file_backed_text_loader
        import comfy.cli_args
        install_file_backed_text_loader(comfy.cli_args.args)
        _emit("miniacc_file_backed_text_configured", dtype="BF16", device="cpu")
    install_forward_markers()
    # nodes.init_extra_nodes dynamically registers V3 classes under absolute
    # module names; only its post-registration registry contains the runtime class.
    wrap_extra_node_initialization()
    runpy.run_path(str(comfy_main), run_name="__main__")


if __name__ == "__main__":
    main()
