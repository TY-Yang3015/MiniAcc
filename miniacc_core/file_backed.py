"""Opt-in, checkpoint-scoped Comfy H3 loading policy.

This module is imported only by the reviewed bootstrap inside ComfyUI.  It uses
Comfy's existing safetensors loader/storage and only changes construction and
assignment for the full BF16 DiT and opt-in original H3 text encoder. It is not a
general torch or nn monkeypatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import (
    COMFY_NVFP4_CLIP,
    COMFY_TURBO_ADAPTER,
    COMFY_TURBO_DIFFUSION,
    COMFY_TURBO_CLIP,
)

DIFFUSION_FILE = COMFY_TURBO_DIFFUSION
DIFFUSION_BYTES = 66280487368
SAFE_TENSOR_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
}
META_THRESHOLD_BYTES = 1024 * 1024


def _dtype_size(dtype: torch.dtype | None) -> int:
    if dtype is None:
        return 4
    return torch.empty((), dtype=dtype).element_size()


def _validate_module_state(module, state_dict, prefix, local_metadata):
    """Validate one H3 operation module before Comfy's normal load logic."""
    expected = {
        name: parameter
        for name, parameter in module._parameters.items()
        if parameter is not None
    }
    expected.update(
        {
            name: buffer
            for name, buffer in module._buffers.items()
            if buffer is not None and name not in module._non_persistent_buffers_set
        }
    )
    for name, target in expected.items():
        key = prefix + name
        if key not in state_dict:
            raise RuntimeError(f"file-backed H3 state is missing {key}")
        source = state_dict[key]
        if source.device.type != "cpu":
            raise RuntimeError(f"file-backed H3 source is not CPU: {key}")
        if tuple(source.shape) != tuple(target.shape):
            raise RuntimeError(
                f"file-backed H3 shape mismatch for {key}: "
                f"{tuple(source.shape)} != {tuple(target.shape)}"
            )
        if source.dtype != target.dtype:
            raise RuntimeError(
                f"file-backed H3 dtype mismatch for {key}: "
                f"{source.dtype} != {target.dtype}"
            )
        if not source.is_contiguous():
            raise RuntimeError(f"file-backed H3 source layout is not contiguous: {key}")
    metadata = dict(local_metadata)
    # PyTorch consumes this flag inside Module._load_from_state_dict.  Only
    # meta parameters get assignment; ordinary small parameters retain default
    # copy behavior and their existing operation/forward semantics.
    if any(parameter is not None and parameter.is_meta for parameter in module._parameters.values()):
        metadata["assign_to_params_buffers"] = True
    return metadata


def operations_for_h3(base_operations: Any):
    """Return Comfy operations with meta-only expensive H3 Linear weights."""

    class CheckedLinear(base_operations.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            bytes_required = in_features * out_features * _dtype_size(dtype)
            target_device = torch.device("meta") if bytes_required >= META_THRESHOLD_BYTES else device
            super().__init__(in_features, out_features, bias, device=target_device, dtype=dtype)

        def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            metadata = _validate_module_state(self, state_dict, prefix, local_metadata)
            return super()._load_from_state_dict(
                state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

    class CheckedRMSNorm(base_operations.RMSNorm):
        def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            metadata = _validate_module_state(self, state_dict, prefix, local_metadata)
            return super()._load_from_state_dict(
                state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

    class H3Operations(base_operations):
        Linear = CheckedLinear
        RMSNorm = CheckedRMSNorm
        _miniacc_file_backed_operations = True

    return H3Operations


def install_file_backed_dit_loader(comfy_args, *, expected_size: int = DIFFUSION_BYTES) -> None:
    """Install a narrow loader wrapper after Comfy parsed its real arguments."""
    if not getattr(comfy_args, "disable_pinned_memory", False):
        raise RuntimeError("file-backed DiT requires Comfy --disable-pinned-memory")
    if getattr(comfy_args, "disable_mmap", False):
        raise RuntimeError("file-backed DiT requires mmap; --disable-mmap is incompatible")
    if not getattr(comfy_args, "disable_dynamic_vram", False):
        raise RuntimeError("file-backed DiT requires DynamicVRAM disabled")

    import comfy.model_management
    import comfy.ops
    import comfy.sd
    from safetensors import safe_open

    if not comfy.model_management.is_device_cpu(comfy.model_management.unet_offload_device()):
        raise RuntimeError("file-backed DiT requires a CPU initial offload device")

    original = comfy.sd.load_diffusion_model
    if getattr(original, "_miniacc_file_backed", False):
        return

    def call_original(path, model_options, disable_dynamic):
        kwargs = {"disable_dynamic": disable_dynamic}
        if model_options is not None:
            kwargs["model_options"] = model_options
        return original(str(path), **kwargs)

    def load(path, model_options=None, disable_dynamic=False):
        path = Path(path).resolve()
        if path.name != DIFFUSION_FILE:
            return call_original(path, model_options, disable_dynamic)
        if path.stat().st_size != expected_size:
            raise RuntimeError("file-backed DiT source size is not the pinned full BF16 asset")
        # Comfy's disable_dynamic=True is the explicit legacy patcher request;
        # the owned command disables DynamicVRAM globally, so both values are
        # valid here.  The installer guard above remains authoritative.
        options = dict(model_options or {})
        if options.get("dtype") not in (None, torch.bfloat16):
            raise RuntimeError("file-backed DiT refuses an incompatible dtype override")
        if "offload_device" in options:
            try:
                offload_device = torch.device(options["offload_device"])
            except (TypeError, RuntimeError) as error:
                raise RuntimeError("file-backed DiT received an invalid offload_device") from error
            if not comfy.model_management.is_device_cpu(offload_device):
                raise RuntimeError("file-backed DiT requires a CPU offload_device")
        existing_operations = options.get("custom_operations")
        if existing_operations is not None and not getattr(existing_operations, "_miniacc_file_backed_operations", False):
            raise RuntimeError("file-backed DiT refuses incompatible custom_operations")
        if existing_operations is None:
            options["custom_operations"] = operations_for_h3(comfy.ops.disable_weight_init)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            source_schema = {}
            for key in handle.keys():
                tensor_slice = handle.get_slice(key)
                dtype_name = tensor_slice.get_dtype()
                try:
                    dtype = SAFE_TENSOR_DTYPES[dtype_name]
                except KeyError as error:
                    raise RuntimeError(f"file-backed DiT source dtype is unsupported: {key}: {dtype_name}") from error
                source_schema[key] = (torch.Size(tensor_slice.get_shape()), dtype)
        result = call_original(path, options, disable_dynamic)
        if result is None or not hasattr(result, "model"):
            raise RuntimeError("file-backed DiT loader returned no model patcher")
        actual = result.model.diffusion_model.state_dict()
        actual_keys = set(actual)
        source_keys = set(source_schema)
        if actual_keys != source_keys:
            missing = sorted(source_keys - actual_keys)
            unexpected = sorted(actual_keys - source_keys)
            raise RuntimeError(
                "file-backed DiT source/model key set mismatch: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        residual_meta = [key for key, value in actual.items() if value.is_meta]
        if residual_meta:
            raise RuntimeError(f"file-backed DiT residual meta state: {residual_meta[:3]}")
        shape_mismatches = []
        dtype_mismatches = []
        for key, (shape, dtype) in source_schema.items():
            if actual[key].shape != shape:
                shape_mismatches.append(f"{key}: {tuple(actual[key].shape)} != {tuple(shape)}")
            if actual[key].dtype != dtype:
                dtype_mismatches.append(f"{key}: {actual[key].dtype} != {dtype}")
        if shape_mismatches or dtype_mismatches:
            raise RuntimeError(
                "file-backed DiT source/model state mismatch: "
                f"shapes={shape_mismatches[:3]} dtypes={dtype_mismatches[:3]}"
            )
        return result

    load._miniacc_file_backed = True
    comfy.sd.load_diffusion_model = load


def install_file_backed_text_loader(comfy_args, *, expected_size=51506295256):
    """Keep the original BF16 conditioner mapped; retain Comfy's CPU forward.

    Its 51.5-GB state must not be copied into another full-size host allocation.
    Large Linear weights use the same checked assignment as the DiT; small
    layers and the embedding keep normal Comfy loading/forward semantics.
    """
    if not getattr(comfy_args, "disable_pinned_memory", False):
        raise RuntimeError("file-backed text requires --disable-pinned-memory")
    if getattr(comfy_args, "disable_mmap", False):
        raise RuntimeError("file-backed text requires mmap")
    if not getattr(comfy_args, "disable_dynamic_vram", False):
        raise RuntimeError("file-backed text requires DynamicVRAM disabled")

    import comfy.ops
    import comfy.sd

    original = comfy.sd.load_clip
    if getattr(original, "_miniacc_file_backed_text", False):
        return

    def load(ckpt_paths, embedding_directory=None, clip_type=None,
             model_options=None, disable_dynamic=False):
        clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION if clip_type is None else clip_type
        options = dict(model_options or {})
        target = len(ckpt_paths) == 1 and Path(ckpt_paths[0]).name == COMFY_TURBO_CLIP
        if target:
            if Path(ckpt_paths[0]).stat().st_size != expected_size:
                raise RuntimeError("original BF16 text encoder file is incomplete")
            if options.get("dtype") not in (None, torch.bfloat16):
                raise RuntimeError("original BF16 text encoder refuses dtype substitution")
            if options.get("custom_operations") is not None:
                raise RuntimeError("original BF16 text encoder refuses custom operation substitution")
            for key in ("load_device", "initial_device", "offload_device"):
                if key in options and torch.device(options[key]).type != "cpu":
                    raise RuntimeError(f"file-backed text requires CPU {key}")
                options[key] = torch.device("cpu")
            options["dtype"] = torch.bfloat16
            # SDClipModel's ordinary full-precision forward already uses manual_cast.
            options["custom_operations"] = operations_for_h3(comfy.ops.manual_cast)
        result = original(
            [str(path) for path in ckpt_paths], embedding_directory=embedding_directory,
            clip_type=clip_type, model_options=options, disable_dynamic=disable_dynamic,
        )
        if target:
            residual = [name for name, value in result.cond_stage_model.named_parameters()
                        if value.is_meta]
            if residual:
                raise RuntimeError(f"file-backed text has unloaded meta weights: {residual[:3]}")
        return result

    load._miniacc_file_backed_text = True
    comfy.sd.load_clip = load
