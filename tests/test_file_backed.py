"""Offline tests for the opt-in H3 file-backed construction seam."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from miniacc_core.config import (
    COMFY_NVFP4_CLIP,
    COMFY_TURBO_ADAPTER,
    COMFY_TURBO_DIFFUSION,
    COMFY_TURBO_CLIP,
    CandidateConfig,
)
from miniacc_core.probe import comfy_server_command

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / ".cache/miniacc/89a54668-runtime/bin/python"


class FileBackedPolicyTests(unittest.TestCase):
    def test_policy_is_exact_structural_control_only(self):
        candidate = CandidateConfig(
            family="comfyui-turbo",
            diffusion_name=COMFY_TURBO_DIFFUSION,
            clip_name=COMFY_NVFP4_CLIP,
            adapter_name=COMFY_TURBO_ADAPTER,
            steps=4,
            video_shift=6.0,
            audio_shift=3.0,
            file_backed_dit=True,
        )
        self.assertTrue(candidate.file_backed_dit)
        with self.assertRaisesRegex(ValueError, "restricted"):
            CandidateConfig(
                family="comfyui-turbo",
                diffusion_name=COMFY_TURBO_DIFFUSION,
                clip_name=COMFY_NVFP4_CLIP,
                adapter_name=COMFY_TURBO_ADAPTER,
                steps=4,
                video_shift=7.0,
                audio_shift=3.0,
                file_backed_dit=True,
            )

    def test_original_bf16_reference_is_distinct_from_quantized_base(self):
        reference = dict(
            family="comfyui-base", diffusion_name=COMFY_TURBO_DIFFUSION,
            clip_name=COMFY_TURBO_CLIP, file_backed_dit=True,
        )
        self.assertTrue(CandidateConfig(**reference).is_bf16_reference)
        self.assertFalse(CandidateConfig().is_bf16_reference)
        for change in ({"clip_name": COMFY_NVFP4_CLIP}, {"steps": 4}, {"video_shift": 6}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "restricted"):
                CandidateConfig(**(reference | change))

    def test_bf16_text_policy_is_explicit_in_owned_bootstrap_command(self):
        command = comfy_server_command(
            "python", Path("main.py"), output_root=Path("media"),
            extra_model_paths=Path("paths.yaml"), file_backed_dit=True,
            file_backed_text=True,
        )
        self.assertIn("--file-backed-text", command)
        self.assertIn("--disable-pinned-memory", command)

    def test_owned_command_disables_pinning_only_for_opt_in(self):
        common = dict(
            python="python",
            comfy_main=Path("main.py"),
            output_root=Path("media"),
            extra_model_paths=Path("paths.yaml"),
            bootstrap=Path("bootstrap.py"),
        )
        normal = comfy_server_command(**common)
        opted = comfy_server_command(**common, file_backed_dit=True)
        self.assertNotIn("--disable-pinned-memory", normal)
        self.assertIn("--file-backed-dit", opted)
        self.assertIn("--disable-pinned-memory", opted)
        self.assertIn("--disable-async-offload", opted)
        self.assertIn("--disable-dynamic-vram", opted)

    def test_bootstrap_consumes_owned_file_policy_and_pinning_flag(self):
        script = r'''
import json, os, sys, types
from pathlib import Path
import miniacc_bootstrap

root = Path(os.environ["FIXTURE_ROOT"])
(root / "comfy").mkdir()
(root / "comfy" / "__init__.py").write_text("")
(root / "comfy" / "options.py").write_text(
    "args_parsing = False\n"
    "def enable_args_parsing():\n"
    "    global args_parsing\n"
    "    args_parsing = True\n"
)
(root / "comfy" / "cli_args.py").write_text(
    "import argparse, comfy.options\n"
    "parser = argparse.ArgumentParser()\n"
    "parser.add_argument('--disable-pinned-memory', action='store_true')\n"
    "parser.add_argument('--disable-dynamic-vram', action='store_true')\n"
    "args = parser.parse_args(None if comfy.options.args_parsing else [])\n"
)
(root / "main.py").write_text(
    "import json\n"
    "from comfy.cli_args import args\n"
    "print(json.dumps(vars(args), sort_keys=True))\n"
)
module = types.ModuleType("miniacc_core.file_backed")
def install(args):
    assert args.disable_pinned_memory and args.disable_dynamic_vram
    print(json.dumps({'policy_installed': True}))
module.install_file_backed_dit_loader = install
sys.modules[module.__name__] = module
miniacc_bootstrap.configure_allocator = lambda *_: None
miniacc_bootstrap.install_forward_markers = lambda: None
miniacc_bootstrap.wrap_extra_node_initialization = lambda: None
miniacc_bootstrap.main()
'''
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env["FIXTURE_ROOT"] = directory
            result = subprocess.run(
                [str(RUNTIME), "-c", script, "--comfy-main", str(Path(directory) / "main.py"),
                 "--file-backed-dit", "--disable-pinned-memory", "--disable-dynamic-vram"],
                cwd=ROOT, env=env, capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn('"policy_installed": true', result.stdout)
        self.assertIn('"disable_pinned_memory": true', result.stdout)

    def test_real_comfy_loader_wrapper_and_guards(self):
        code = r'''
import os, sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path.cwd() / ".local/ComfyUI"))
sys.argv = ["fixture", "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import comfy.sd
from safetensors.torch import save_file
from miniacc_core.file_backed import install_file_backed_dit_loader

root = Path(os.environ["FIXTURE_ROOT"])
path = root / "minimax_h3_fl2va_bf16.safetensors"
save_file({"weight": torch.zeros((2, 2)), "bias": torch.zeros(2)}, str(path))
other = root / "ordinary.safetensors"
save_file({"weight": torch.zeros((2, 2))}, str(other))
size = path.stat().st_size
args = SimpleNamespace(disable_pinned_memory=True, disable_mmap=False, disable_dynamic_vram=True)
mode = "ok"
calls = []
def original(source, **kwargs):
    assert isinstance(source, str)
    calls.append((source, kwargs))
    if source.endswith("ordinary.safetensors"):
        assert "model_options" not in kwargs
        return "fallback"
    if mode == "residual":
        model = torch.nn.Linear(2, 2, device="meta")
    elif mode == "extra":
        model = torch.nn.Linear(2, 2)
        model.register_parameter("extra", torch.nn.Parameter(torch.zeros(1)))
    elif mode == "shape":
        model = torch.nn.Linear(3, 2)
    elif mode == "dtype":
        model = torch.nn.Linear(2, 2)
        model.weight = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.float16))
    else:
        model = torch.nn.Linear(2, 2)
    return SimpleNamespace(model=SimpleNamespace(diffusion_model=model))
comfy.sd.load_diffusion_model = original
install_file_backed_dit_loader(args, expected_size=size)
wrapped = comfy.sd.load_diffusion_model
result = wrapped(path, model_options=None, disable_dynamic=False)
assert result.model.diffusion_model.weight.shape == (2, 2)
assert calls[-1][1]["model_options"]["custom_operations"]._miniacc_file_backed_operations
assert wrapped is comfy.sd.load_diffusion_model
assert wrapped(other) == "fallback"
try:
    wrapped(path, model_options={"dtype": torch.float16})
except RuntimeError as error:
    assert "dtype override" in str(error)
else:
    raise AssertionError("incompatible dtype override accepted")
try:
    wrapped(path, model_options={"offload_device": torch.device("cuda")})
except RuntimeError as error:
    assert "offload_device" in str(error)
else:
    raise AssertionError("incompatible offload override accepted")
for name, bad in (("pin", SimpleNamespace(disable_pinned_memory=False, disable_mmap=False, disable_dynamic_vram=True)),
                  ("mmap", SimpleNamespace(disable_pinned_memory=True, disable_mmap=True, disable_dynamic_vram=True)),
                  ("dynamic", SimpleNamespace(disable_pinned_memory=True, disable_mmap=False, disable_dynamic_vram=False))):
    try:
        install_file_backed_dit_loader(bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError(name + " guard accepted")
# Reinstall against a fresh upstream function to exercise the source-size gate.
comfy.sd.load_diffusion_model = original
for key in ("_miniacc_file_backed",):
    if hasattr(comfy.sd.load_diffusion_model, key):
        delattr(comfy.sd.load_diffusion_model, key)
install_file_backed_dit_loader(args, expected_size=size + 1)
try:
    comfy.sd.load_diffusion_model(path)
except RuntimeError as error:
    assert "source size" in str(error)
else:
    raise AssertionError("wrong source size accepted")
# Exercise final key, meta, shape and dtype diagnostics through the real wrapper.
for mode_name, text in (("extra", "missing=[] unexpected=['extra']"),
                        ("residual", "residual meta"),
                        ("shape", "shapes="),
                        ("dtype", "dtypes=")):
    mode = mode_name
    comfy.sd.load_diffusion_model = original
    if hasattr(comfy.sd.load_diffusion_model, "_miniacc_file_backed"):
        delattr(comfy.sd.load_diffusion_model, "_miniacc_file_backed")
    install_file_backed_dit_loader(args, expected_size=size)
    try:
        comfy.sd.load_diffusion_model(path)
    except RuntimeError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(mode_name + " validation accepted")
print({"fixture_bytes": size, "fallback": True, "guards": 3, "validation_modes": 4})
'''
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "triton.py").write_text(
                "raise ImportError('CPU fixture disables optional Triton backend')\n"
            )
            env["PYTHONPATH"] = directory + os.pathsep + str(ROOT)
            env["FIXTURE_ROOT"] = directory
            result = subprocess.run(
                [str(RUNTIME), "-c", code], cwd=ROOT, env=env,
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("validation_modes': 4", result.stdout)

    def test_real_mapped_direct_diff_and_eager_restore(self):
        code = r'''
import hashlib, os, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path.cwd() / ".local/ComfyUI"))
sys.argv = ["fixture", "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
from safetensors.torch import save_file
from safetensors import safe_open
from comfy import lora
from comfy.model_patcher import LowVramPatch, ModelPatcher

root = Path(os.environ["FIXTURE_ROOT"])
path = root / "mapped.safetensors"
save_file({"weight": torch.zeros((2, 2), dtype=torch.bfloat16)}, str(path))
def digest():
    return hashlib.sha256(path.read_bytes()).hexdigest()
original_digest = digest()
diff = torch.full((2, 2), 0.5, dtype=torch.float32)
patches = {"weight": [(1.0, ("diff", (diff,)), 1.0, None, None)]}
with safe_open(str(path), framework="pt", device="cpu") as handle:
    mapped = handle.get_tensor("weight")
    before = mapped.clone()
    lowvram = LowVramPatch("weight", patches)
    observed = lowvram(mapped)
    expected = before.clone()
    lora.calculate_weight(patches["weight"], expected, "weight")
    assert torch.equal(observed, expected)
    assert digest() == original_digest
with safe_open(str(path), framework="pt", device="cpu") as handle:
    mapped_for_eager = handle.get_tensor("weight")
    before_eager = mapped_for_eager.clone()
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(mapped_for_eager)
    patcher = ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    patcher.patches = patches
    patcher.patch_weight_to_device("weight")
    eager_expected = before_eager.clone()
    lora.calculate_weight(patches["weight"], eager_expected, "weight")
    assert torch.equal(model.weight, eager_expected)
    assert digest() == original_digest
    patcher.unpatch_model()
    assert torch.equal(model.weight, before_eager)
    assert digest() == original_digest
print({"direct_diff_patch": True, "eager_restore": True, "file_immutable": True})
'''
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "triton.py").write_text(
                "raise ImportError('CPU fixture disables optional Triton backend')\n"
            )
            env["PYTHONPATH"] = directory + os.pathsep + str(ROOT)
            env["FIXTURE_ROOT"] = directory
            result = subprocess.run(
                [str(RUNTIME), "-c", code], cwd=ROOT, env=env,
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("file_immutable': True", result.stdout)

    def test_real_detection_loader_mapping_and_rank_two_lora(self):
        code = r'''
"""Real Comfy detection/load and rank-two LoRA; tiny synthetic CPU checkpoint."""
from pathlib import Path
import gc
import hashlib
import json
import os
import sys

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
HERE = Path(os.environ['FIXTURE_ROOT']).resolve()
PROJECT = Path.cwd()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / '.local/ComfyUI'))
sys.argv = ['cpu-fixture', '--cpu', '--disable-pinned-memory', '--disable-dynamic-vram', '--disable-async-offload']
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.cli_args
import comfy.sd
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.ops import disable_weight_init
from comfy.model_patcher import LowVramPatch
from comfy.weight_adapter.lora import LoRAAdapter
from safetensors.torch import save_file
import miniacc_core.file_backed as loading

path = HERE / loading.DIFFUSION_FILE
assert not path.exists() and not (HERE / 'result.json').exists()
assert not torch.cuda.is_initialized()
kwargs = dict(
    hidden_size=8, num_layers=1, token_refiner_num_layers=0,
    num_attention_heads=2, attention_head_dim=4, ffn_hidden_size=16,
    latents_dim=24, audio_latents_dim=32, text_dim=8,
    timestep_input_dim=4, time_embed_hidden_size=8, time_embed_dim=4,
    rope_inv_freq_len=2, dtype=torch.bfloat16, device=None,
)
# Only the constructor and fresh synthetic state are used; no H3 forward.
template = MiniMaxH3Model(**kwargs, operations=disable_weight_init)
state = {key: torch.full(value.shape, 0.125, dtype=value.dtype) for key, value in template.state_dict().items()}
save_file(state, str(path))
key_count = len(state)
del template, state
assert path.stat().st_size <= 1024 * 1024
source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
# Fixture-scale policy inputs only; production source and default1MiB
# threshold remain unchanged on disk and in every other process.
loading.META_THRESHOLD_BYTES = 32
preview = MiniMaxH3Model(**kwargs, operations=loading.operations_for_h3(disable_weight_init))
assert preview.condition_proj.weight.is_meta
assert preview.video_patch_proj.weight.is_meta
assert preview.video_patch_proj.weight.dtype == torch.float32
assert preview.rope.inv_freq.device.type == 'cpu'
del preview
loading.install_file_backed_dit_loader(comfy.cli_args.args, expected_size=path.stat().st_size)


def mappings():
    result = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5] == str(path):
            first, last = fields[0].split('-')
            result.append((int(first, 16), int(last, 16), fields[1], int(fields[4])))
    return result


def assert_mapped(tensor):
    rows = [row for row in mappings() if row[0] <= tensor.data_ptr() < row[1]]
    assert len(rows) == 1 and rows[0][2] == 'rw-p'
    assert rows[0][3] == path.stat().st_ino


def check_actual_loader_and_lora():
    # The original Comfy loader/detection/constructor/state loader are NOT
    # replaced. Only the public policy's expected_size is fixture-scaled.
    patcher = comfy.sd.load_diffusion_model(str(path), model_options={'dtype': torch.bfloat16}, disable_dynamic=True)
    assert not patcher.is_dynamic()
    model = patcher.model.diffusion_model
    assert isinstance(model, MiniMaxH3Model)
    assert model.hidden_size == 8 and len(model.blocks) == 1
    assert type(model.condition_proj).__name__ == 'CheckedLinear'
    assert not any(value.is_meta for value in model.state_dict().values())
    assert model.video_patch_proj.weight.dtype == torch.float32
    assert model.condition_proj.weight.dtype == torch.bfloat16
    assert model.rope.inv_freq.device.type == 'cpu'
    gc.collect()
    assert_mapped(model.condition_proj.weight)
    base = model.condition_proj.weight.detach().clone()
    base_bias = model.condition_proj.bias.detach().clone()
    original_pointer = model.condition_proj.weight.data_ptr()
    up = torch.full((8, 2), 0.125, dtype=torch.float32)
    down = torch.full((2, 8), 0.25, dtype=torch.float32)
    adapter = LoRAAdapter(set(), (up, down, 2.0, None, None, None))
    key = 'diffusion_model.condition_proj.weight'
    assert patcher.add_patches({key: adapter}) == [key]
    expected_weight = (base.float() + up @ down).to(base.dtype)
    inputs = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8)
    expected_output = torch.nn.functional.linear(inputs, expected_weight, base_bias)
    patcher.load(device_to=torch.device('cpu'), lowvram_model_memory=1, force_patch_weights=False, full_load=False)
    assert any(isinstance(function, LowVramPatch) for function in model.condition_proj.weight_function)
    with torch.inference_mode():
        actual_output = model.condition_proj(inputs)  # Only one tiny Linear, not H3.
    assert torch.equal(actual_output, expected_output)
    assert torch.equal(model.condition_proj.weight, base), 'lazy cast modified mapped base values'
    assert model.condition_proj.weight.data_ptr() == original_pointer
    assert_mapped(model.condition_proj.weight)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
    patcher.unpatch_model()
    assert torch.equal(model.condition_proj.weight, base)
    # Eager patch and restoration use the same actual rank-two adapter.
    patcher.patch_weight_to_device(key, device_to=torch.device('cpu'))
    assert torch.equal(model.condition_proj.weight, expected_weight)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
    patcher.unpatch_model()
    assert torch.equal(model.condition_proj.weight, base)
    assert model.condition_proj.weight.data_ptr() == original_pointer
    assert_mapped(model.condition_proj.weight)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
    return {
        'actual_comfy_detection_and_loader': True,
        'actual_checked_h3_constructor_and_state_load': True,
        'detected_hidden_size': model.hidden_size,
        'detected_layers': len(model.blocks),
        'mixed_dtype_and_cpu_buffer_preserved': True,
        'model_only_mapped_storage': True,
        'actual_rank_two_lora': True,
        'actual_lowvram_cast_output_matches': True,
        'lazy_cast_base_values_and_pointer_unchanged': True,
        'eager_patch_and_restore_values_pointer': True,
    }


result = check_actual_loader_and_lora()
gc.collect()
assert mappings() == [], 'fixture mappings outlived all model/patcher owners'
assert not torch.cuda.is_initialized()
result.update(
    fixture_bytes=path.stat().st_size, state_keys=key_count, source_sha256=source_hash,
    mapping_teardown=True, checkpoint_unchanged=True, cuda_initialized=False,
    h3_real_checkpoint_payload_read=False, h3_forward_executed=False,
    tiny_linear_forward_executed=True, native_fit_established=False,
)
with (HERE / 'result.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result, indent=2))
'''
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "triton.py").write_text(
                "raise ImportError('CPU fixture hides optional GPU Triton backend')\n"
            )
            env["PYTHONPATH"] = directory + os.pathsep + str(ROOT)
            env["FIXTURE_ROOT"] = directory
            result = subprocess.run(
                [str(RUNTIME), "-c", code], cwd=ROOT, env=env,
                capture_output=True, text=True, check=False, timeout=90,
            )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn('"actual_comfy_detection_and_loader": true', result.stdout)
        self.assertIn('"actual_rank_two_lora": true', result.stdout)
        self.assertIn('"mapping_teardown": true', result.stdout)

    def test_real_comfy_h3_constructor_meta_assign_and_dtype(self):
        code = r'''
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path.cwd() / ".local/ComfyUI"))
sys.argv = ["fixture", "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
from miniacc_core.file_backed import operations_for_h3
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.ops import disable_weight_init
ops = operations_for_h3(disable_weight_init)
model = MiniMaxH3Model(
    hidden_size=4096, num_layers=0, token_refiner_num_layers=0,
    num_attention_heads=1, attention_head_dim=1, ffn_hidden_size=1,
    latents_dim=24, audio_latents_dim=32, text_dim=5120,
    time_embed_hidden_size=4096, time_embed_dim=16, dtype=torch.bfloat16,
    device=None, operations=ops,
)
state = {}
meta_before = []
for key, value in model.state_dict().items():
    if value.is_meta:
        meta_before.append(key)
    state[key] = torch.zeros(tuple(value.shape), dtype=value.dtype, device="cpu")
assert meta_before, "expected expensive H3 Linear parameters on meta"
assert model.rope.inv_freq.device.type == "cpu"
assert model.video_patch_proj.weight.is_meta
assert model.video_patch_proj.weight.dtype == torch.float32
missing, unexpected = model.load_state_dict(state, strict=True, assign=False)
assert not missing and not unexpected
assert not any(value.is_meta for value in model.state_dict().values())
assert model.video_patch_proj.weight.dtype == torch.float32
assert model.condition_proj.weight.dtype == torch.bfloat16
missing_state = dict(state)
missing_state.pop("condition_proj.weight")
try:
    model.condition_proj._load_from_state_dict(
        missing_state, "condition_proj.", {}, True, [], [], []
    )
except RuntimeError as error:
    assert "missing condition_proj.weight" in str(error)
else:
    raise AssertionError("missing state was accepted")
wrong_dtype = dict(state)
wrong_dtype["condition_proj.weight"] = wrong_dtype["condition_proj.weight"].float()
try:
    model.condition_proj._load_from_state_dict(
        wrong_dtype, "condition_proj.", {}, True, [], [], []
    )
except RuntimeError as error:
    assert "dtype mismatch" in str(error)
else:
    raise AssertionError("wrong dtype was accepted")
print({"meta_before": len(meta_before), "parameters": len(state), "residual_meta": 0,
       "video_dtype": str(model.video_patch_proj.weight.dtype),
       "condition_dtype": str(model.condition_proj.weight.dtype),
       "buffer_device": model.rope.inv_freq.device.type})
'''
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        # The pinned runtime's Triton build probes for an active CUDA driver at
        # import time.  Hide only that optional backend in a temporary fixture;
        # Comfy's real CPU operations and H3 class remain under test.
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "triton.py").write_text(
                "raise ImportError('CPU fixture disables optional Triton backend')\n"
            )
            env["PYTHONPATH"] = directory + os.pathsep + str(ROOT)
            result = subprocess.run(
                [str(RUNTIME), "-c", code], cwd=ROOT, env=env,
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("residual_meta': 0", result.stdout)
        self.assertIn("torch.float32", result.stdout)
        self.assertIn("torch.bfloat16", result.stdout)


if __name__ == "__main__":
    unittest.main()
