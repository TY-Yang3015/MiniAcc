#!/usr/bin/env python3
"""Build the frozen Light4 T2VA AdaLN sidecar with native single-rank math.

Preparation only: no denoiser, media generation, training, or speed measurement.
The native cache builder preserves each plan's GEMM batch size. Publication
requires a complete build, native lookup round-trip, and resource checks.
"""
from __future__ import annotations
if not __debug__:
    raise RuntimeError('optimized Python disables required contract assertions')

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(os.environ.get('MINIACC_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
# The selected executable runtime pins PYTHONPATH to its own site-packages.
# Direct script execution therefore needs the project root explicitly.
sys.path.insert(0, str(ROOT))
from miniacc_core.stage3 import append_jsonl, atomic_write_json, query_resources, resource_violation

SNAPSHOT = ROOT / '.local/sglang-hf-cache/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA/transformer'
ADAPTER = ROOT / '.local/sglang-adapters/minimax_h3_fl2v_turbo_4step_v0.1.safetensors'
ADAPTER_SHA = '5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29'
PLANS = Path(os.environ.get('MINIACC_ADALN_PLAN_PATH', ROOT / 'artifacts/review/stage3-adaln-plan-parent-20260915/native-plans.json'))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_header(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'expected regular safetensors file: {path}')
    with path.open('rb') as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError('missing safetensors header length')
        size = struct.unpack('<Q', size_bytes)[0]
        if not 0 < size <= 16 * 1024**2:
            raise ValueError('unexpected safetensors header size')
        header = json.loads(handle.read(size))
    return {key: value for key, value in header.items() if key != '__metadata__'}


def checkpoint_files(snapshot: Path, allowed_blob_root: Path | None = None) -> list[Path]:
    """Resolve only explicitly approved read-only HF blob links, not arbitrary links."""
    files = []
    for path in sorted(snapshot.glob('*.safetensors')):
        if path.is_symlink():
            if allowed_blob_root is None:
                raise ValueError(f'checkpoint symlink requires an approved blob root: {path}')
            resolved = path.resolve(strict=True)
            if resolved.parent != allowed_blob_root.resolve(strict=True) or not resolved.is_file():
                raise ValueError(f'checkpoint link escapes approved HF blob root: {path}')
            path = resolved
        files.append(path)
    return files


def audit_adapter(path: Path = ADAPTER) -> dict:
    header = tensor_header(path)
    affected = [key for key in header if any(part in key.lower() for part in
                ('adaln', 'time_embed', 'timestep', 'norm1.linear'))]
    if affected:
        raise ValueError('adapter targets a cache dependency: ' + ', '.join(affected[:8]))
    actual = digest(path)
    if actual != ADAPTER_SHA or len(header) != 624:
        raise ValueError('not the frozen LightX2V4 adapter')
    return {'sha256': actual, 'tensor_count': len(header), 'dependency_target_keys': affected}


class ResourceGuard:
    """Monitor this process, optionally its newly created owned session group."""

    def __init__(self, output: Path, timeout_seconds: float, *, own_process_group: bool = False,
                 query_timeout_seconds: float = 4.0, gpu_index: int = 0):
        if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
            raise ValueError('physical GPU index must be a nonnegative integer')
        self.gpu_index = gpu_index
        if not 0 < query_timeout_seconds <= 10:
            raise ValueError('resource query timeout must be in (0, 10] seconds')
        self.query_timeout_seconds = query_timeout_seconds
        self.owned_group = os.getpgrp() if own_process_group else None
        if self.owned_group is not None and (self.owned_group != os.getpid() or os.getsid(0) != os.getpid()):
            raise ValueError('group termination requires this process to lead its own new session')
        self.output = output
        self.deadline = time.monotonic() + timeout_seconds
        self.stop = threading.Event()
        self.failure = None
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def sample(self):
        snapshot = query_resources(gpu_index=self.gpu_index, timeout_seconds=self.query_timeout_seconds)
        row = {'timestamp_epoch': time.time(), 'monitored_gpu_index': self.gpu_index, **asdict(snapshot)}
        row['violation'] = resource_violation(snapshot, gpu_index=self.gpu_index)
        append_jsonl(self.output / 'resources.jsonl', row)
        if row['violation']:
            raise RuntimeError(row['violation'])
        if time.monotonic() >= self.deadline:
            raise TimeoutError('sidecar preparation deadline exceeded')

    def _watch(self):
        while not self.stop.wait(0.5):
            try:
                self.sample()
            except Exception as error:
                self.failure = f'{type(error).__name__}: {error}'
                try:
                    atomic_write_json(self.output / 'failure.json',
                                      {'status': 'resource_or_deadline_failure', 'error': self.failure})
                finally:
                    # A full/unwritable evidence disk must not disable abort.
                    if self.owned_group is not None and os.getpgrp() == self.owned_group:
                        os.killpg(self.owned_group, signal.SIGTERM)
                    else:
                        os.kill(os.getpid(), signal.SIGTERM)
                return

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('resource monitor did not settle')
        if self.failure:
            raise RuntimeError(self.failure)
        if kind is None:
            self.sample()


@contextmanager
def native_single_rank(linear, h3):
    # Native linears accept GroupCoordinator metadata. No collective is called
    # at world_size=1; only the group lookup is supplied by this standalone tool.
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    with patch.object(linear, 'get_tp_group', lambda: group), \
         patch.object(h3, 'get_tp_world_size', lambda: 1):
        yield


def build(output: Path, timeout_seconds: float = 900,
          expected_device: str = 'NVIDIA GeForce RTX 4090', physical_gpu_index: int = 0) -> dict:
    if expected_device not in ('NVIDIA GeForce RTX 4090', 'NVIDIA A100 80GB PCIe'):
        raise ValueError('unsupported preparation device profile')
    if expected_device == 'NVIDIA A100 80GB PCIe' and os.environ.get('CUDA_VISIBLE_DEVICES') != str(physical_gpu_index):
        raise ValueError('remote CUDA visibility must match the explicit physical GPU index')
    if not 0 < timeout_seconds <= 1800:
        raise ValueError('preparation timeout must be in (0, 1800] seconds')
    output = output.absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('refusing symlinked output path')
    output.mkdir(parents=True, exist_ok=False)
    try:
        with ResourceGuard(output, timeout_seconds,
                           query_timeout_seconds=10 if expected_device == 'NVIDIA A100 80GB PCIe' else 4,
                           gpu_index=physical_gpu_index):
            import torch
            from safetensors import safe_open
            from safetensors.torch import save_file
            from sglang.multimodal_gen.runtime.layers import linear
            from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as h3

            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required for native CUDA rounding')
            if torch.cuda.get_device_name(0) != expected_device:
                raise RuntimeError('CUDA device does not match the explicit preparation profile')
            if expected_device == 'NVIDIA A100 80GB PCIe':
                import socket
                if socket.gethostname().split('.')[0] != 'wolpy08':
                    raise RuntimeError('remote preparation is authorized only on wolpy08')
            if torch.backends.cuda.matmul.allow_tf32:
                raise RuntimeError('unexpected TF32 setting; do not silently change native math')
            torch.cuda.set_device(0)
            adapter = audit_adapter()
            plan_record = json.loads(PLANS.read_text())
            source_root = Path(h3.__file__).parents[2]
            for relative, expected in plan_record['source_sha256'].items():
                if digest(source_root / relative) != expected:
                    raise ValueError(f'native plan source changed: {relative}')
            config = json.loads((SNAPSHOT / 'config.json').read_text())
            arch = SimpleNamespace(**config)
            if (arch.num_layers, arch.hidden_size, arch.adaln_out_features) != (50, 5376, 96768):
                raise ValueError('unexpected frozen model architecture')
            blob_root = (ROOT / '.local/sglang-hf-cache/models--MiniMaxAI--MiniMax-H3/blobs'
                         if expected_device == 'NVIDIA A100 80GB PCIe' else None)
            files = checkpoint_files(SNAPSHOT, blob_root)
            atomic_write_json(output / 'checkpoint-files.json',
                              {'snapshot': str(SNAPSHOT), 'resolved_read_only_files': [str(p) for p in files]})
            if len(files) != 13:
                raise ValueError('expected 13 frozen transformer shards')
            index = {}
            for path in files:
                for key in tensor_header(path):
                    if key in index:
                        raise ValueError(f'duplicate checkpoint key: {key}')
                    index[key] = path
            plans = [torch.tensor(row['fp32_timesteps'], dtype=torch.float32) for row in plan_record['plans']]
            if [list(h3._plan_key(t)) for t in plans] != [r['fp32_bits'] for r in plan_record['plans']]:
                raise ValueError('FP32 plan bit patterns did not round-trip')
            if [t.numel() for t in plans] != [1, 2, 2, 2]:
                raise ValueError('unexpected native GEMM plan shapes')
            with native_single_rank(linear, h3), torch.no_grad():
                embedder = h3.MiniMaxH3TimeEmbedder(arch, prefix='time_embedder').eval()
                weights = {}
                for name in embedder.state_dict():
                    full = 'time_embedder.' + name
                    with safe_open(index[full], framework='pt', device='cpu') as handle:
                        value = handle.get_tensor(full)
                        if value.dtype != torch.float32:
                            raise ValueError('time embedding must remain FP32')
                        weights[name] = value
                embedder.load_state_dict(weights, strict=True, assign=True)
                embedder.to('cuda:0')
                cache = h3.MiniMaxH3AdalnCache(arch, weight_files=[str(p) for p in files],
                                             max_plans=4, max_plan_width=2)
                cache.load(torch.device('cuda:0'))
                def embed(t):
                    return torch.nn.functional.silu(embedder(t)).to(torch.bfloat16)
                cache.build(plans, embed=embed)
                torch.cuda.synchronize()
                if cache.rebuilds != 1:
                    raise ValueError('unexpected native cache build count')
                tensors = {name: getattr(cache, name).detach().cpu().contiguous() for name in
                           ('plan_timesteps', 'plan_lengths', 'block_params', 'final_params')}
                provenance = {'model_revision': '42ed227ee7df40d41602854ae760620d6eb651fe',
                    'adapter': adapter, 'plan_record_sha256': digest(PLANS),
                    'native_source_sha256': plan_record['source_sha256'],
                    'producer_device': torch.cuda.get_device_name(0), 'tp_size': 1,
                    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                    'plan_gemm_rows': [t.numel() for t in plans],
                    'scope': 'native same-device sidecar preparation; no full-model equivalence or speed claim'}
                path = output / 'cache.safetensors'
                save_file(tensors, str(path), metadata={'format_version': '2', 'model_variant': 'fl2va',
                          'miniacc_provenance': json.dumps(provenance, sort_keys=True)})
                loaded = h3.MiniMaxH3AdalnCache(arch, path=str(path), model_variant='fl2va')
                loaded.load(torch.device('cpu'))
                for name, value in tensors.items():
                    if not torch.equal(getattr(loaded, name), value):
                        raise ValueError(f'sidecar storage round-trip changed {name}')
                for expected_slot, plan in enumerate(plans):
                    if loaded.lookup(plan).item() != expected_slot:
                        raise ValueError('native sidecar lookup mismatch')
                result = {'status': 'prepared_not_benchmarked', 'sidecar': str(path),
                          'sha256': digest(path), 'provenance': provenance,
                          'native_storage_roundtrip': True, 'native_lookup_checks': 4}
        # Publish only after the resource monitor has settled and final checks pass.
        result['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(output / 'ready.json', result)
        return result
    except Exception as error:
        atomic_write_json(output / 'failure.json', {'status': 'preparation_failed',
                          'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-device', default='NVIDIA GeForce RTX 4090',
                        choices=['NVIDIA GeForce RTX 4090', 'NVIDIA A100 80GB PCIe'])
    parser.add_argument('--physical-gpu-index', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build(args.output, expected_device=args.expected_device,
                           physical_gpu_index=args.physical_gpu_index), indent=2))
