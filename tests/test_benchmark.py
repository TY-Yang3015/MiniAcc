"""CPU tests for reference identity, subset scoring metadata, and aggregation."""
import errno
import json
import os
import socket
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from miniacc_core.benchmark import (
    DEVELOPMENT_DIMENSIONS, NORMALIZATION, ROOT, PYTHON,
    normalized_score, percentile, profile_config, quality_index, scoring_metadata,
    check_port_available, record_admission_failure,
)
from miniacc_core.data import PromptDataModule


class BenchmarkTests(unittest.TestCase):
    def test_port_admission_allows_closed_server_time_wait(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            address = listener.getsockname()
            with socket.create_connection(address, timeout=3) as client:
                server, _ = listener.accept()
                with server:
                    server.settimeout(3)
                    server.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b'')
                    client.shutdown(socket.SHUT_WR)
                    self.assertEqual(server.recv(1), b'')
        # The old bare-bind guard fails here although no listener exists.
        with socket.socket() as old_probe:
            with self.assertRaises(OSError) as error:
                old_probe.bind(address)
            self.assertEqual(error.exception.errno, errno.EADDRINUSE)
        check_port_available(address[1])

    def test_port_admission_still_rejects_active_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            with self.assertRaises(OSError) as error:
                check_port_available(listener.getsockname()[1])
            self.assertEqual(error.exception.errno, errno.EADDRINUSE)

    def test_admission_stop_does_not_fabricate_native_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group = root / 'q0'
            group.mkdir()
            (group / 'runs.jsonl').write_text('existing completed record\n')
            with patch('miniacc_core.benchmark.ROOT', root):
                record_admission_failure(group, 'next-job', 2, 128,
                                         OSError(errno.EADDRINUSE, 'Address already in use'))
            status = json.loads((group / 'status.json').read_text())
            self.assertEqual(status['status'], 'blocked_before_spawn')
            self.assertEqual(status['completed'], 2)
            self.assertFalse(status['native_job_started'])
            self.assertEqual((group / 'runs.jsonl').read_text(), 'existing completed record\n')
            self.assertEqual(len((group / 'batch-events.jsonl').read_text().splitlines()), 1)

    def test_original_reference_is_not_quantized_or_distilled(self):
        q0 = profile_config('q0')
        self.assertTrue(q0.is_bf16_reference)
        self.assertEqual(q0.steps, 49)
        self.assertIsNone(q0.adapter_name)
        self.assertFalse(profile_config('comfy-t4-nvfp4-text').is_bf16_reference)
        with self.assertRaises(ValueError):
            profile_config('sglang-h3')

    def test_scalar_uses_all_seven_normalized_dimensions(self):
        raw = {d: (low + high) / 2 for d, (low, high) in NORMALIZATION.items()}
        self.assertAlmostEqual(quality_index(raw), 50)
        self.assertIsNone(quality_index(dict(list(raw.items())[:-1])))
        with self.assertRaises(ValueError):
            quality_index(raw | {'dynamic_degree': float('nan')})
        self.assertGreater(normalized_score('overall_consistency', 0.5), 100)

    def test_percentiles_do_not_fabricate_missing_timings(self):
        self.assertIsNone(percentile([], .5))
        self.assertEqual(percentile([1, 3], .5), 2)
        self.assertEqual(percentile([None, 3], .95), 3)

    def test_overall_metadata_preserves_actual_eligibility(self):
        manifest = ROOT / 'data/stage1/vbench_dev_manifest.json'
        data = PromptDataModule(manifest)
        first = data.first_job()
        eligible_prompt = next(p for p in data.manifest['prompts']
                               if 'overall_consistency' in p['eligible_standard_development_dimensions'])
        eligible_job = next(j for j in data.jobs() if j['prompt_id'] == eligible_prompt['id'])
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory)
            rows = [{'job_id': job['id'], 'status': 'completed', 'media': [f"/{job['id']}.mp4"]}
                    for job in (first, eligible_job)]
            (group / 'runs.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            custom, standard = scoring_metadata(group, manifest)
        self.assertEqual(len(custom), 2)
        self.assertEqual(set(custom[0]['dimension']), set(DEVELOPMENT_DIMENSIONS[:-1]))
        self.assertEqual(len(standard), 1)
        self.assertEqual(standard[0]['prompt_en'], eligible_prompt['prompt_en'])
        self.assertIn('overall_consistency', standard[0]['dimension'])

    def test_mapped_bf16_text_linear_keeps_comfy_float_forward(self):
        # Real Comfy manual_cast operations and real mapped safetensors, no GPU.
        script = r'''
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / '.local/ComfyUI'))
sys.argv = ['test', '--cpu', '--disable-dynamic-vram', '--disable-pinned-memory']
import comfy.options
comfy.options.enable_args_parsing()
import torch, comfy.ops
from safetensors.torch import save_file, load_file
from miniacc_core.file_backed import operations_for_h3
with tempfile.TemporaryDirectory() as directory:
    path = str(Path(directory) / 'text.safetensors')
    weight = torch.arange(1024 * 1024).reshape(1024, 1024).remainder(31).to(torch.bfloat16)
    save_file({'weight': weight}, path)
    source = load_file(path)
    layer = operations_for_h3(comfy.ops.manual_cast).Linear(1024, 1024, bias=False, device='cpu', dtype=torch.bfloat16)
    assert layer.weight.is_meta
    layer.load_state_dict(source)
    assert layer.weight.data_ptr() == source['weight'].data_ptr()
    x = torch.ones(1, 1024, dtype=torch.float32)
    with torch.inference_mode():
        actual = layer(x)
        expected = torch.nn.functional.linear(x, source['weight'].float())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert layer.weight.dtype == torch.bfloat16
    assert not torch.cuda.is_initialized()
print('mapped BF16 text operation: CPU float forward unchanged')
'''
        with tempfile.TemporaryDirectory() as directory:
            # Same CPU-only fixture isolation as the existing real-loader tests;
            # installed Triton otherwise probes a CUDA driver during import.
            Path(directory, 'triton.py').write_text("raise ImportError('CPU fixture disables optional GPU backend')\n")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=directory + os.pathsep + str(ROOT))
            result = subprocess.run([str(PYTHON), '-c', script], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
