"""Local reference generation, official VBench subset scoring, and result matrices.

Generation is serial and stops on the first failed job. Existing successful jobs
are reused by exact manifest ID, never regenerated as an automatic warm-up.
Importing this module does not import torch or load an evaluation model.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time

from .config import (
    COMFY_TURBO_CLIP, COMFY_TURBO_DIFFUSION, COMFY_TURBO_ADAPTER,
    COMFY_NVFP4_CLIP, DEVELOPMENT_DIMENSIONS, CandidateConfig,
)
from .data import PromptDataModule

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path.home() / '.cache/miniacc/89a54668-runtime/bin/python'
VBENCH_SOURCE = Path.home() / '.cache/miniacc/vbench-reference-20260911/VBench-fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490'
# scripts/constant.py in the same official VBench revision. No clipping or
# official full-suite weighting is applied to this seven-dimension index.
NORMALIZATION = {
    'subject_consistency': (0.1462, 1.0),
    'background_consistency': (0.2615, 1.0),
    'motion_smoothness': (0.706, 0.9975),
    'dynamic_degree': (0.0, 1.0),
    'aesthetic_quality': (0.0, 1.0),
    'imaging_quality': (0.0, 1.0),
    'overall_consistency': (0.0, 0.364),
}


def normalized_score(dimension, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'non-finite {dimension} score')
    low, high = NORMALIZATION[dimension]
    return 100 * (value - low) / (high - low)


def quality_index(scores):
    if any(scores.get(d) is None for d in DEVELOPMENT_DIMENSIONS):
        return None
    return statistics.mean(normalized_score(d, scores[d]) for d in DEVELOPMENT_DIMENSIONS)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def profile_config(profile):
    if profile == 'q0':
        return CandidateConfig(diffusion_name=COMFY_TURBO_DIFFUSION,
                               clip_name=COMFY_TURBO_CLIP, file_backed_dit=True)
    if profile in ('comfy-t4-bf16-text', 'comfy-t4-nvfp4-text'):
        return CandidateConfig(
            family='comfyui-turbo', diffusion_name=COMFY_TURBO_DIFFUSION,
            clip_name=COMFY_TURBO_CLIP if profile == 'comfy-t4-bf16-text' else COMFY_NVFP4_CLIP,
            adapter_name=COMFY_TURBO_ADAPTER, steps=4, video_shift=6,
            file_backed_dit=True,
        )
    raise ValueError(f'profile is not implemented: {profile}')


def probe_command(candidate, job_id, group, manifest, timeout):
    command = [str(PYTHON), str(ROOT / 'local_probe.py'),
               '--manifest', str(manifest), '--job-id', job_id,
               '--run-root', str(group / 'native'),
               '--output', str(group / 'results' / f'{job_id}.json'),
               '--family', candidate.family, '--diffusion-name', candidate.diffusion_name,
               '--clip-name', candidate.clip_name,
               '--steps', str(candidate.steps), '--video-shift', str(candidate.video_shift),
               '--audio-shift', str(candidate.audio_shift), '--file-backed-dit',
               '--allocator-gib', '16', '--cpu-threads', '8', '--vae-device', 'gpu',
               '--timeout', str(timeout)]
    if candidate.adapter_name:
        command += ['--adapter-name', candidate.adapter_name, '--adapter-scale', '1.0']
    return command


def records(group):
    ledger = Path(group) / 'runs.jsonl'
    return [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []


def measured_phases(result):
    """Read phase markers and sampled extrema, without rehashing artifacts."""
    raw_log = Path(result['raw_log']) if result.get('raw_log') else None
    measured = {}
    if raw_log is None or not raw_log.exists():
        return measured
    markers = {}
    for line in raw_log.read_text(errors='replace').splitlines():
        try:
            event = json.JSONDecoder().raw_decode(line[line.index('{'):])[0]
        except (ValueError, json.JSONDecodeError):
            continue
        name = event.get('event', '')
        if name in ('miniacc_sampler_start', 'miniacc_sampler_end',
                    'miniacc_audio_decode_start', 'miniacc_audio_decode_end',
                    'miniacc_video_decode_start', 'miniacc_video_decode_end'):
            markers[name] = event['timestamp_monotonic']
    for phase in ('sampler', 'audio_decode', 'video_decode'):
        start, end = markers.get(f'miniacc_{phase}_start'), markers.get(f'miniacc_{phase}_end')
        measured[phase + '_seconds'] = end - start if start is not None and end is not None else None
    resource_file = raw_log.parent / 'resources.jsonl'
    if resource_file.exists():
        samples = [json.loads(line) for line in resource_file.read_text().splitlines()]
        gpu = [g for sample in samples for g in sample['gpus'] if g['index'] == 0]
        measured['sampled_gpu_used_max_mib'] = max((g['memory_used_mib'] for g in gpu), default=None)
        measured['sampled_gpu_free_min_mib'] = min((g['memory_free_mib'] for g in gpu), default=None)
        measured['sampled_host_available_min_bytes'] = min((s['host_available_bytes'] for s in samples), default=None)
        measured['resource_sample_count'] = len(samples)
    return measured


def log_result(record, group):
    with (group / 'runs.jsonl').open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    with (ROOT / 'BENCHMARKS.md').open('a') as stream:
        stream.write(
            f"\n- **{record['finished_utc']} {group.name} / {record['job_id']}**: "
            f"{record['status']}, CLI {record['exit_code']}; "
            f"process-cold E2E {record.get('cold_e2e_seconds')} s, "
            f"observed forwards {record.get('actual_forward_count')}. "
            f"[Result]({Path(record['result']).relative_to(ROOT)}); quality pending official scoring.\n"
        )


def check_port_available(port=8188):
    """Match the server's Unix address reuse without admitting a live listener."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))


def record_admission_failure(group, job_id, completed, requested, error):
    """Record a batch stop separately from a failed native generation attempt."""
    event = {
        'event': 'admission_failure', 'job_id': job_id,
        'completed': completed, 'requested': requested, 'native_job_started': False,
        'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'error_type': type(error).__name__, 'error': str(error),
    }
    with (group / 'batch-events.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')
    write_json(group / 'status.json', dict(event, status='blocked_before_spawn'))
    with (ROOT / 'BENCHMARKS.md').open('a') as stream:
        stream.write(f"\n- **{event['recorded_utc']} {group.name} batch admission stop**: "
                     f"{completed}/{requested} completed; next job {job_id} was not launched. "
                     f"{event['error_type']}: {error}. Existing clips retained; not a native clip failure. "
                     f"[Batch events]({(group / 'batch-events.jsonl').relative_to(ROOT)}).\n")


def generate(args):
    import fcntl
    from .runtime import ResourceGuard

    group = args.root.resolve() / args.profile
    group.mkdir(parents=True, exist_ok=True)
    # One project GPU job, including across separately launched batch runners.
    with (ROOT / '.local/benchmark-gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        candidate = profile_config(args.profile)
        data = PromptDataModule(args.manifest)
        jobs = list(data.jobs())
        if args.profile != 'q0':
            if args.allocation == 'filter':
                jobs = [job for job in jobs if job['allocation'] == 'cheap_filter']
            scored_ledger = ROOT / 'artifacts/evaluation/q0-scored/scored-ledger.json'
            if not scored_ledger.exists():
                raise ValueError('finish and score the cached Q0 reference before candidate quality screening')
            frozen_scores = json.loads(scored_ledger.read_text())
            if set(frozen_scores.get('scored_dimensions', [])) != set(DEVELOPMENT_DIMENSIONS):
                raise ValueError('cached Q0 scoring must contain all seven development dimensions')
        if not 1 <= args.count <= len(jobs):
            raise ValueError(f'count must be between 1 and {len(jobs)}')
        settings = {
            'profile': args.profile, 'manifest': data.manifest,
            'diffusion': candidate.diffusion_name, 'text_encoder': candidate.clip_name,
            'adapter': candidate.adapter_name, 'steps': candidate.steps,
            'video_shift': candidate.video_shift, 'audio_shift': candidate.audio_shift,
            'allocator_gib': 16, 'cpu_threads': 8, 'timeout_seconds': args.timeout,
            'timing': 'process-cold; ordinary filesystem/compile caches, not OS-cache-cold',
            'residency': False,
        }
        settings_file = group / 'settings.json'
        if settings_file.exists() and json.loads(settings_file.read_text()) != settings:
            raise ValueError('existing group has different settings; use a new output root')
        if not settings_file.exists():
            write_json(settings_file, settings)
        previous = records(group)
        if any(row['status'] != 'completed' for row in previous):
            raise ValueError('previous failed attempt retained; diagnose it before a separately explicit retry')
        complete = {row['job_id'] for row in previous}
        for job in jobs[:args.count]:
            if job['id'] in complete:
                continue
            try:
                encoder = ROOT / '.local/models/text_encoders' / candidate.clip_name
                if not encoder.is_file():
                    raise FileNotFoundError(f'required original encoder is not yet available: {encoder}')
                ResourceGuard().check()
                compute = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=15,
                )
                if compute.returncode != 0 or compute.stdout.strip():
                    raise RuntimeError('GPU is already in use or compute-process admission is unavailable')
                check_port_available()
            except Exception as error:
                record_admission_failure(group, job['id'], len(complete), args.count, error)
                raise
            logs = group / 'logs'
            logs.mkdir(exist_ok=True)
            command = probe_command(candidate, job['id'], group, args.manifest.resolve(), args.timeout)
            write_json(group / 'status.json', {'status': 'running', 'job_id': job['id'],
                                             'completed': len(complete), 'requested': args.count})
            write_json(logs / f"{job['id']}.command.json", command)
            started = time.monotonic()
            interrupted = False
            with (logs / f"{job['id']}.stdout").open('xb') as out, (logs / f"{job['id']}.stderr").open('xb') as err:
                process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                try:
                    # The CLI's independent RuntimeOwner deadline owns teardown.
                    # Do not kill only its parent and orphan the Comfy process group.
                    code = process.wait()
                except KeyboardInterrupt:
                    interrupted = True
                    process.send_signal(signal.SIGINT)
                    code = process.wait()
            result_path = group / 'results' / f"{job['id']}.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            count = result.get('actual_forward_count')
            valid = (code == 0 and count == candidate.steps
                     and result.get('history_status', {}).get('completed') is True
                     and bool(result.get('outputs')))
            record = {
                'job_id': job['id'], 'prompt_id': job['prompt_id'], 'seed': job['seed'],
                'status': 'completed' if valid else 'failed', 'exit_code': code,
                'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'cli_wall_seconds': time.monotonic() - started, 'result': str(result_path),
                'cold_e2e_seconds': result.get('cold_e2e_seconds'),
                'request_to_history_seconds': result.get('elapsed_seconds'),
                'process_to_cleanup_seconds': result.get('process_to_cleanup_seconds'),
                'actual_forward_count': count,
                'media': [item['path'] for item in result.get('outputs', [])],
                **measured_phases(result),
            }
            log_result(record, group)
            if valid:
                complete.add(job['id'])
            write_json(group / 'status.json', {'status': 'ready' if valid else 'failed',
                                             'completed': len(complete), 'requested': args.count,
                                             'last_job': record})
            matrix(args.root)
            if not valid or interrupted:
                raise RuntimeError(f"stopped after {job['id']}: CLI {code}, observed {count} forwards")
        return 0


def scoring_metadata(group, manifest):
    """Custom-input first six; original eligible metadata for overall consistency."""
    data = PromptDataModule(manifest)
    by_prompt = {}
    for row in records(group):
        if row['status'] != 'completed' or len(row['media']) != 1:
            continue
        job = data.job(row['job_id'])
        by_prompt.setdefault(job['prompt_id'], []).extend(row['media'])
    custom, standard = [], []
    for prompt in data.manifest['prompts']:
        videos = by_prompt.get(prompt['id'], [])
        if not videos:
            continue
        custom.append({'prompt_en': prompt['prompt_en'],
                       'dimension': list(DEVELOPMENT_DIMENSIONS[:-1]), 'video_list': videos})
        rows = [row for row in prompt['official_metadata_rows']
                if 'overall_consistency' in row['dimension']]
        if rows:
            # Preserve the official row, including any auxiliary metadata.
            standard.append(dict(rows[0], video_list=videos))
    return custom, standard


def score_dimension(args):
    # This entry point runs only in the separate VBench environment.
    import torch
    from .runtime import ResourceGuard
    ResourceGuard().check()
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(16 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    sys.path.insert(0, str(VBENCH_SOURCE))
    from vbench.utils import init_submodules

    custom, standard = scoring_metadata(args.group, args.manifest)
    dimension = args.dimension
    metadata = standard if dimension == 'overall_consistency' else custom
    completed_count = sum(row['status'] == 'completed' for row in records(args.group))
    output = args.group / 'quality' / str(completed_count) / dimension
    output.mkdir(parents=True, exist_ok=False)
    if not metadata:
        write_json(output / 'result.json', {'dimension': dimension, 'status': 'no_eligible_videos'})
        return 0
    info = output / 'input.json'
    write_json(info, metadata)
    module = importlib.import_module(f'vbench.{dimension}')
    compute = getattr(module, f'compute_{dimension}')
    started = time.monotonic()
    submodules = init_submodules([dimension], local=True, read_frame=False)[dimension]
    # Same official computation functions as VBench.evaluate, with explicitly
    # materialized subset metadata rather than its five-filename/full-suite loop.
    with torch.inference_mode():
        raw, details = compute(str(info), torch.device('cuda'), submodules)
    result = {'dimension': dimension, 'status': 'completed', 'raw': float(raw),
              'normalized': normalized_score(dimension, raw), 'details': details,
              'elapsed_seconds_including_backbone_load': time.monotonic() - started,
              'mode': 'standard_metadata' if dimension == 'overall_consistency' else 'custom_input',
              'videos': sum(len(row['video_list']) for row in metadata),
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    write_json(output / 'result.json', result)
    with (ROOT / 'BENCHMARKS.md').open('a') as stream:
        stream.write(f"\n- **VBench {args.group.name} / {dimension}**: {result['videos']} clips, "
                     f"raw {result['raw']:.6f}, normalized {result['normalized']:.3f}; "
                     f"{result['elapsed_seconds_including_backbone_load']:.3f} s including backbone loading. "
                     f"[Result]({(output / 'result.json').relative_to(ROOT)}).\n")
    return 0


def score_group(args):
    """Bound each official dimension in its own process; no H3/evaluator overlap."""
    import fcntl
    from .runtime import ResourceGuard, RuntimeOwner
    group = args.group.resolve()
    completed = sum(row['status'] == 'completed' for row in records(group))
    if not completed:
        raise ValueError('no completed reference/candidate clips to score')
    python = VBENCH_SOURCE.parent / 'env/bin/python'
    with (ROOT / '.local/benchmark-gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for dimension in DEVELOPMENT_DIMENSIONS:
            output = group / 'quality' / str(completed) / dimension
            result = output / 'result.json'
            if result.exists():
                continue
            logs = group / 'quality' / str(completed) / 'logs'
            logs.mkdir(parents=True, exist_ok=True)
            command = [str(python), '-m', 'miniacc_core.benchmark', 'score-dimension',
                       '--group', str(group), '--dimension', dimension,
                       '--manifest', str(args.manifest.resolve())]
            env = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                       NUMEXPR_NUM_THREADS='8', VBENCH_CACHE_DIR=str(Path.home() / '.cache/vbench'))
            owner = RuntimeOwner(guard=ResourceGuard(sample_log=logs / f'{dimension}.resources.jsonl'),
                                 timeout_seconds=args.timeout)
            code = None
            try:
                code = owner.run(command, logs / f'{dimension}.log', cwd=ROOT, env=env,
                                 operation=lambda deadline, owned: owned.process.wait())
                if code != 0 or not result.exists():
                    raise RuntimeError(f'VBench {dimension} failed with exit {code}')
            except BaseException as error:
                with (ROOT / 'BENCHMARKS.md').open('a') as stream:
                    stream.write(f'\n- **VBench {group.name} / {dimension} failed**: '
                                 f'{type(error).__name__}: {error}; '
                                 f'[log]({(logs / (dimension + ".log")).relative_to(ROOT)}).\n')
                raise
            matrix(group.parent)
    return 0


def percentile(values, fraction):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def matrix(root):
    root = Path(root).resolve()
    rows = []
    for profile, note in (
        ('q0', 'Original BF16 reference; offline generation cost, not speed denominator.'),
        ('comfy-t4-bf16-text', 'BF16 four-step streaming control; not a resident competitive baseline.'),
        ('comfy-t4-nvfp4-text', 'NVFP4-text streaming control; not Q0 or a resident baseline.'),
        ('comfy-turbo-resident', 'Resident compressed Turbo export/adapter not yet implemented.'),
        ('sglang-h3', 'Local runtime/adapter not installed; no substitute results.'),
        ('fast-h3-dense', 'Local export/runtime unverified; not declared hardware-impossible.'),
        ('fast-h3-vsa', 'Faithful released sparse kernel unavailable on SM89.'),
    ):
        group = root / profile
        attempts = records(group)
        batch_events_path = group / 'batch-events.jsonl'
        admission_stops = len(batch_events_path.read_text().splitlines()) if batch_events_path.exists() else 0
        status_path = group / 'status.json'
        generation_status = json.loads(status_path.read_text())['status'] if status_path.exists() else 'not_started'
        good = [row for row in attempts if row['status'] == 'completed']
        raw_scores = {}
        for dimension in DEVELOPMENT_DIMENSIONS:
            path = group / 'quality' / str(len(good)) / dimension / 'result.json'
            if path.exists():
                measured = json.loads(path.read_text())
                if measured['status'] == 'completed':
                    raw_scores[dimension] = measured['raw']
        summary = {
            'pipeline': profile, 'completed_clips': len(good),
            'failed_clips': len(attempts) - len(good),
            'admission_stops': admission_stops, 'generation_status': generation_status,
            # The PDF disallows scalar Qdev and p95 claims at the 4/8/16
            # development sample sizes; retain only per-metric values.
            'cold_e2e_p50_seconds': percentile([r['cold_e2e_seconds'] for r in good], .5),
            'warm_e2e_p50_seconds': None, 'warm_e2e_p95_seconds': None,
            'generation_fps_p50': percentile([124 / r['cold_e2e_seconds'] for r in good
                                              if r.get('cold_e2e_seconds')], .5),
            'sampler_p50_seconds': percentile([r.get('sampler_seconds') for r in good], .5),
            'audio_decode_p50_seconds': percentile([r.get('audio_decode_seconds') for r in good], .5),
            'video_decode_p50_seconds': percentile([r.get('video_decode_seconds') for r in good], .5),
            'sampled_gpu_used_max_mib': max((r['sampled_gpu_used_max_mib'] for r in attempts
                                           if r.get('sampled_gpu_used_max_mib') is not None), default=None),
            'sampled_host_available_min_bytes': min((r['sampled_host_available_min_bytes'] for r in attempts
                                                    if r.get('sampled_host_available_min_bytes') is not None), default=None),
            **{dimension: normalized_score(dimension, raw_scores[dimension])
               if dimension in raw_scores else None for dimension in DEVELOPMENT_DIMENSIONS},
            'note': note,
        }
        if len(good) == 1:
            summary['note'] += ' Timing is one observation, not a latency distribution.'
        rows.append(summary)
        if len(raw_scores) == 7:
            write_json(group / 'scores.json', {'label': 'VBench development subset; NOT official VBench total',
                                             'completed_clips': len(good), 'raw': raw_scores,
                                             'normalized': {dimension: normalized_score(dimension, raw_scores[dimension])
                                                            for dimension in raw_scores}})
    destination = ROOT / 'results/stage1'
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'benchmark_matrix.json', rows)
    with (destination / 'benchmark_matrix.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Stage 1 benchmark matrix', '',
             'Live measurements; missing values mean **not measured**, not zero. No baseline selected.',
             'No scalar Qdev, official VBench total, or p95 is reported at the 4/8/16 development sample sizes.',
             'Retain the PDF’s proposed ≤5-point per-dimension Q0 gate; scalar scores cannot hide a failing dimension.',
             'Cold timings include server startup through observed history/output materialization. Warm timings are not yet measured.',
             '[Full numeric matrix](benchmark_matrix.csv) · [JSON](benchmark_matrix.json)', '',
             '| Pipeline | Clips / native failures / admission stops | Cold E2E median (s) | Outcome / limitation |',
             '|---|---:|---:|---|']
    def display(value):
        return '—' if value is None else f'{value:.3f}'
    for row in rows:
        lines.append(f"| {row['pipeline']} | {row['completed_clips']} / {row['failed_clips']} / {row['admission_stops']} | "
                     f"{display(row['cold_e2e_p50_seconds'])} | {row['note']} |")
    (destination / 'benchmark_matrix.md').write_text('\n'.join(lines) + '\n')
    return rows


def request_stop(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    generate_parser = sub.add_parser('generate')
    generate_parser.add_argument('--root', type=Path, required=True)
    generate_parser.add_argument('--profile', choices=['q0', 'comfy-t4-bf16-text', 'comfy-t4-nvfp4-text'], required=True)
    generate_parser.add_argument('--count', type=int, default=1)
    generate_parser.add_argument('--allocation', choices=['filter', 'shortlist'], default='filter')
    generate_parser.add_argument('--timeout', type=float, default=3600)
    generate_parser.add_argument('--manifest', type=Path, default=ROOT / 'data/stage1/vbench_dev_manifest.json')
    scorer = sub.add_parser('score-dimension')
    scorer.add_argument('--group', type=Path, required=True)
    scorer.add_argument('--dimension', choices=DEVELOPMENT_DIMENSIONS, required=True)
    scorer.add_argument('--manifest', type=Path, default=ROOT / 'data/stage1/vbench_dev_manifest.json')
    score_parser = sub.add_parser('score')
    score_parser.add_argument('--group', type=Path, required=True)
    score_parser.add_argument('--timeout', type=float, default=3600)
    score_parser.add_argument('--manifest', type=Path, default=ROOT / 'data/stage1/vbench_dev_manifest.json')
    table = sub.add_parser('matrix')
    table.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'generate':
        signal.signal(signal.SIGTERM, request_stop)
        return generate(args)
    if args.command == 'score':
        return score_group(args)
    if args.command == 'score-dimension':
        return score_dimension(args)
    matrix(args.root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
