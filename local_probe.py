"""Thin legacy entrypoint; implementation lives in ``miniacc_core``."""
from __future__ import annotations

from pathlib import Path
import signal

from miniacc_core.cli import probe_main as _main
from miniacc_core.config import CandidateConfig
from miniacc_core.evaluation import ffprobe, validate_media
from miniacc_core.models import build_prompt_graph as _build_graph, post_json
from miniacc_core.probe import (
    MediaValidationError,
    OwnedProcess,
    OwnedProcessError,
    ProcessWatchdog,
    ResourceGuard,
    _exclusive_path,
    _write_report,
    comfy_server_command,
    create_run_dir,
    forward_events,
    headroom_violation,
    output_paths,
    run_one,
    run_owned_probe,
    resource_snapshot,
    wait_for_history,
    wait_for_server,
)
from miniacc_core.runtime import ResourceGuardViolation, run_owned_process
from miniacc_core.runtime import (
    DEFAULT_EXECUTABLE_ROOT,
    MIN_EXECUTABLE_FREE_BYTES,
    MIN_HOST_HEADROOM_BYTES,
    MIN_PROJECT_FREE_BYTES,
    MIN_VRAM_RESERVE_MIB,
)
from miniacc_core.data import load_manifest

ROOT = Path(__file__).resolve().parent


def build_prompt_graph(
    prompt,
    seed,
    *,
    diffusion_name,
    clip_name,
    video_vae_name,
    audio_vae_name,
    video_tile_size=512,
    video_tile_overlap=64,
    video_temporal_size=64,
    video_temporal_overlap=8,
):
    """Preserve the historical graph helper without private patch plumbing."""
    candidate = CandidateConfig(
        diffusion_name=diffusion_name,
        clip_name=clip_name,
        video_vae_name=video_vae_name,
        audio_vae_name=audio_vae_name,
    )
    return _build_graph(
        prompt,
        seed,
        candidate=candidate,
        video_tile_size=video_tile_size,
        video_tile_overlap=video_tile_overlap,
        video_temporal_size=video_temporal_size,
        video_temporal_overlap=video_temporal_overlap,
    )


def main(argv=None):
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
