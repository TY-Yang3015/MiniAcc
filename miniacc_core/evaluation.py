"""CPU-only media validation; this layer never loads a model or allocates CUDA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time


class MediaValidationError(RuntimeError):
    """Output discovery or media validation failed."""


def _remaining(deadline: float | None, maximum: float) -> float:
    if deadline is None:
        return maximum
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("probe deadline expired during media validation")
    return min(maximum, remaining)


def sha256_file(path: Path, *, deadline=None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("probe deadline expired while hashing media")
            digest.update(block)
    return digest.hexdigest()


def _run(args, *, timeout=15):
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return "", str(error), 127
    return result.stdout, result.stderr, result.returncode


def output_paths(history: dict, output_root: Path) -> list[Path]:
    paths = []
    root = Path(output_root).resolve()
    for node in history.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            for item in node.get(key, []):
                filename = item.get("filename")
                if not filename or item.get("type", "output") != "output":
                    continue
                if key == "images" and Path(filename).suffix.lower() not in {
                    ".mp4",
                    ".mkv",
                    ".webm",
                }:
                    continue
                path = Path(output_root) / item.get("subfolder", "") / filename
                if not path.resolve().is_relative_to(root):
                    raise MediaValidationError(
                        "reported media path escapes the owned output directory"
                    )
                if path not in paths:
                    paths.append(path)
    return paths


def ffprobe(path: Path, *, deadline=None):
    stdout, stderr, code = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        timeout=_remaining(deadline, 60.0),
    )
    if code:
        raise RuntimeError(f"ffprobe failed for {path}: {stderr.strip()}")
    return json.loads(stdout)


def validate_media(path: Path, expected: dict, *, deadline=None, hash_media=True):
    path = Path(path)
    probe = ffprobe(path, deadline=deadline)
    streams = probe.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise ValueError("output must contain both video and audio streams")
    checks = {
        "width": video.get("width") == expected["width"],
        "height": video.get("height") == expected["height"],
        "frames": int(video.get("nb_frames", -1)) == expected["frames"],
        "fps": video.get("avg_frame_rate") == f"{expected['fps']}/1",
        "audio_channels": audio.get("channels") == expected["audio_channels"],
        "audio_rate": int(audio.get("sample_rate", -1)) == expected["audio_rate"],
    }
    _, decode_stderr, decode_code = _run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        timeout=_remaining(deadline, 120.0),
    )
    result = {
        "path": str(path),
        "sha256": sha256_file(path, deadline=deadline) if hash_media else None,
        "checks": checks,
        "finite_decode": decode_code == 0,
        "decode_stderr": decode_stderr.strip() or None,
        "ffprobe": probe,
    }
    if not all(checks.values()) or decode_code:
        raise ValueError(json.dumps({"media_validation": result}, sort_keys=True))
    return result


class MediaValidator:
    def __init__(self, expected, *, hash_media=True):
        self.expected = expected
        self.hash_media = hash_media

    def validate(self, path, *, deadline=None):
        return validate_media(
            Path(path), self.expected, deadline=deadline, hash_media=self.hash_media
        )
