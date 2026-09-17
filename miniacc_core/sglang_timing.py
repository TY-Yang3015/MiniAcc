"""Small CPU-only helpers for corrected SGLang caller timing.

These helpers are deliberately independent of the SGLang runtime.  They mark
only the caller interval and validate persisted records; they never synchronize
CUDA or inspect/reuse generated media.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Append one durable-enough, flushed JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()


def execute_caller_request(
    record: dict[str, Any],
    output_path: Path,
    *,
    submit: Callable[[], dict[str, Any]],
    poll: Callable[[str], dict[str, Any]],
    materialize: Callable[[str, Path], None],
    clock: Callable[[], float],
    on_update: Callable[[dict[str, Any]], None] | None = None,
    deadline_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Run the native caller request path through output materialization.

    Callbacks are the real HTTP submit/poll/download entry points in the
    launcher, while CPU tests can provide a stub transport. Validation is not
    a callback here and therefore cannot extend the returned interval.
    """
    started = float(clock())
    record["request_start_monotonic"] = started
    if on_update is not None:
        on_update(record)
    created = submit()
    request_id = created.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError("missing video id")
    record["submitted"] = created
    record["server_request_id"] = request_id
    if on_update is not None:
        on_update(record)
    while True:
        state = poll(request_id)
        status = state.get("status")
        if status in ("completed", "failed"):
            break
        if float(clock()) - started > deadline_seconds:
            raise TimeoutError("request deadline expired")
    record["status"] = status
    record["server_result"] = state
    if status != "completed":
        raise RuntimeError(str(state))
    materialize(request_id, output_path)
    complete_caller_timing(record, started, output_path, clock=clock)
    return record


def complete_caller_timing(
    record: dict[str, Any],
    started: float,
    output_path: Path,
    *,
    clock: Callable[[], float],
) -> dict[str, Any]:
    """Close the caller interval immediately after output materialization.

    The caller interval starts before request submission and ends after the
    downloaded output has been written.  Validation must happen after this
    function returns.  A missing output is recorded as an incomplete request,
    rather than silently treating validation time as generation time.
    """
    ended = float(clock())
    materialized = output_path.is_file()
    if not math.isfinite(float(started)) or not math.isfinite(ended) or ended < started:
        raise ValueError("caller timing clocks must be finite and monotonic")
    record.update(
        {
            "request_start_monotonic": float(started),
            "output_materialized_monotonic": ended if materialized else None,
            "request_end_monotonic": ended,
            "request_wall_seconds_before_validation": ended - float(started),
            "output_materialized": materialized,
            "caller_timing_boundary": "request submission through output materialization before ffprobe/ffmpeg validation",
        }
    )
    return record


def initial_setup_seconds(server_process_launch: dict[str, Any], application_start_epoch: float) -> float:
    """Compute setup from the immediate process-launch marker, not script start."""
    started = float(server_process_launch["wall_epoch"])
    ended = float(application_start_epoch)
    if not math.isfinite(started) or not math.isfinite(ended) or ended < started:
        raise ValueError("server setup clocks must be finite and monotonic")
    return ended - started


def validate_caller_timing(record: dict[str, Any]) -> None:
    """Reject malformed caller records before they enter an aggregate."""
    start = record.get("request_start_monotonic")
    end = record.get("request_end_monotonic")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (start, end)):
        raise ValueError("missing or non-finite caller monotonic clocks")
    if end < start or not record.get("output_materialized"):
        raise ValueError("caller record does not contain a materialized monotonic interval")
    if not isinstance(record.get("server_request_id"), str) or not record["server_request_id"]:
        raise ValueError("caller record is missing server request correlation id")


def read_mem_available_bytes(meminfo: Path = Path("/proc/meminfo")) -> int:
    """Read Linux MemAvailable without confusing it with free memory."""
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) * 1024
    raise RuntimeError(f"MemAvailable is unavailable in {meminfo}")


def verified_owned_process(pid: int, executable: str, proc_root: Path = Path("/proc")) -> bool:
    """Return true only for a live process with the expected command path."""
    if pid <= 0:
        return False
    try:
        cmdline = (proc_root / str(pid) / "cmdline").read_bytes().decode(errors="replace").replace("\x00", " ")
    except OSError:
        return False
    return bool(cmdline and executable in cmdline and " serve " in f" {cmdline} ")
