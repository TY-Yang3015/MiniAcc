"""Owned-process and resource-admission contracts.

This module owns process groups, resource checks and watchdogs. It does not
select models or score outputs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
GPU_QUERY = (
    "nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,"
    "memory.free,driver_version,power.limit --format=csv,noheader,nounits"
)
MIN_VRAM_RESERVE_MIB = 2048
MIN_HOST_HEADROOM_BYTES = 8 * 1024**3
MIN_PROJECT_FREE_BYTES = 100 * 1024**3
MIN_EXECUTABLE_FREE_BYTES = 15 * 1024**3
DEFAULT_EXECUTABLE_ROOT = Path.home() / ".cache/miniacc"


def _run(args, timeout=15):
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return "", str(error), 127
    return result.stdout, result.stderr, result.returncode


def collect_forward_events(raw_log):
    """Aggregate only explicit runtime forward markers from an owned log."""
    result = {
        "attempts": 0,
        "completions": 0,
        "errors": 0,
        "observed": False,
        "source": str(raw_log) if raw_log else None,
    }
    if raw_log is None or not Path(raw_log).exists():
        return result
    for line in Path(raw_log).read_text(errors="replace").splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            event = json.JSONDecoder().raw_decode(line[start:])[0]
        except json.JSONDecodeError:
            continue
        key = {
            "miniacc_dit_forward_start": "attempts",
            "miniacc_dit_forward_end": "completions",
            "miniacc_dit_forward_error": "errors",
        }.get(event.get("event"))
        if key:
            result[key] += 1
            result["observed"] = True
    return result


def resource_snapshot():
    stdout, stderr, code = _run(GPU_QUERY.split())
    gpus = []
    if code == 0:
        for row in csv.reader(stdout.splitlines()):
            if len(row) != 8:
                continue
            values = [value.strip() for value in row]
            try:
                total, used, free = (int(float(values[index])) for index in (3, 4, 5))
            except ValueError:
                continue
            gpus.append(
                {
                    "index": int(values[0]),
                    "name": values[1],
                    "compute_capability": values[2],
                    "memory_total_mib": total,
                    "memory_used_mib": used,
                    "memory_free_mib": free,
                    "driver_version": values[6],
                    "power_limit_w_reported": values[7],
                }
            )

    available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break

    try:
        project = shutil.disk_usage(ROOT).free
    except OSError:
        project = None
    executable = Path(
        os.environ.get("MINIACC_EXECUTABLE_ROOT", DEFAULT_EXECUTABLE_ROOT)
    )
    try:
        executable_free = shutil.disk_usage(
            executable if executable.exists() else executable.parent
        ).free
    except OSError:
        executable_free = None
    return {
        "timestamp_monotonic": time.monotonic(),
        "gpu_query_returncode": code,
        "gpu_query_stderr": stderr.strip() or None,
        "gpus": gpus,
        "host_available_bytes": available,
        "project_free_bytes": project,
        "executable_free_bytes": executable_free,
        "executable_root": str(executable),
    }


def headroom_violation(snapshot, gpu_index=0, budget=None):
    budget = budget or ResourceBudget(gpu_index=gpu_index)
    gpus = snapshot.get("gpus")
    if not isinstance(gpus, list):
        return "GPU telemetry is unavailable"
    gpu = next((item for item in gpus if item.get("index") == gpu_index), None)
    if gpu is None:
        return f"GPU index {gpu_index} is not observable"
    if gpu.get("memory_free_mib") is None:
        return "GPU free-memory telemetry is unavailable"
    if gpu["memory_free_mib"] < budget.min_vram_reserve_mib:
        return f"GPU free memory {gpu['memory_free_mib']} MiB is below {budget.min_vram_reserve_mib} MiB reserve"
    for key, label, limit in (
        ("host_available_bytes", "host available", budget.min_host_headroom_bytes),
        ("project_free_bytes", "project free space", budget.min_project_free_bytes),
        (
            "executable_free_bytes",
            "executable free space",
            budget.min_executable_free_bytes,
        ),
    ):
        value = snapshot.get(key)
        if value is None:
            return f"{label} telemetry is unavailable"
        if value < limit:
            return f"{label} {value} bytes is below {limit} bytes reserve"
    return None


@dataclass(frozen=True)
class ResourceBudget:
    gpu_index: int = 0
    min_vram_reserve_mib: int = MIN_VRAM_RESERVE_MIB
    min_host_headroom_bytes: int = MIN_HOST_HEADROOM_BYTES
    min_project_free_bytes: int = MIN_PROJECT_FREE_BYTES
    min_executable_free_bytes: int = MIN_EXECUTABLE_FREE_BYTES


class ResourceGuardViolation(RuntimeError):
    pass


class ResourceGuard:
    def __init__(self, snapshot_fn=resource_snapshot, budget=None, sample_log=None):
        self.snapshot_fn = snapshot_fn
        self.budget = budget or ResourceBudget()
        self.last_snapshot = None
        self.sample_log = sample_log
        self._sample_lock = threading.Lock()
        if sample_log is not None:
            Path(sample_log).open("x").close()

    def check(self):
        snapshot = self.snapshot_fn()
        with self._sample_lock:
            self.last_snapshot = snapshot
            if self.sample_log is not None:
                with Path(self.sample_log).open("a") as stream:
                    stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
        violation = headroom_violation(snapshot, self.budget.gpu_index, self.budget)
        if violation:
            raise ResourceGuardViolation(violation)
        return snapshot


class OwnedProcessError(RuntimeError):
    pass


class OwnedProcess:
    """Own a process group and terminate every descendant on bounded failure."""

    def __init__(self, command, raw_log, *, env=None, cwd=None):
        self.command = command
        self.raw_log = Path(raw_log)
        self.env = env
        self.cwd = cwd
        self.process = None
        self._stream = None

    def start(self):
        self.raw_log.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.raw_log.open("xb")
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=self.env,
                stdout=self._stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            self._stream.close()
            raise
        return self

    def terminate(self, grace_seconds=5.0):
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self.process.poll() is None:
            self.process.wait()

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.terminate()
        self.close()


class ProcessWatchdog:
    def __init__(self, owned, deadline, guard, interval=1.0):
        self.owned = owned
        self.deadline = deadline
        self.guard = guard
        self.interval = interval
        self.failure = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._watch,
            name="miniacc-resource-watchdog",
            daemon=True,
        )
        self._started = False

    def _watch(self):
        while True:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                self.failure = OwnedProcessError(
                    "owned probe exceeded absolute deadline"
                )
                self.owned.terminate()
                return
            if self._stop.wait(min(self.interval, remaining)):
                return
            try:
                self.guard.check()
            except BaseException as error:
                self.failure = error
                self.owned.terminate()
                return

    def start(self):
        self._thread.start()
        self._started = True

    def stop(self):
        self._stop.set()
        if self._started:
            self._thread.join(timeout=10.0)


def run_owned_process(
    command, raw_log, *, timeout, guard=None, poll_seconds=1.0, env=None, cwd=None
):
    owned = OwnedProcess(command, raw_log, env=env, cwd=cwd).start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            code = owned.process.poll()
            if code is not None:
                return code
            if guard is not None:
                guard.check()
            if time.monotonic() >= deadline:
                raise OwnedProcessError(f"owned process exceeded {timeout:g}s deadline")
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    except ResourceGuardViolation as error:
        owned.terminate()
        raise OwnedProcessError(f"owned process stopped: {error}") from error
    except OwnedProcessError:
        owned.terminate()
        raise
    except BaseException:
        owned.terminate()
        raise
    finally:
        owned.close()


class RuntimeOwner:
    """Own admission, the live process group, watchdog and teardown."""

    def __init__(self, guard=None, timeout_seconds=3600.0, host=None):
        self.guard = guard or ResourceGuard()
        self.timeout_seconds = timeout_seconds
        self.host = host
        self.owned = None
        self.watchdog = None
        self.deadline = None
        self.process_started_at = None
        self.process_to_cleanup_seconds = None

    def admit(self):
        return self.guard.check()

    def run(
        self,
        command,
        raw_log,
        *,
        startup=None,
        operation=None,
        env=None,
        cwd=None,
        timeout=None,
    ):
        """Run callbacks while this owner controls every process lifetime edge."""
        self.admit()
        duration = self.timeout_seconds if timeout is None else timeout
        self.deadline = time.monotonic() + duration
        self.owned = OwnedProcess(command, raw_log, env=env, cwd=cwd)
        result = None
        raised = None
        watchdog_failure = None
        try:
            self.process_started_at = time.monotonic()
            self.process_to_cleanup_seconds = None
            self.owned.start()
            self.watchdog = ProcessWatchdog(
                self.owned, self.deadline, self.guard, interval=1.0
            )
            self.watchdog.start()
            if startup is not None:
                startup(self.deadline, self.owned)
            if operation is not None:
                result = operation(self.deadline, self.owned)
        except BaseException as error:
            raised = error
        finally:
            # Terminate even after a successful callback: server descendants are
            # owned by this object, never by the application or CLI.
            deadline_expired = time.monotonic() >= self.deadline
            cleanup_error = None
            try:
                self.owned.terminate()
            except BaseException as error:
                cleanup_error = error
            try:
                if self.watchdog is not None:
                    self.watchdog.stop()
                    watchdog_failure = self.watchdog.failure
            except BaseException as error:
                cleanup_error = cleanup_error or error
            try:
                self.owned.close()
            except BaseException as error:
                cleanup_error = cleanup_error or error
            self.process_to_cleanup_seconds = time.monotonic() - self.process_started_at
            self.owned = None
            self.watchdog = None
            self.deadline = None
        if watchdog_failure is not None:
            raise OwnedProcessError(str(watchdog_failure)) from watchdog_failure
        if isinstance(raised, TimeoutError) and deadline_expired:
            raise OwnedProcessError(
                "owned probe exceeded absolute deadline"
            ) from raised
        if raised is not None:
            raise raised
        if cleanup_error is not None:
            raise cleanup_error
        return result


def create_run_dir(root: Path, requested: Path | None = None) -> Path:
    if requested is not None:
        requested.parent.mkdir(parents=True, exist_ok=True)
        requested.mkdir()
        return requested
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="probe-", dir=root))
