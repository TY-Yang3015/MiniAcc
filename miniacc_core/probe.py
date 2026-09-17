"""Legacy ComfyUI probe helpers; production execution is owned by ApplicationInterface."""

from __future__ import annotations
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .config import WorkloadConfig
from .evaluation import MediaValidationError, output_paths, validate_media
from .models import build_prompt_graph, post_json
from .runtime import (
    OwnedProcess,
    OwnedProcessError,
    ProcessWatchdog,
    ResourceGuard,
    collect_forward_events,
    create_run_dir,
    headroom_violation,
    resource_snapshot,
)


def wait_for_history(
    base_url, prompt_id, timeout, poll_seconds=2.0, guard=None, absolute_deadline=None
):
    deadline = (
        min(time.monotonic() + timeout, absolute_deadline)
        if absolute_deadline is not None
        else time.monotonic() + timeout
    )
    while time.monotonic() < deadline:
        if guard is not None:
            guard.check()
        try:
            with urlopen(
                f"{base_url.rstrip('/')}/history/{prompt_id}",
                timeout=min(30.0, max(0.1, deadline - time.monotonic())),
            ) as response:
                history = json.loads(response.read())
            if prompt_id in history:
                return history[prompt_id]
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(
        f"ComfyUI history did not complete within {timeout:g}s: {prompt_id}"
    )


# Legacy callers retain this name; production uses the same runtime collector.
forward_events = collect_forward_events


def _bounded_timeout(deadline, maximum):
    if deadline is None:
        return maximum
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("probe deadline expired during media validation")
    return min(maximum, remaining)


def run_one(
    base_url,
    job,
    graph,
    output_root,
    timeout,
    *,
    guard=None,
    raw_log=None,
    deadline=None,
    history_path=None,
    application=None,
):
    before = guard.check() if guard is not None else resource_snapshot()
    violation = headroom_violation(before)
    if violation:
        raise RuntimeError(f"admission rejected: {violation}")
    started = time.monotonic()
    effective_deadline = deadline if deadline is not None else started + timeout
    if time.monotonic() >= effective_deadline:
        raise TimeoutError("probe deadline expired before prompt submission")
    if application is not None:
        application.load_inference_model(deadline=effective_deadline)
        return application.infer(
            (job,),
            output_root=output_root,
            guard=guard,
            deadline=effective_deadline,
            raw_log=raw_log,
        )[0]
    response = post_json(
        f"{base_url.rstrip('/')}/prompt",
        {"client_id": "miniacc_local_probe", "prompt": graph},
        timeout=min(30.0, max(0.1, effective_deadline - time.monotonic())),
    )
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {response}")
    history = wait_for_history(
        base_url,
        prompt_id,
        timeout,
        guard=guard,
        absolute_deadline=effective_deadline,
    )
    if history_path is not None:
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(history_path).open("x") as stream:
            stream.write(json.dumps(history, indent=2) + "\n")
    elapsed = time.monotonic() - started
    if guard is not None:
        guard.check()
    after = guard.check() if guard is not None else resource_snapshot()
    violation = headroom_violation(after)
    status = history.get("status", {})
    if status.get("status_str") != "success" or not status.get("completed", False):
        raise RuntimeError(f"ComfyUI execution failed: {status}")
    outputs = output_paths(history, Path(output_root))
    if not outputs:
        raise MediaValidationError(
            "ComfyUI reported success but returned no materialized output"
        )
    expected = WorkloadConfig().media_expectation()
    media = []
    try:
        for path in outputs:
            if guard is not None:
                guard.check()
            if time.monotonic() >= effective_deadline:
                raise TimeoutError("probe deadline expired during media validation")
            media.append(validate_media(path, expected, deadline=effective_deadline))
    except (OSError, ValueError, RuntimeError) as error:
        raise MediaValidationError(str(error)) from error
    events = forward_events(raw_log)
    return {
        "job": job,
        "prompt_id": prompt_id,
        "elapsed_seconds": elapsed,
        "resource_before": before,
        "resource_after": after,
        "headroom_violation_after": violation,
        "history_status": status,
        "outputs": media,
        "expected_forward_count": None,
        "forward_events": events,
        "actual_forward_count": events["completions"] if events["observed"] else None,
        "actual_forward_count_note": "Only explicit runtime forward markers count; configured NFE and sampler progress are not used as a proxy.",
    }


def comfy_server_command(
    python,
    comfy_main,
    *,
    output_root,
    extra_model_paths,
    bootstrap=None,
    port=8188,
    allocator_gib=16.0,
    cpu_threads=8,
    vae_device="cpu",
    file_backed_dit=False,
    file_backed_text=False,
    hardware_config=None,
):
    if bootstrap is None:
        bootstrap = Path(__file__).resolve().parents[1] / "miniacc_bootstrap.py"
    if vae_device not in {"cpu", "gpu"}:
        raise ValueError("vae_device must be cpu or gpu")
    vae_args = ["--cpu-vae"] if vae_device == "cpu" else []
    profile = hardware_config
    if profile is not None:
        from .config import HardwareConfig
        if not isinstance(profile, HardwareConfig):
            raise TypeError("hardware_config must be a HardwareConfig")
        if profile.parallel_backend == "fsdp_inference_experimental":
            raise ValueError(
                "hardware profile backend fsdp_inference_experimental is experimental and unsupported by Comfy probe"
            )
        if profile.parallel_backend not in {"none", "independent_jobs"}:
            raise ValueError(f"unsupported hardware profile backend: {profile.parallel_backend}")
        allocator_gib = profile.allocator_gib
    command = [
        python,
        str(bootstrap),
        "--comfy-main",
        str(comfy_main),
        "--allocator-gib",
        str(allocator_gib),
        "--cpu-threads",
        str(cpu_threads),
        *( ["--file-backed-dit"] if file_backed_dit else [] ),
        *( ["--file-backed-text"] if file_backed_text else [] ),
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--disable-auto-launch",
        *( ["--runtime-profile", profile.runtime_profile] if profile is not None else [] ),
        *( ["--novram"] if profile is None or profile.vram_mode == "novram" else [] ),
        *vae_args,
        *( [] if profile is not None and profile.vram_mode == "normal" else ["--disable-smart-memory"] ),
        "--cache-none",
        "--disable-async-offload",
        *( ["--disable-pinned-memory"] if file_backed_dit or file_backed_text else [] ),
        "--disable-dynamic-vram",
        "--disable-cuda-malloc",
        "--reserve-vram",
        str(profile.vram_headroom_gib if profile is not None else 2.0),
        "--extra-model-paths-config",
        str(extra_model_paths),
        "--output-directory",
        str(output_root),
    ]
    return command


def wait_for_server(
    base_url,
    deadline,
    guard=None,
    *,
    owned=None,
    readiness_log=None,
    expected_run_id=None,
):
    while time.monotonic() < deadline:
        if (
            owned is not None
            and getattr(owned, "process", None) is not None
            and owned.process.poll() is not None
        ):
            raise RuntimeError(
                f"ComfyUI server exited during startup with code {owned.process.poll()}"
            )
        if guard is not None:
            guard.check()
        try:
            with urlopen(
                f"{base_url.rstrip('/')}/system_stats",
                timeout=min(5.0, max(0.1, deadline - time.monotonic())),
            ):
                if readiness_log is not None:
                    ready = False
                    for line in (
                        Path(readiness_log).read_text(errors="replace").splitlines()
                    ):
                        start = line.find("{")
                        if start < 0:
                            continue
                        try:
                            event = json.JSONDecoder().raw_decode(line[start:])[0]
                        except json.JSONDecodeError:
                            continue
                        if event.get("event") == "miniacc_checkpoint_hook_ready" and (
                            expected_run_id is None
                            or event.get("run_id") == expected_run_id
                        ):
                            ready = True
                            break
                    if not ready:
                        raise RuntimeError(
                            "ComfyUI HTTP server is up before checkpoint hook readiness"
                        )
                return
        except (HTTPError, URLError, TimeoutError):
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(
        "ComfyUI server did not become ready before the absolute deadline"
    )


def run_owned_probe(
    command,
    raw_log,
    *,
    base_url,
    job,
    graph,
    output_root,
    timeout,
    guard=None,
    env=None,
    expected_run_id=None,
    history_path=None,
    application=None,
):
    if guard is None:
        raise ValueError("owned probe requires a ResourceGuard")
    guard.check()
    deadline = time.monotonic() + timeout
    owned = OwnedProcess(command, raw_log, env=env).start()
    watchdog = ProcessWatchdog(owned, deadline, guard, interval=1.0)
    result = None
    raised = None
    try:
        watchdog.start()
        wait_for_server(
            base_url,
            deadline,
            guard=guard,
            owned=owned,
            readiness_log=raw_log,
            expected_run_id=expected_run_id,
        )
        result = run_one(
            base_url,
            job,
            graph,
            output_root,
            timeout,
            guard=guard,
            raw_log=raw_log,
            deadline=deadline,
            history_path=history_path,
            application=application,
        )
    except BaseException as error:
        raised = error
    finally:
        try:
            if application is not None:
                application.close()
        finally:
            owned.terminate()
            watchdog.stop()
            owned.close()
    if watchdog.failure is not None:
        raise OwnedProcessError(str(watchdog.failure)) from watchdog.failure
    if raised is not None:
        if isinstance(raised, TimeoutError) and time.monotonic() >= deadline:
            raise OwnedProcessError(
                "owned probe exceeded absolute deadline"
            ) from raised
        raise raised
    return result


def _exclusive_path(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing existing/conflicting evidence path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_report(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
