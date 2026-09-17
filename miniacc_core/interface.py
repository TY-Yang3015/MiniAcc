"""Application lifecycle: the sole execution front door for one owned run."""

from __future__ import annotations

from pathlib import Path
import time

from .evaluation import MediaValidationError, output_paths
from .models import UnsupportedCapabilityError
from .runtime import OwnedProcessError, collect_forward_events, headroom_violation


class ApplicationInterface:
    """Compose manager, runtime owner, artifacts and media validation."""

    def __init__(self, *, data, manager, runtime=None, artifacts=None, evaluator=None):
        self.data = data
        self.manager = manager
        self.runtime = runtime
        self.artifacts = artifacts
        self.evaluator = evaluator
        self._loaded = False
        self._closed = False
        self._submitted = False

    def load_inference_model(self, candidate=None, *, deadline=None):
        if self._closed:
            raise RuntimeError("application is closed")
        manager_candidate = getattr(self.manager, "candidate", None)
        if (
            candidate is not None
            and manager_candidate is not None
            and candidate != manager_candidate
        ):
            raise ValueError("application candidate does not match manager candidate")
        try:
            if self.runtime is not None:
                self.runtime.admit()
            result = self.manager.load_inference_model(candidate, deadline=deadline)
        except BaseException:
            self.close()
            raise
        self._loaded = True
        return result

    def infer(self, jobs=None, *, output_root, guard=None, deadline=None, raw_log=None):
        """Submit, complete, persist history, validate, and return one measurement."""
        if self._closed:
            raise RuntimeError("application is closed")
        if not self._loaded:
            raise RuntimeError("load_inference_model must precede infer")
        selected = tuple(self.data.jobs() if jobs is None else jobs)
        if self._submitted:
            raise RuntimeError("application accepts only one inference submission")
        if len(selected) != 1:
            raise ValueError("owned application runs accept exactly one job")
        if guard is None and self.runtime is not None:
            guard = getattr(self.runtime, "guard", None)
        job = selected[0]
        before = None
        started = time.monotonic()
        try:
            before = guard.check() if guard is not None else None
            submission = self.manager.infer(
                self.data.prompt_for(job), job["seed"], deadline=deadline
            )
            self._submitted = True
            if self.artifacts is not None:
                graph = submission.get("graph")
                if graph is not None:
                    self.artifacts.write_json(
                        "submitted-graph.json",
                        {
                            "graph": graph,
                            "graph_sha256": submission.get("graph_sha256"),
                        },
                    )
                self.artifacts.write_json(
                    "application-submissions.json",
                    {"results": [{"job": job, **submission}]},
                )
            history = self.manager.wait_for_history(
                submission["prompt_id"],
                _history_timeout(self.runtime, deadline),
                guard=guard,
                deadline=deadline,
            )
            if self.artifacts is not None:
                self.artifacts.write_history(history)
            completed_at = time.monotonic()
            after = guard.check() if guard is not None else None
            status = history.get("status", {})
            if status.get("status_str") != "success" or not status.get(
                "completed", False
            ):
                raise RuntimeError(f"ComfyUI execution failed: {status}")
            paths = output_paths(history, Path(output_root))
            if not paths:
                raise MediaValidationError(
                    "runtime reported success but returned no materialized output"
                )
            if self.evaluator is None:
                raise RuntimeError("no media evaluator was injected")
            validated = []
            try:
                for path in paths:
                    if guard is not None:
                        guard.check()
                    validated.append(self.evaluator.validate(path, deadline=deadline))
            except MediaValidationError:
                raise
            except TimeoutError as error:
                if deadline is not None and time.monotonic() >= deadline:
                    raise OwnedProcessError(
                        "owned probe exceeded absolute deadline"
                    ) from error
                raise MediaValidationError(str(error)) from error
            except (OSError, ValueError, RuntimeError) as error:
                raise MediaValidationError(str(error)) from error
            events = collect_forward_events(raw_log)
            violation = None
            if after is not None:
                violation = headroom_violation(
                    after,
                    guard.budget.gpu_index if guard is not None else 0,
                    guard.budget if guard is not None else None,
                )
            result = {
                "job": job,
                "prompt_id": submission["prompt_id"],
                "graph": submission.get("graph"),
                "graph_sha256": submission.get("graph_sha256"),
                "elapsed_seconds": completed_at - started,
                "completion_monotonic": completed_at,
                "resource_before": before,
                "resource_after": after,
                "headroom_violation_after": violation,
                "history_status": status,
                "outputs": validated,
                "expected_forward_count": None,
                "forward_events": events,
                "actual_forward_count": (
                    events["completions"] if events["observed"] else None
                ),
                "actual_forward_count_note": "Only explicit runtime forward markers count; configured NFE and sampler progress are not used as a proxy.",
            }
            return [result]
        except BaseException:
            self.close()
            raise

    def run(
        self,
        command,
        raw_log,
        *,
        output_root,
        jobs,
        timeout=None,
        readiness_log=None,
        expected_run_id=None,
        env=None,
        cwd=None,
    ):
        """Own the server and route startup/inference through this interface."""
        if self.runtime is None:
            raise RuntimeError("application runtime owner is required")

        def startup(deadline, owned):
            waiter = getattr(self.manager, "wait_until_ready", None)
            if waiter is None:
                raise RuntimeError("manager does not implement runtime readiness")
            waiter(
                deadline,
                guard=self.runtime.guard,
                owned=owned,
                readiness_log=readiness_log,
                expected_run_id=expected_run_id,
            )

        def operation(deadline, _owned):
            self.load_inference_model(self.manager.candidate, deadline=deadline)
            return self.infer(
                jobs,
                output_root=output_root,
                guard=self.runtime.guard,
                deadline=deadline,
                raw_log=raw_log,
            )

        try:
            return self.runtime.run(
                command,
                raw_log,
                startup=startup,
                operation=operation,
                env=env,
                cwd=cwd,
                timeout=timeout,
            )
        finally:
            self.close()

    def record_history(self, history):
        """Compatibility hook for the legacy probe adapter."""
        if self.artifacts is None:
            return None
        return self.artifacts.write_history(history)

    def validate(self, path, *, deadline=None):
        if self._closed:
            raise RuntimeError("application is closed")
        if self.evaluator is None:
            raise RuntimeError("no media evaluator was injected")
        try:
            return self.evaluator.validate(path, deadline=deadline)
        except MediaValidationError:
            raise
        except TimeoutError as error:
            if deadline is not None and time.monotonic() >= deadline:
                raise OwnedProcessError(
                    "owned probe exceeded absolute deadline"
                ) from error
            raise MediaValidationError(str(error)) from error
        except (OSError, ValueError, RuntimeError) as error:
            raise MediaValidationError(str(error)) from error

    def fit(self, *args, **kwargs):
        raise UnsupportedCapabilityError("training fit is deferred and unavailable")

    def close(self):
        if not self._closed:
            try:
                self.manager.close()
            finally:
                self._closed = True
                self._loaded = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _history_timeout(runtime, deadline):
    if deadline is not None:
        return max(0.0, deadline - time.monotonic())
    return (
        getattr(runtime, "timeout_seconds", 3600.0) if runtime is not None else 3600.0
    )
