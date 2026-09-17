"""Integration contracts for the new application lifecycle (no model execution)."""

from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import URLError

from miniacc_core.artifacts import ArtifactStore
from miniacc_core.config import (
    CandidateConfig,
    WorkloadConfig,
    HISTORICAL_BASE_DIFFUSION,
    COMFY_TURBO_CLIP,
    COMFY_NVFP4_CLIP,
)
from miniacc_core import cli
from miniacc_core.evaluation import MediaValidationError, MediaValidator
from miniacc_core.interface import ApplicationInterface
from miniacc_core.models import CandidateCatalog, ComfyBaseManager, build_prompt_graph
from miniacc_core.runtime import RuntimeOwner, collect_forward_events
from miniacc_core.probe import run_one, wait_for_history
from miniacc_core.registry import build_candidate_registry
import shlex


ROOT = Path(__file__).resolve().parents[1]
MEDIA = (
    ROOT
    / ".local/artifacts/run05-preflight-EsKjhi/probe-runs/probe-plmptu9e/media/video/MiniAcc_H3_Base_00001_.mp4"
)


class OneJobData:
    def jobs(self):
        return ({"id": "job", "prompt_id": "prompt", "seed": 7},)

    def prompt_for(self, job):
        return "exact frozen prompt"


class Runtime:
    def __init__(self):
        self.admitted = 0

    def admit(self):
        self.admitted += 1
        return {"ok": True}


class FakeEvaluator:
    def __init__(self, fail=False):
        self.fail = fail
        self.paths = []

    def validate(self, path, *, deadline=None):
        self.paths.append(Path(path))
        if self.fail:
            raise ValueError("invalid media")
        return {"path": str(path), "valid": True}


class ApplicationLifecycleTests(unittest.TestCase):
    def test_cli_constructs_and_runs_the_application_interface(self):
        snapshot = {
            "gpus": [{"index": 0, "memory_free_mib": 4096}],
            "host_available_bytes": 16 * 1024**3,
            "project_free_bytes": 200 * 1024**3,
            "executable_free_bytes": 20 * 1024**3,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("main.py", "bootstrap.py", "paths.yaml"):
                (root / name).write_text("fixture")
            manager = ComfyBaseManager(
                candidate=CandidateConfig(),
                post=lambda url, value, timeout: {"prompt_id": "cli-p"},
            )
            manager._get = lambda path, deadline=None: {}
            manager.wait_until_ready = lambda *args, **kwargs: None
            manager.wait_for_history = lambda *args, **kwargs: {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "16": {"videos": [{"filename": "clip.mp4", "type": "output"}]}
                },
            }

            class FakeOwned:
                def __init__(self, command, raw_log, **kwargs):
                    self.raw_log = Path(raw_log)
                    self.process = type("Process", (), {"poll": lambda self: None})()
                    self.terminated = False
                    self.closed = False

                def start(self):
                    self.raw_log.write_text(
                        '{"event":"miniacc_dit_forward_start"}\n'
                        '{"event":"miniacc_dit_forward_end"}\n'
                    )
                    return self

                def terminate(self):
                    self.terminated = True

                def close(self):
                    self.closed = True

            class FakeWatchdog:
                failure = None

                def __init__(self, *args, **kwargs):
                    pass

                def start(self):
                    pass

                def stop(self):
                    pass

            manager.wait_until_ready = lambda *args, **kwargs: None
            manager.wait_for_history = lambda *args, **kwargs: {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "16": {"videos": [{"filename": "clip.mp4", "type": "output"}]}
                },
            }
            fake_evaluator = type(
                "Evaluator",
                (),
                {
                    "validate": lambda self, path, *, deadline=None: {
                        "path": str(path),
                        "valid": True,
                    }
                },
            )()

            with patch.object(
                cli.ModelManagerFactory, "create", return_value=manager
            ), patch.object(
                cli.ResourceGuard, "check", return_value=snapshot
            ), patch.object(
                cli, "MediaValidator", return_value=fake_evaluator
            ), patch(
                "miniacc_core.runtime.OwnedProcess", FakeOwned
            ), patch(
                "miniacc_core.runtime.ProcessWatchdog", FakeWatchdog
            ):
                code = cli.probe_main(
                    [
                        "--run-root",
                        str(root / "runs"),
                        "--output",
                        str(root / "result.json"),
                        "--python",
                        str(Path(__import__("sys").executable)),
                        "--comfy-main",
                        str(root / "main.py"),
                        "--bootstrap",
                        str(root / "bootstrap.py"),
                        "--extra-model-paths",
                        str(root / "paths.yaml"),
                    ]
                )
            result = json.loads((root / "result.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual(result["actual_forward_count"], 1)
            self.assertEqual(
                result["history_status"], {"status_str": "success", "completed": True}
            )
            self.assertTrue(result["graph_sha256"])

    def test_cli_real_media_failure_is_classified_and_history_is_retained(self):
        snapshot = {
            "gpus": [{"index": 0, "memory_free_mib": 4096}],
            "host_available_bytes": 16 * 1024**3,
            "project_free_bytes": 200 * 1024**3,
            "executable_free_bytes": 20 * 1024**3,
        }
        history = {
            "status": {"status_str": "success", "completed": True},
            "outputs": {"16": {"images": [{"filename": "clip.mp4", "type": "output"}]}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("main.py", "bootstrap.py", "paths.yaml"):
                (root / name).write_text("offline fixture")
            manager = ComfyBaseManager(
                candidate=CandidateConfig(),
                post=lambda url, value, timeout: {"prompt_id": "invalid-media"},
            )
            manager._get = lambda path, deadline=None: {}
            manager.wait_until_ready = lambda *args, **kwargs: None
            manager.wait_for_history = lambda *args, **kwargs: history

            class FakeOwned:
                instances = []

                def __init__(self, command, raw_log, **kwargs):
                    self.raw_log = Path(raw_log)
                    self.process = type("Process", (), {"poll": lambda self: None})()
                    self.terminated = False
                    self.closed = False
                    self.instances.append(self)

                def start(self):
                    self.raw_log.write_text(
                        '{"event":"miniacc_dit_forward_start"}\n'
                        '{"event":"miniacc_dit_forward_end"}\n'
                    )
                    return self

                def terminate(self):
                    self.terminated = True

                def close(self):
                    self.closed = True

            class FakeWatchdog:
                failure = None

                def __init__(self, *args, **kwargs):
                    pass

                def start(self):
                    pass

                def stop(self):
                    pass

            with patch.object(
                cli.ModelManagerFactory, "create", return_value=manager
            ), patch.object(cli.ResourceGuard, "check", return_value=snapshot), patch(
                "miniacc_core.runtime.OwnedProcess", FakeOwned
            ), patch(
                "miniacc_core.runtime.ProcessWatchdog", FakeWatchdog
            ), patch(
                "miniacc_core.evaluation.ffprobe", return_value={"streams": []}
            ):
                code = cli.probe_main(
                    [
                        "--run-dir",
                        str(root / "run"),
                        "--python",
                        str(Path(__import__("sys").executable)),
                        "--comfy-main",
                        str(root / "main.py"),
                        "--bootstrap",
                        str(root / "bootstrap.py"),
                        "--extra-model-paths",
                        str(root / "paths.yaml"),
                    ]
                )
            result = json.loads((root / "run/result.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(result["failure_class"], "output_validation_failure")
            self.assertTrue((root / "run/history.json").is_file())
            self.assertTrue(FakeOwned.instances[-1].terminated)
            self.assertTrue(FakeOwned.instances[-1].closed)

    def manager(self, post):
        manager = ComfyBaseManager(
            candidate=CandidateConfig(),
            post=post,
        )
        manager._get = lambda path, deadline=None: {"ok": True}
        return manager

    def test_cli_components_share_application_lifecycle_and_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime()
            evaluator = FakeEvaluator()
            manager = self.manager(lambda url, value, timeout: {"prompt_id": "p"})
            manager.wait_for_history = lambda *args, **kwargs: {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "16": {"videos": [{"filename": "clip.mp4", "type": "output"}]}
                },
            }
            app = ApplicationInterface(
                data=OneJobData(),
                manager=manager,
                runtime=runtime,
                artifacts=ArtifactStore(Path(directory)),
                evaluator=evaluator,
            )
            app.load_inference_model()
            result = app.infer(
                ({"id": "job", "prompt_id": "prompt", "seed": 7},),
                output_root=Path(directory) / "media",
            )
            self.assertEqual(result[0]["prompt_id"], "p")
            self.assertEqual(runtime.admitted, 1)
            self.assertTrue(
                (Path(directory) / "application-submissions.json").is_file()
            )
            self.assertEqual(app.validate("clip.mp4")["valid"], True)
            app.close()
            self.assertFalse(manager.loaded)

    def test_load_infer_and_validation_failures_close_manager(self):
        for phase in ("load", "infer", "validation"):
            with self.subTest(phase=phase):
                runtime = Runtime()
                evaluator = FakeEvaluator(fail=phase == "validation")
                post = lambda url, value, timeout: {"prompt_id": "p"}
                manager = self.manager(post)
                if phase == "load":
                    manager._get = lambda path, deadline=None: (_ for _ in ()).throw(
                        RuntimeError("startup")
                    )
                if phase == "validation":
                    manager.wait_for_history = lambda *args, **kwargs: {
                        "status": {"status_str": "success", "completed": True},
                        "outputs": {
                            "16": {
                                "videos": [{"filename": "clip.mp4", "type": "output"}]
                            }
                        },
                    }
                if phase == "infer":
                    manager._post = lambda url, value, timeout: (_ for _ in ()).throw(
                        RuntimeError("submit")
                    )
                app = ApplicationInterface(
                    data=OneJobData(),
                    manager=manager,
                    runtime=runtime,
                    evaluator=evaluator,
                )
                if phase == "load":
                    with self.assertRaisesRegex(RuntimeError, "startup"):
                        app.load_inference_model()
                else:
                    app.load_inference_model()
                    if phase == "infer":
                        with self.assertRaisesRegex(RuntimeError, "submit"):
                            app.infer(
                                ({"id": "job", "prompt_id": "prompt", "seed": 7},),
                                output_root="media",
                            )
                    else:
                        with self.assertRaisesRegex(
                            MediaValidationError, "invalid media"
                        ):
                            app.infer(
                                ({"id": "job", "prompt_id": "prompt", "seed": 7},),
                                output_root="media",
                            )
                self.assertTrue(app._closed)
                self.assertFalse(manager.loaded)

    def test_existing_native_mp4_uses_injected_validator_api(self):
        self.assertTrue(MEDIA.is_file(), MEDIA)
        result = MediaValidator(WorkloadConfig().media_expectation()).validate(MEDIA)
        self.assertTrue(result["finite_decode"])
        self.assertTrue(all(result["checks"].values()))

    def test_stalled_history_respects_absolute_deadline(self):
        with patch("miniacc_core.probe.urlopen", side_effect=URLError("offline")):
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                wait_for_history("http://unused", "p", timeout=0.02, poll_seconds=2)
            self.assertLess(time.monotonic() - started, 1.0)

    def test_new_run_one_routes_submission_and_validation_through_application(self):
        history = {
            "status": {"status_str": "success", "completed": True},
            "outputs": {
                "16": {
                    "videos": [
                        {"filename": "clip.mp4", "subfolder": "video", "type": "output"}
                    ]
                }
            },
        }
        snapshot = {
            "gpus": [{"index": 0, "memory_free_mib": 4096}],
            "host_available_bytes": 16 * 1024**3,
            "project_free_bytes": 200 * 1024**3,
            "executable_free_bytes": 20 * 1024**3,
        }
        manager = self.manager(
            lambda url, value, timeout: {"prompt_id": "application-p"}
        )
        manager.wait_for_history = lambda *args, **kwargs: history
        evaluator = FakeEvaluator()
        app = ApplicationInterface(
            data=OneJobData(), manager=manager, evaluator=evaluator
        )
        with patch(
            "miniacc_core.probe.resource_snapshot", return_value=snapshot
        ), patch("miniacc_core.probe.Path.is_file", return_value=True):
            result = run_one(
                "http://unused",
                {"id": "job", "seed": 7},
                {},
                Path("media"),
                1,
                application=app,
            )
        self.assertEqual(result["prompt_id"], "application-p")
        self.assertEqual(evaluator.paths, [Path("media/video/clip.mp4")])
        app.close()


class GraphAdapterTests(unittest.TestCase):
    def test_turbo_graph_uses_model_only_lora_and_author_schedule_fields(self):
        candidate = CandidateConfig(
            family="comfyui-turbo",
            adapter_name="turbo.safetensors",
            steps=4,
            video_shift=6.0,
            audio_shift=3.0,
        )
        graph = build_prompt_graph("verbatim", 9, candidate=candidate)
        self.assertEqual(graph["2"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(graph["2"]["inputs"]["strength_model"], 1.0)
        self.assertEqual(graph["3"]["inputs"]["shift_video"], 6.0)
        self.assertEqual(graph["10"]["inputs"]["steps"], 4)
        self.assertEqual(graph["7"]["inputs"]["clip"], ["4", 0])


class RegistryTests(unittest.TestCase):
    def test_catalog_does_not_report_comfy_assets_for_sglang(self):
        readiness = CandidateCatalog(ROOT / ".local/models").inspect(
            CandidateConfig(family="sglang-h3")
        )
        self.assertEqual(
            readiness.implementation_status, "runtime_not_installed_or_not_inspected"
        )
        self.assertEqual(readiness.available_assets, ())
        self.assertIn("runtime:sglang-h3", readiness.missing_assets)

    def test_registry_keeps_all_families_and_blockers_explicit(self):
        document = build_candidate_registry(ROOT / ".local/models")
        families = {item["family"] for item in document["candidates"]}
        self.assertEqual(
            families, {"comfyui-turbo", "sglang-h3", "fast-h3-dense", "fast-h3-vsa"}
        )
        vsa = next(
            item for item in document["candidates"] if item["family"] == "fast-h3-vsa"
        )
        self.assertEqual(vsa["eligibility"], "blocked_sm89_kernel")
        self.assertIsNone(vsa["invocation"]["command"])

    def test_comfy_names_are_from_pinned_tree_and_controls_are_distinct(self):
        tree = (
            ROOT
            / ".local/artifacts/stage1-family-registry/public-header-audit/comfy-org-tree-015.json"
        ).read_text()
        for name in (
            COMFY_TURBO_CLIP,
            "minimax_h3_fl2va_bf16.safetensors",
            "minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_audio_vae_fp32.safetensors",
        ):
            self.assertIn(name, tree)
        document = build_candidate_registry(ROOT / ".local/models")
        bf16 = next(
            item for item in document["candidates"] if item["id"] == "S1-FILT-06-CUI-T4"
        )
        nvfp4 = next(
            item
            for item in document["candidates"]
            if item["id"] == "S1-STRUCT-04-CUI-T4-NVFP4-TEXT"
        )
        self.assertEqual(bf16["config"]["clip"], COMFY_TURBO_CLIP)
        self.assertEqual(nvfp4["config"]["clip"], COMFY_NVFP4_CLIP)
        self.assertNotEqual(bf16["config"]["clip"], nvfp4["config"]["clip"])
        self.assertIn("CPU-emulated", nvfp4["config"]["residency"])
        self.assertTrue(nvfp4["config"]["file_backed_dit"])
        self.assertIn("--file-backed-dit", shlex.split(nvfp4["invocation"]["command"]))


class ReadinessCorrectionTests(unittest.TestCase):
    def test_forward_measurement_uses_markers_and_no_steps_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.raw.log"
            path.write_text(
                '{"event":"miniacc_dit_forward_start"}\n'
                '{"event":"miniacc_dit_forward_end"}\n'
            )
            measured = collect_forward_events(path)
            self.assertEqual(measured["completions"], 1)
            self.assertEqual(1 if measured["observed"] else None, 1)
            empty = collect_forward_events(Path(directory) / "missing.log")
            self.assertFalse(empty["observed"])
            self.assertIsNone(empty["completions"] if empty["observed"] else None)

    def test_turbo_defaults_are_full_bf16_and_forbidden_pairs_fail_closed(self):
        candidate = CandidateConfig(
            family="comfyui-turbo", adapter_name="released.safetensors"
        )
        self.assertNotEqual(candidate.diffusion_name, HISTORICAL_BASE_DIFFUSION)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            CandidateConfig(family="comfyui-base", adapter_name="released.safetensors")
        with self.assertRaisesRegex(ValueError, "compressed/pruned"):
            CandidateConfig(
                family="comfyui-turbo",
                adapter_name="released.safetensors",
                diffusion_name=HISTORICAL_BASE_DIFFUSION,
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            CandidateConfig(adapter_alpha=8)

    def test_registry_comfy_command_records_every_asset_and_gpu_vae(self):
        document = build_candidate_registry(ROOT / ".local/models")
        candidate = next(
            item for item in document["candidates"] if item["id"] == "S1-FILT-06-CUI-T4"
        )
        command = shlex.split(candidate["invocation"]["command"])
        flags = {
            command[index]: command[index + 1]
            for index in range(len(command) - 1)
            if command[index].startswith("--")
        }
        config = candidate["config"]
        self.assertEqual(flags["--diffusion-name"], config["base"])
        self.assertEqual(flags["--clip-name"], config["clip"])
        self.assertEqual(flags["--video-vae-name"], config["video_vae"])
        self.assertEqual(flags["--audio-vae-name"], config["audio_vae"])
        self.assertEqual(flags["--steps"], "4")
        self.assertEqual(flags["--vae-device"], "gpu")
        self.assertEqual(flags["--allocator-gib"], "16")
        self.assertEqual(flags["--timeout"], "3600")
        self.assertEqual(candidate["artifact_contract"]["status"], "implemented")
        planned = next(
            item for item in document["candidates"] if item["family"] == "sglang-h3"
        )
        self.assertEqual(
            planned["artifact_contract"]["status"], "planned_not_implemented"
        )

    def test_cli_rejects_internal_artifact_alias_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with patch.object(ApplicationInterface, "run") as runner:
                code = cli.probe_main(
                    [
                        "--run-dir",
                        str(run_dir),
                        "--output",
                        str(run_dir / "application-submissions.json"),
                    ]
                )
            self.assertEqual(code, 1)
            runner.assert_not_called()

    def test_application_completion_persists_actual_graph_and_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ComfyBaseManager(
                candidate=CandidateConfig(),
                post=lambda url, value, timeout: {"prompt_id": "complete"},
            )
            manager._get = lambda path, deadline=None: {"ok": True}
            manager.wait_for_history = (
                lambda prompt_id, timeout, guard=None, deadline=None: {
                    "status": {"status_str": "success", "completed": True},
                    "outputs": {
                        "16": {
                            "videos": [
                                {
                                    "filename": "clip.mp4",
                                    "subfolder": "video",
                                    "type": "output",
                                }
                            ]
                        }
                    },
                }
            )
            evaluator = FakeEvaluator()
            app = ApplicationInterface(
                data=OneJobData(),
                manager=manager,
                artifacts=ArtifactStore(Path(directory)),
                evaluator=evaluator,
            )
            app.load_inference_model()
            result = app.infer(output_root=Path(directory) / "media")
            self.assertEqual(
                result[0]["outputs"],
                [
                    {
                        "path": str(Path(directory) / "media/video/clip.mp4"),
                        "valid": True,
                    }
                ],
            )
            graph = json.loads((Path(directory) / "submitted-graph.json").read_text())
            self.assertEqual(graph["graph_sha256"], result[0]["graph_sha256"])
            self.assertTrue((Path(directory) / "history.json").is_file())
            with self.assertRaisesRegex(RuntimeError, "one inference submission"):
                app.infer(output_root=Path(directory) / "media")
            self.assertEqual(result[0]["prompt_id"], "complete")
            app.close()

    def test_runtime_owner_tears_down_after_success_and_failure(self):
        class FakeGuard:
            def check(self):
                return {"ok": True}

        class FakeOwned:
            def __init__(self, *args, **kwargs):
                self.terminated = False
                self.closed = False
                self.process = object()

            def start(self):
                return self

            def terminate(self):
                self.terminated = True

            def close(self):
                self.closed = True

        class FakeWatchdog:
            failure = None

            def __init__(self, *args, **kwargs):
                self.started = False
                self.stopped = False

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

        with patch(
            "miniacc_core.runtime.OwnedProcess", return_value=FakeOwned()
        ) as owned_ctor, patch("miniacc_core.runtime.ProcessWatchdog", FakeWatchdog):
            owner = RuntimeOwner(guard=FakeGuard(), timeout_seconds=1)
            self.assertEqual(
                owner.run(["server"], Path("raw.log"), operation=lambda *_: "ok"), "ok"
            )
            owned = owned_ctor.return_value
            self.assertTrue(owned.terminated)
            self.assertTrue(owned.closed)
        failing = FakeOwned()
        with patch("miniacc_core.runtime.OwnedProcess", return_value=failing), patch(
            "miniacc_core.runtime.ProcessWatchdog", FakeWatchdog
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                RuntimeOwner(guard=FakeGuard(), timeout_seconds=1).run(
                    ["server"],
                    Path("raw.log"),
                    operation=lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
                )
        self.assertTrue(failing.terminated)
        self.assertTrue(failing.closed)
        timed = FakeOwned()
        with patch("miniacc_core.runtime.OwnedProcess", return_value=timed), patch(
            "miniacc_core.runtime.ProcessWatchdog", FakeWatchdog
        ):
            with self.assertRaisesRegex(RuntimeError, "absolute deadline"):
                RuntimeOwner(guard=FakeGuard(), timeout_seconds=0).run(
                    ["server"],
                    Path("raw.log"),
                    operation=lambda *_: (_ for _ in ()).throw(TimeoutError("expired")),
                )
        self.assertTrue(timed.terminated)
        self.assertTrue(timed.closed)

    def test_deadline_is_passed_to_comfy_post(self):
        seen = []
        manager = ComfyBaseManager(
            candidate=CandidateConfig(),
            post=lambda url, value, timeout: seen.append(timeout) or {"prompt_id": "p"},
        )
        manager._get = lambda path, deadline=None: {"ok": True}
        manager.load_inference_model()
        manager.infer("prompt", 7, deadline=time.monotonic() + 0.05)
        self.assertEqual(len(seen), 1)
        self.assertGreater(seen[0], 0)
        self.assertLessEqual(seen[0], 0.05)


if __name__ == "__main__":
    unittest.main()
