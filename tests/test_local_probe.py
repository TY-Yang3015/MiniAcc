"""Offline tests for local probe graph and measurement guards."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import local_probe
import miniacc_bootstrap


ROOT = Path(__file__).resolve().parents[1]


class ProbeGraphTests(unittest.TestCase):
    def test_graph_keeps_native_workload_and_exact_prompt_seed(self):
        graph = local_probe.build_prompt_graph(
            "exact frozen prompt",
            20260909,
            diffusion_name="d.safetensors",
            clip_name="c.safetensors",
            video_vae_name="v.safetensors",
            audio_vae_name="a.safetensors",
        )
        self.assertEqual(graph["7"]["inputs"]["prompt"], "exact frozen prompt")
        self.assertEqual(graph["11"]["inputs"]["noise_seed"], 20260909)
        self.assertEqual(
            (
                graph["7"]["inputs"]["width"],
                graph["7"]["inputs"]["height"],
                graph["7"]["inputs"]["length"],
            ),
            (1344, 768, 124),
        )
        self.assertEqual(graph["10"]["inputs"]["steps"], 49)
        self.assertNotIn(
            "LoraLoaderModelOnly", {node["class_type"] for node in graph.values()}
        )
        self.assertEqual(graph["3"]["inputs"]["shift_video"], 12.0)
        self.assertEqual(graph["3"]["inputs"]["shift_audio"], 3.0)
        self.assertEqual(graph["15"]["inputs"]["fps"], 24)
        self.assertEqual(graph["13"]["class_type"], "VAEDecodeTiled")
        self.assertEqual(graph["14"]["class_type"], "VAEDecodeAudio")
        self.assertEqual(graph["14"]["inputs"], {"samples": ["12", 0], "vae": ["6", 0]})

    def test_exact_job_selection_keeps_frozen_seed_and_prompt(self):
        from miniacc_core.data import PromptDataModule
        data = PromptDataModule(ROOT / "data/stage1/vbench_dev_manifest.json")
        later = data.jobs()[32]
        self.assertEqual(data.job(later["id"]), later)
        self.assertEqual(data.job(), data.first_job())
        with self.assertRaises(ValueError):
            data.job("not-a-frozen-job")

    def test_unknown_job_is_rejected_before_any_runtime_spawn(self):
        with patch("miniacc_core.cli.ApplicationInterface.run") as run, patch("builtins.print"):
            code = local_probe.main(["--job-id", "not-a-frozen-job"])
        self.assertEqual(code, 1)
        run.assert_not_called()

    def test_manifest_first_job_is_the_probe_input(self):
        manifest = local_probe.load_manifest(
            ROOT / "data/stage1/vbench_dev_manifest.json"
        )
        job = manifest["jobs"][0]
        prompt = next(
            item["prompt_en"]
            for item in manifest["prompts"]
            if item["id"] == job["prompt_id"]
        )
        self.assertEqual(job["seed"], 20260909)
        self.assertTrue(prompt)


class RunnerMeasurementTests(unittest.TestCase):
    def test_cli_rejects_reserved_observation_paths_before_spawn(self):
        for name in ("resources.jsonl", "av-latent.safetensors", "history.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory) / "run"
                with patch("miniacc_core.cli.ApplicationInterface.run") as run, patch(
                    "miniacc_core.probe.resource_snapshot", return_value={}
                ), patch("builtins.print") as output:
                    code = local_probe.main(
                        ["--run-dir", str(run_dir), "--output", str(run_dir / name)]
                    )
                self.assertEqual(code, 1)
                run.assert_not_called()
                self.assertIn(
                    "paths conflict", json.loads(output.call_args.args[0])["error"]
                )

    def test_cli_separates_media_root_from_report_root(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "reports" / "result.json"
            for media_root in (None, Path(directory) / "server-media"):
                with self.subTest(media_root=media_root):
                    run_report = (
                        Path(directory) / f"report-{media_root is not None}.json"
                    )
                    args = [
                        "--output",
                        str(run_report),
                        "--run-root",
                        str(Path(directory) / "runs"),
                        "--python",
                        sys.executable,
                        "--comfy-main",
                        str(Path(directory) / "main.py"),
                        "--bootstrap",
                        str(Path(directory) / "bootstrap.py"),
                        "--extra-model-paths",
                        str(Path(directory) / "paths.yaml"),
                    ]
                    for path in (
                        Path(directory) / "main.py",
                        Path(directory) / "bootstrap.py",
                        Path(directory) / "paths.yaml",
                    ):
                        path.write_text("x")
                    if media_root is not None:
                        args += ["--comfy-output", str(media_root)]
                    with patch(
                        "miniacc_core.cli.ApplicationInterface.run", return_value=[{}]
                    ) as run, patch.object(
                        local_probe.ResourceGuard,
                        "check",
                        return_value={"gpus": [], "host_available_bytes": 1},
                    ), patch(
                        "builtins.print"
                    ):
                        self.assertEqual(local_probe.main(args), 0)
                    default_media = run.call_args.args[1].parent / "media"
                    self.assertEqual(
                        run.call_args.kwargs["output_root"], media_root or default_media
                    )
                    self.assertNotEqual(
                        run.call_args.kwargs["output_root"], run_report.parent
                    )

    def test_success_resolves_media_paths_against_server_output_root(self):
        history = {
            "status": {"status_str": "success", "completed": True},
            "outputs": {
                "16": {"videos": [{"filename": "clip.mp4", "subfolder": "video"}]}
            },
        }
        snapshot = {
            "gpus": [{"index": 0, "memory_free_mib": 4096}],
            "host_available_bytes": 16 * 1024**3,
            "project_free_bytes": 200 * 1024**3,
            "executable_free_bytes": 20 * 1024**3,
        }
        for key in ("videos", "gifs", "images"):
            history["outputs"]["16"] = {
                key: [{"filename": "clip.mp4", "subfolder": "video", "type": "output"}]
            }
            with self.subTest(schema=key), patch(
                "miniacc_core.probe.resource_snapshot", return_value=snapshot
            ), patch(
                "miniacc_core.probe.post_json", return_value={"prompt_id": "p"}
            ), patch(
                "miniacc_core.probe.wait_for_history", return_value=history
            ), patch(
                "miniacc_core.probe.validate_media", return_value={"validated": True}
            ) as validate:
                result = local_probe.run_one(
                    "http://127.0.0.1:8188", {"id": "job"}, {}, Path("server-output"), 1
                )
            self.assertEqual(
                validate.call_args.args[0], Path("server-output/video/clip.mp4")
            )
            self.assertEqual(result["outputs"], [{"validated": True}])

    def test_v3_output_discovery_ignores_thumbnails_temp_and_duplicates(self):
        video = {"filename": "clip.mp4", "subfolder": "video", "type": "output"}
        history = {
            "outputs": {
                "16": {
                    "images": [
                        video,
                        {"filename": "thumbnail.png", "type": "output"},
                        {"filename": "temporary.mp4", "type": "temp"},
                    ],
                    "animated": [True],
                    "videos": [video],
                }
            }
        }
        self.assertEqual(
            local_probe.output_paths(history, Path("media")),
            [Path("media/video/clip.mp4")],
        )
        self.assertEqual(local_probe.output_paths({"outputs": {}}, Path("media")), [])

    def test_output_discovery_rejects_paths_outside_owned_media(self):
        for filename in ("../outside.mp4", "/outside.mp4"):
            with self.subTest(filename=filename):
                history = {"outputs": {"16": {"images": [{"filename": filename}]}}}
                with self.assertRaises(local_probe.MediaValidationError):
                    local_probe.output_paths(history, Path("/owned-media"))

    def test_history_snapshot_survives_output_discovery_failure(self):
        history = {
            "status": {"status_str": "success", "completed": True},
            "outputs": {},
        }
        snapshot = {
            "gpus": [{"index": 0, "memory_free_mib": 4096}],
            "host_available_bytes": 16 * 1024**3,
            "project_free_bytes": 200 * 1024**3,
            "executable_free_bytes": 20 * 1024**3,
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "miniacc_core.probe.resource_snapshot", return_value=snapshot
        ), patch(
            "miniacc_core.probe.post_json", return_value={"prompt_id": "fixture"}
        ), patch(
            "miniacc_core.probe.wait_for_history", return_value=history
        ):
            path = Path(directory) / "history.json"
            with self.assertRaises(local_probe.MediaValidationError):
                local_probe.run_one(
                    "http://unused", {}, {}, Path(directory), 1, history_path=path
                )
            self.assertEqual(json.loads(path.read_text()), history)

    def test_media_validation_failure_is_not_setup_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "result.json"
            for name in ("main.py", "bootstrap.py", "paths.yaml"):
                (root / name).write_text("x")
            args = [
                "--output",
                str(report),
                "--run-root",
                str(root / "runs"),
                "--python",
                sys.executable,
                "--comfy-main",
                str(root / "main.py"),
                "--bootstrap",
                str(root / "bootstrap.py"),
                "--extra-model-paths",
                str(root / "paths.yaml"),
            ]
            with patch(
                "miniacc_core.cli.ApplicationInterface.run",
                side_effect=local_probe.MediaValidationError("bad media"),
            ), patch.object(
                local_probe.ResourceGuard,
                "check",
                return_value={"gpus": [], "host_available_bytes": 1},
            ), patch(
                "builtins.print"
            ):
                self.assertEqual(local_probe.main(args), 1)
            self.assertEqual(
                json.loads(report.read_text())["failure_class"],
                "output_validation_failure",
            )

    def test_cli_rejects_existing_evidence_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            existing.mkdir()
            with patch("miniacc_core.cli.ApplicationInterface.run") as run:
                result = local_probe.main(
                    ["--run-dir", str(existing), "--output", str(root / "result.json")]
                )
            self.assertEqual(result, 1)
            run.assert_not_called()

    def test_cli_rejects_conflicting_output_paths_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("miniacc_core.cli.ApplicationInterface.run") as run:
                result = local_probe.main(
                    [
                        "--run-root",
                        str(root / "runs"),
                        "--output",
                        str(root / "same"),
                        "--server-log",
                        str(root / "same"),
                    ]
                )
            self.assertEqual(result, 1)
            run.assert_not_called()

    def test_cli_admission_rejects_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("main.py", "bootstrap.py", "paths.yaml"):
                (root / name).write_text("x")
            args = [
                "--run-root",
                str(root / "runs"),
                "--python",
                "/bin/python",
                "--comfy-main",
                str(root / "main.py"),
                "--bootstrap",
                str(root / "bootstrap.py"),
                "--extra-model-paths",
                str(root / "paths.yaml"),
            ]
            with patch.object(
                local_probe.ResourceGuard,
                "check",
                side_effect=local_probe.ResourceGuardViolation("no telemetry"),
            ), patch("miniacc_core.cli.ApplicationInterface.run") as run, patch(
                "builtins.print"
            ):
                self.assertEqual(local_probe.main(args), 1)
            run.assert_not_called()

    def test_server_command_contains_bootstrap_and_allocator_ceiling(self):
        command = local_probe.comfy_server_command(
            "python",
            Path("main.py"),
            output_root=Path("media"),
            extra_model_paths=Path("paths.yaml"),
            bootstrap=Path("bootstrap.py"),
        )
        self.assertIn("bootstrap.py", command)
        self.assertIn("--allocator-gib", command)
        self.assertIn("16.0", command)
        self.assertIn("--disable-cuda-malloc", command)
        self.assertIn("--cpu-vae", command)

    def test_a100_hardware_profile_consumes_memory_and_vram_policy(self):
        from miniacc_core.config import load_hardware_config
        profile = load_hardware_config(Path("exp_configs/wolf8-a100-4x.yaml"))
        command = local_probe.comfy_server_command(
            "python", Path("main.py"), output_root=Path("media"),
            extra_model_paths=Path("paths.yaml"), hardware_config=profile,
            vae_device="gpu",
        )
        self.assertIn("--runtime-profile", command)
        self.assertIn("a100-independent", command)
        from dataclasses import replace
        single_gpu = replace(profile, parallel_backend="none", world_size=1)
        command = local_probe.comfy_server_command(
            "python", Path("main.py"), output_root=Path("media"),
            extra_model_paths=Path("paths.yaml"), hardware_config=single_gpu,
            vae_device="gpu",
        )
        self.assertIn("--runtime-profile", command)
        self.assertIn("a100-independent", command)
        self.assertIn("71.0", command)
        self.assertIn("8.0", command)
        self.assertNotIn("--novram", command)
        self.assertNotIn("--disable-smart-memory", command)

    def test_gpu_vae_device_omits_only_cpu_vae(self):
        command = local_probe.comfy_server_command(
            "python",
            Path("main.py"),
            output_root=Path("media"),
            extra_model_paths=Path("paths.yaml"),
            vae_device="gpu",
        )
        self.assertNotIn("--cpu-vae", command)
        self.assertIn("--novram", command)
        self.assertIn("--disable-smart-memory", command)
        self.assertIn("--disable-cuda-malloc", command)

    def test_failed_history_is_not_reported_as_success(self):
        error_history = {"status": {"status_str": "error", "completed": False}}
        with patch(
            "miniacc_core.probe.resource_snapshot",
            return_value={
                "gpus": [{"index": 0, "memory_free_mib": 4096}],
                "host_available_bytes": 16 * 1024**3,
                "project_free_bytes": 200 * 1024**3,
                "executable_free_bytes": 20 * 1024**3,
            },
        ), patch(
            "miniacc_core.probe.post_json", return_value={"prompt_id": "p"}
        ), patch(
            "miniacc_core.probe.wait_for_history", return_value=error_history
        ):
            with self.assertRaisesRegex(RuntimeError, "execution failed"):
                local_probe.run_one(
                    "http://127.0.0.1:8188", {"id": "job"}, {}, Path("."), 1
                )


class LifecycleTests(unittest.TestCase):
    def test_timeout_terminates_owned_process_group_and_keeps_raw_log(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_log = Path(directory) / "server.log"
            process = unittest.mock.MagicMock()
            process.pid = 1234
            process.poll.side_effect = [None, None, 0]
            with patch(
                "miniacc_core.runtime.subprocess.Popen", return_value=process
            ), patch("miniacc_core.runtime.os.killpg") as killpg:
                with self.assertRaisesRegex(local_probe.OwnedProcessError, "deadline"):
                    local_probe.run_owned_process(["server"], raw_log, timeout=0)
            self.assertEqual(
                killpg.call_args_list[0].args, (1234, local_probe.signal.SIGTERM)
            )
            self.assertTrue(raw_log.exists())

    def test_resource_violation_terminates_owned_process(self):
        with tempfile.TemporaryDirectory() as directory:
            process = unittest.mock.MagicMock()
            process.pid = 5678
            process.poll.return_value = None
            guard = local_probe.ResourceGuard(
                snapshot_fn=lambda: {
                    "gpus": [{"index": 0, "memory_free_mib": 1}],
                    "host_available_bytes": 16 * 1024**3,
                    "project_free_bytes": 200 * 1024**3,
                    "executable_free_bytes": 20 * 1024**3,
                }
            )
            with patch(
                "miniacc_core.runtime.subprocess.Popen", return_value=process
            ), patch("miniacc_core.runtime.os.killpg") as killpg:
                with self.assertRaisesRegex(local_probe.OwnedProcessError, "stopped"):
                    local_probe.run_owned_process(
                        ["server"],
                        Path(directory) / "server.log",
                        timeout=10,
                        guard=guard,
                    )
            self.assertEqual(
                killpg.call_args_list[0].args, (5678, local_probe.signal.SIGTERM)
            )

    def test_startup_death_is_reported_without_waiting_for_deadline(self):
        class DeadProcess:
            def poll(self):
                return 17

        with self.assertRaisesRegex(RuntimeError, "exited during startup"):
            local_probe.wait_for_server(
                "http://127.0.0.1:1",
                time.monotonic() + 60,
                owned=types.SimpleNamespace(process=DeadProcess()),
            )

    def test_http_readiness_requires_registered_hook_before_prompt(self):
        class LiveProcess:
            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text('{"event":"other"}\n')
            response = unittest.mock.MagicMock()
            with patch("miniacc_core.probe.urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "checkpoint hook readiness"):
                    local_probe.wait_for_server(
                        "http://127.0.0.1:1",
                        time.monotonic() + 60,
                        owned=types.SimpleNamespace(process=LiveProcess()),
                        readiness_log=log,
                        expected_run_id="run",
                    )

    def test_watchdog_terminates_when_http_wait_blocks(self):
        class FakeOwned:
            def __init__(self, *args, **kwargs):
                self.terminated = False

            def start(self):
                return self

            def terminate(self):
                self.terminated = True

            def close(self):
                pass

        guard = local_probe.ResourceGuard(
            snapshot_fn=lambda: {
                "gpus": [{"index": 0, "memory_free_mib": 4096}],
                "host_available_bytes": 16 * 1024**3,
                "project_free_bytes": 200 * 1024**3,
                "executable_free_bytes": 20 * 1024**3,
            }
        )
        fake = FakeOwned()
        with patch("miniacc_core.probe.OwnedProcess", return_value=fake), patch(
            "miniacc_core.probe.wait_for_server",
            side_effect=lambda *args, **kwargs: time.sleep(0.1),
        ), patch("miniacc_core.probe.run_one", return_value={}):
            with self.assertRaises(local_probe.OwnedProcessError):
                local_probe.run_owned_probe(
                    ["server"],
                    Path("raw.log"),
                    base_url="http://127.0.0.1:1",
                    job={},
                    graph={},
                    output_root=Path("media"),
                    timeout=0.02,
                    guard=guard,
                )
        self.assertTrue(fake.terminated)

    def test_forward_count_requires_explicit_runtime_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_log = Path(directory) / "raw.log"
            raw_log.write_text(
                ' 0%| | 0/49 {"event":"miniacc_dit_forward_start"}\n'
                '{"event":"miniacc_dit_forward_end"}\n'
                ' 2%| | 1/49 {"error":"failed", "event":"miniacc_dit_forward_error"}\n'
                "sampler 1/49\n"
            )
            self.assertEqual(local_probe.forward_events(raw_log)["attempts"], 1)
            self.assertEqual(local_probe.forward_events(raw_log)["completions"], 1)
            self.assertEqual(local_probe.forward_events(raw_log)["errors"], 1)


class BootstrapTests(unittest.TestCase):
    def test_registered_hooks_select_v3_execute_and_v1_decode(self):
        output = ({"samples": object()},)

        class Sampler:
            @classmethod
            def execute(cls):
                return output

            sample = execute

        class Audio:
            @classmethod
            def execute(cls, value):
                return value

            decode = execute

        class Video:
            FUNCTION = "decode"

            def decode(self, value):
                return value

        fake_nodes = types.ModuleType("nodes")
        fake_nodes.NODE_CLASS_MAPPINGS = {
            "SamplerCustomAdvanced": Sampler,
            "VAEDecodeAudio": Audio,
            "VAEDecodeTiled": Video,
        }
        with patch.dict(sys.modules, {"nodes": fake_nodes}), patch.dict(
            os.environ,
            {"MINIACC_LATENT_CHECKPOINT": "/tmp/unused-checkpoint.safetensors"},
        ), patch.object(miniacc_bootstrap, "save_av_latent") as save, patch.object(
            miniacc_bootstrap, "_emit"
        ) as emit:
            self.assertIs(miniacc_bootstrap.install_registered_hooks(), Sampler)
            self.assertIs(Sampler.sample(), output)
            self.assertIs(Audio.execute(output), output)
            self.assertIs(getattr(Video(), Video.FUNCTION)(output), output)
        save.assert_called_once_with(
            output[0]["samples"], Path("/tmp/unused-checkpoint.safetensors")
        )
        self.assertEqual(
            [call.args[0] for call in emit.call_args_list],
            [
                "miniacc_checkpoint_hook_ready",
                "miniacc_sampler_start",
                "miniacc_sampler_end",
                "miniacc_audio_decode_start",
                "miniacc_audio_decode_end",
                "miniacc_video_decode_start",
                "miniacc_video_decode_end",
            ],
        )

    def test_timing_observers_preserve_native_returns_and_emit_phase_boundaries(self):
        classes = {}
        for name, method in (
            ("UNETLoader", "load_unet"),
            ("CLIPLoader", "load_clip"),
            ("VAELoader", "load_vae"),
            ("LoraLoaderModelOnly", "load_lora_model_only"),
        ):
            def loader(self, *args, _name=name, **kwargs):
                return _name
            classes[name] = type(name, (), {method: loader})
        for name, method in (
            ("MiniMaxH3ImageToVideo", "execute"),
            ("SamplerCustomAdvanced", "execute"),
            ("VAEDecodeAudio", "execute"),
            ("VAEDecodeTiled", "decode"),
            ("CreateVideo", "execute"),
            ("SaveVideo", "execute"),
        ):
            def execute(cls, *args, _name=name, **kwargs):
                return _name
            classes[name] = type(name, (), {method: classmethod(execute)})
        fake_nodes = types.ModuleType("nodes")
        fake_nodes.NODE_CLASS_MAPPINGS = classes
        with patch.dict(sys.modules, {"nodes": fake_nodes}), patch.dict(
            os.environ, {"MINIACC_TIMING": "1"}
        ), patch.object(miniacc_bootstrap, "_emit") as emit:
            installed = miniacc_bootstrap.install_timing_observers()
            self.assertEqual(len(installed), 10)
            self.assertEqual(classes["SamplerCustomAdvanced"].execute(), "SamplerCustomAdvanced")
            vae_loader = classes["VAELoader"]()
            self.assertEqual(vae_loader.load_vae("video"), "VAELoader")
            self.assertEqual(vae_loader.load_vae("audio"), "VAELoader")
        events = [call.args[0] for call in emit.call_args_list]
        self.assertEqual(events[0], "miniacc_timing_hook_ready")
        self.assertEqual(events[1:3], ["miniacc_timing_start", "miniacc_timing_end"])
        self.assertEqual(events[3:5], ["miniacc_timing_start", "miniacc_timing_end"])
        self.assertEqual(emit.call_args_list[1].kwargs["phase"], "denoising")
        self.assertEqual(emit.call_args_list[3].kwargs["phase"], "initial_model_loading")
        timing_calls = [call for call in emit.call_args_list if call.args[0] == "miniacc_timing_start"]
        self.assertEqual(sum(call.kwargs.get("node_id") == "VAELoader" for call in timing_calls), 2)
        self.assertEqual(sum(call.args[0] == "miniacc_timing_end" and call.kwargs.get("node_id") == "VAELoader" for call in emit.call_args_list), 2)

    def test_sampler_checkpoint_wraps_registered_v3_dispatch_not_ordinary_import(self):
        latent = object()

        class FakeNodeOutput:
            def __init__(self):
                self.args = ({"samples": latent}, {"samples": object()})

            def __getitem__(self, index):
                return self.args[index]

        output = FakeNodeOutput()
        calls = []

        class OrdinarySampler:
            pass

        class RegisteredSampler:
            @classmethod
            def execute(cls, value):
                calls.append((cls, value))
                return output

            sample = execute

            @classmethod
            def EXECUTE_NORMALIZED(cls, value):
                return cls.execute(value)

        fake_nodes = types.ModuleType("nodes")
        fake_nodes.NODE_CLASS_MAPPINGS = {"SamplerCustomAdvanced": RegisteredSampler}
        with patch.dict(sys.modules, {"nodes": fake_nodes}), patch.dict(
            os.environ,
            {"MINIACC_LATENT_CHECKPOINT": "/tmp/unused-checkpoint.safetensors"},
        ), patch.object(miniacc_bootstrap, "save_av_latent") as save:
            self.assertIsNot(OrdinarySampler, RegisteredSampler)
            miniacc_bootstrap.install_latent_checkpoint(RegisteredSampler)
            miniacc_bootstrap.install_latent_checkpoint(RegisteredSampler)  # idempotent
            self.assertIs(RegisteredSampler.EXECUTE_NORMALIZED("input"), output)
        self.assertEqual(calls, [(RegisteredSampler, "input")])
        save.assert_called_once_with(latent, Path("/tmp/unused-checkpoint.safetensors"))

    def test_checkpoint_retains_both_streams_and_refuses_overwrite(self):
        class Tensor:
            shape = (1, 2, 3)

            def detach(self):
                return self

            def cpu(self):
                return self

            def contiguous(self):
                return self

        video, audio = Tensor(), Tensor()
        samples = types.SimpleNamespace(is_nested=True, unbind=lambda: (video, audio))
        finite = types.SimpleNamespace(
            all=lambda: types.SimpleNamespace(item=lambda: True)
        )
        fake_torch = types.SimpleNamespace(isfinite=lambda _: finite)
        fake_safe = types.ModuleType("safetensors.torch")
        saved = []

        def serialize(tensors, metadata):
            saved.append((tensors, metadata))
            return b"fixture-safetensors-bytes"

        fake_safe.save = serialize
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "safetensors": types.ModuleType("safetensors"),
                "safetensors.torch": fake_safe,
            },
        ), patch.object(miniacc_bootstrap, "_emit") as emit:
            path = Path(directory) / "av.safetensors"
            miniacc_bootstrap.save_av_latent(samples, path)
            self.assertEqual(saved[0][0], {"video": video, "audio": audio})
            self.assertEqual(
                emit.call_args.kwargs["finite"], {"video": True, "audio": True}
            )
            with self.assertRaises(FileExistsError):
                miniacc_bootstrap.save_av_latent(samples, path)
            self.assertEqual(path.read_bytes(), b"fixture-safetensors-bytes")
            finite.all = lambda: types.SimpleNamespace(item=lambda: False)
            bad_path = Path(directory) / "nonfinite.safetensors"
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                miniacc_bootstrap.save_av_latent(samples, bad_path)
            self.assertTrue(bad_path.is_file())  # retain invalid latents for diagnosis

    def test_comfy_flags_are_parsed_before_marker_import(self):
        # A separate interpreter reproduces Comfy's import-time argument cache
        # without importing torch, initializing CUDA, or launching a server.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "comfy"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "options.py").write_text(
                "args_parsing = False\n"
                "def enable_args_parsing():\n"
                "    global args_parsing\n"
                "    args_parsing = True\n"
            )
            (package / "cli_args.py").write_text(
                "import argparse, comfy.options\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--novram', action='store_true')\n"
                "parser.add_argument('--cpu-vae', action='store_true')\n"
                "parser.add_argument('--output-directory')\n"
                "parser.add_argument('--extra-model-paths-config')\n"
                "args = parser.parse_args(None if comfy.options.args_parsing else [])\n"
            )
            main = root / "main.py"
            main.write_text(
                "import json, comfy.options\n"
                "comfy.options.enable_args_parsing()\n"
                "from comfy.cli_args import args\n"
                "print(json.dumps(vars(args)))\n"
            )
            script = (
                "import importlib, miniacc_bootstrap\n"
                "miniacc_bootstrap.configure_allocator = lambda *_: None\n"
                "miniacc_bootstrap.install_forward_markers = "
                "lambda: importlib.import_module('comfy.cli_args')\n"
                "miniacc_bootstrap.wrap_extra_node_initialization = lambda: None\n"
                "miniacc_bootstrap.main()\n"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    "--comfy-main",
                    str(main),
                    "--novram",
                    "--cpu-vae",
                    "--output-directory",
                    str(root / "media"),
                    "--extra-model-paths-config",
                    str(root / "models.yaml"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            self.assertEqual(
                json.loads(result.stdout),
                {
                    "novram": True,
                    "cpu_vae": True,
                    "output_directory": str(root / "media"),
                    "extra_model_paths_config": str(root / "models.yaml"),
                },
            )

    def test_allocator_rejects_unknown_or_non_native_backend(self):
        for backend in (None, "unknown", "cudaMallocAsync"):
            with self.subTest(backend=backend):
                cuda = types.SimpleNamespace(
                    is_available=lambda: True,
                    device_count=lambda: 1,
                    get_device_properties=lambda _: types.SimpleNamespace(
                        total_memory=24 * 1024**3
                    ),
                    set_per_process_memory_fraction=lambda *args, **kwargs: None,
                    get_per_process_memory_fraction=lambda _: 16 / 24,
                )
                if backend is not None:
                    cuda.get_allocator_backend = lambda: backend
                with patch.dict(
                    sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}
                ), patch.object(miniacc_bootstrap, "_emit") as emit:
                    with self.assertRaisesRegex(RuntimeError, "native"):
                        miniacc_bootstrap.configure_allocator()
                    emit.assert_not_called()

    def test_allocator_configuration_sets_and_verifies_fraction(self):
        state = {"fraction": None}

        class FakeCuda:
            def is_available(self):
                return True

            def device_count(self):
                return 1

            def get_device_properties(self, device):
                return types.SimpleNamespace(total_memory=24 * 1024**3)

            def set_per_process_memory_fraction(self, value, device=0):
                state["fraction"] = value

            def get_per_process_memory_fraction(self, device=0):
                return state["fraction"]

            def get_allocator_backend(self):
                return "native"

        fake_torch = types.SimpleNamespace(cuda=FakeCuda())
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
            miniacc_bootstrap, "_emit"
        ) as emit:
            miniacc_bootstrap.configure_allocator()
        self.assertAlmostEqual(state["fraction"], 16 / 24)
        emit.assert_called_once()
        self.assertEqual(emit.call_args.args[0], "miniacc_allocator_configured")

    def test_forward_marker_producer_wraps_h3_forward_without_changing_result(self):
        class FakeModel:
            def forward(self, x, timestep):
                return (x, timestep)

        fake_model = types.ModuleType("comfy.ldm.minimax.model")
        fake_model.MiniMaxH3Model = FakeModel
        with patch.dict(
            sys.modules,
            {
                "comfy": types.ModuleType("comfy"),
                "comfy.ldm": types.ModuleType("comfy.ldm"),
                "comfy.ldm.minimax": types.ModuleType("comfy.ldm.minimax"),
                "comfy.ldm.minimax.model": fake_model,
            },
        ), patch.object(miniacc_bootstrap, "_emit") as emit:
            miniacc_bootstrap.install_forward_markers()
            result = FakeModel().forward("x", "sigma")
        self.assertEqual(result, ("x", "sigma"))
        self.assertEqual(
            [call.args[0] for call in emit.call_args_list],
            ["miniacc_dit_forward_start", "miniacc_dit_forward_end"],
        )


class ResourceGuardTests(unittest.TestCase):
    def test_samples_survive_guard_violation_and_cannot_be_overwritten(self):
        snapshot = {
            "gpus": [{"index": 0, "memory_free_mib": 4096}],
            "host_available_bytes": 16 * 1024**3,
            "project_free_bytes": 200 * 1024**3,
            "executable_free_bytes": 20 * 1024**3,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resources.jsonl"
            guard = local_probe.ResourceGuard(
                snapshot_fn=lambda: snapshot, sample_log=path
            )
            guard.check()
            snapshot["gpus"][0]["memory_free_mib"] = 2047
            with self.assertRaises(local_probe.ResourceGuardViolation):
                guard.check()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                [row["gpus"][0]["memory_free_mib"] for row in rows], [4096, 2047]
            )
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                local_probe.ResourceGuard(sample_log=path)
            self.assertEqual(path.read_bytes(), original)

    def test_headroom_guard_rejects_low_vram_and_host(self):
        snapshot = {
            "gpus": [{"index": 0, "memory_free_mib": 2047}],
            "host_available_bytes": 9 * 1024**3,
            "project_free_bytes": 200 * 1024**3,
            "executable_free_bytes": 20 * 1024**3,
        }
        self.assertIn("GPU free", local_probe.headroom_violation(snapshot))
        snapshot["gpus"][0]["memory_free_mib"] = 4096
        snapshot["host_available_bytes"] = 7 * 1024**3
        self.assertIn("host available", local_probe.headroom_violation(snapshot))

    def test_headroom_guard_accepts_contractual_reserves(self):
        snapshot = {
            "gpus": [{"index": 0, "memory_free_mib": 2048}],
            "host_available_bytes": 8 * 1024**3,
            "project_free_bytes": 100 * 1024**3,
            "executable_free_bytes": 15 * 1024**3,
        }
        self.assertIsNone(local_probe.headroom_violation(snapshot))

    def test_unknown_host_or_disk_telemetry_rejects(self):
        snapshot = {
            "gpus": [{"index": 0, "memory_free_mib": 4096}],
            "host_available_bytes": None,
            "project_free_bytes": 200 * 1024**3,
            "executable_free_bytes": 20 * 1024**3,
        }
        self.assertIn("host available", local_probe.headroom_violation(snapshot))
        snapshot["host_available_bytes"] = 16 * 1024**3
        snapshot["project_free_bytes"] = None
        self.assertIn("project free", local_probe.headroom_violation(snapshot))

    def test_media_validation_requires_both_streams(self):
        with patch("miniacc_core.evaluation.ffprobe", return_value={"streams": []}):
            with self.assertRaisesRegex(ValueError, "both video and audio"):
                local_probe.validate_media(
                    Path("missing.mp4"),
                    {
                        "width": 1344,
                        "height": 768,
                        "frames": 124,
                        "fps": 24,
                        "audio_channels": 2,
                        "audio_rate": 32000,
                    },
                )


if __name__ == "__main__":
    unittest.main()
