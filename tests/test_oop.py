"""Offline contracts for the refactored dependency-injected seams."""

from pathlib import Path
import tempfile
import unittest

from miniacc_core.artifacts import ArtifactStore
from miniacc_core.config import CandidateConfig, WorkloadConfig
from miniacc_core.data import PromptDataModule
from miniacc_core.interface import ApplicationInterface
from miniacc_core.models import (
    CandidateCatalog,
    ModelManager,
    ModelManagerFactory,
    UnsupportedCapabilityError,
    build_prompt_graph,
)


class FakeData:
    def jobs(self):
        return ({"id": "job", "prompt_id": "prompt", "seed": 7},)

    def prompt_for(self, job):
        return "frozen prompt"


class FakeManager(ModelManager):
    def __init__(self, fail=False):
        self.loaded = False
        self.closed = False
        self.fail = fail
        self.candidate = CandidateConfig()

    @property
    def capabilities(self):
        return frozenset({"t2va"})

    def load_inference_model(self, candidate=None, *, deadline=None):
        self.loaded = True
        return {"resident": False}

    def infer(self, prompt, seed, *, deadline=None):
        if self.fail:
            raise RuntimeError("inference failed")
        return {"prompt": prompt, "seed": seed}

    def close(self):
        self.closed = True
        self.loaded = False


class OOPContracts(unittest.TestCase):
    def test_graph_uses_immutable_native_workload(self):
        graph = build_prompt_graph("x", 3)
        inputs = graph["7"]["inputs"]
        self.assertEqual(
            (inputs["width"], inputs["height"], inputs["length"]), (1344, 768, 124)
        )

    def test_interface_injected_lifecycle_and_failure_cleanup(self):
        manager = FakeManager(fail=True)
        app = ApplicationInterface(data=FakeData(), manager=manager)
        app.load_inference_model()
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            app.infer((FakeData().jobs()[0],), output_root="media")
        self.assertTrue(manager.closed)

    def test_factory_does_not_route_unimplemented_families_to_base(self):
        with self.assertRaises(UnsupportedCapabilityError):
            ModelManagerFactory().create(CandidateConfig(family="sglang-h3"))

    def test_catalog_reports_local_assets_without_claiming_compatibility(self):
        root = Path(__file__).resolve().parents[1] / ".local/models"
        readiness = CandidateCatalog(root).inspect(
            CandidateConfig(family="comfyui-turbo")
        )
        self.assertEqual(readiness.compatibility, "unverified")
        self.assertIn(
            "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
            readiness.available_assets,
        )
        self.assertIn("Local assets are present", readiness.note)

    def test_artifacts_are_create_only_and_path_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            store.write_json("report.json", {"ok": True})
            with self.assertRaises(FileExistsError):
                store.write_json("report.json", {"ok": False})
            with self.assertRaises(ValueError):
                store.reserve("../outside.json")


if __name__ == "__main__":
    unittest.main()
