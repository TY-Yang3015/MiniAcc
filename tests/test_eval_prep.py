import json
from pathlib import Path
import tempfile
import unittest

from miniacc_core.eval_prep import (
    CUSTOM_DIMENSIONS, DEVELOPMENT_DIMENSIONS, load_eval_config, normalize_metric,
    prepare_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/stage1/vbench_dev_manifest.json"


class EvaluationPreparationTests(unittest.TestCase):
    def test_current_layout_maps_prompt_seed_without_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "q0"
            job_root = root / "gpu-0"
            (job_root / "media").mkdir(parents=True)
            (job_root / "media" / "clip.mp4").write_bytes(b"not scored")
            result = {"status": "success", "job": {"id": "vbench-0195-s20260909"}}
            (job_root / "result.json").write_text(json.dumps(result))
            prepared = prepare_manifest(root, MANIFEST)
            self.assertEqual(prepared["funnel"]["available_prompt_count"], 1)
            entry = prepared["entries"][0]
            self.assertEqual(entry["prompt_en"], "A person is squat")
            self.assertEqual(entry["seed"], 20260909)
            self.assertEqual(entry["eligible_metrics"]["custom_input"], list(CUSTOM_DIMENSIONS))
            self.assertFalse(entry["eligible_metrics"]["overall_consistency"])
            self.assertEqual(entry["scores"], {"raw": {}, "normalized": {}})
            self.assertEqual(entry["media"]["status"], "unprobed")
            self.assertNotIn("quality_index", prepared)

    def test_overall_eligibility_preserves_official_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_root = Path(tmp) / "gpu-0"
            (job_root / "media").mkdir(parents=True)
            (job_root / "media" / "clip.mp4").write_bytes(b"not scored")
            (job_root / "result.json").write_text(json.dumps({"job": {"id": "vbench-0749-s20260909"}}))
            entry = prepare_manifest(Path(tmp), MANIFEST)["entries"][0]
            self.assertTrue(entry["eligible_metrics"]["overall_consistency"])
            self.assertEqual(entry["official_metadata_rows"][0]["prompt_en"], entry["prompt_en"])

    def test_cpu_profile_and_pinned_metadata_are_consumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared = prepare_manifest(
                Path(tmp), MANIFEST,
                eval_config=ROOT / "exp_configs/vbench-eval-cpu.yaml",
            )
            self.assertEqual(prepared["evaluation_profile"]["device"], "cpu")
            self.assertFalse(prepared["evaluation_profile"]["run_scorers"])
            self.assertEqual(prepared["evaluation_profile"]["tiers"], [4, 8, 16])
            self.assertEqual(prepared["vbench_source"]["sha256"], "5dd2de80ee43cda750b2b72ea7023657c0b90d3702041c7e4608c65dbe50dccd")

    def test_normalization_is_per_metric_only(self):
        self.assertEqual(normalize_metric("subject_consistency", 0.1462), 0.0)
        with self.assertRaises(ValueError):
            normalize_metric("missing", 1.0)

    def test_invalid_media_does_not_claim_ready_for_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "gpu-0"
            (root / "media").mkdir(parents=True)
            (root / "media" / "clip.mp4").write_bytes(b"not media")
            (root / "result.json").write_text(json.dumps({"job": {"id": "vbench-0195-s20260909"}}))
            prepared = prepare_manifest(Path(tmp), MANIFEST, ffprobe=Path("/bin/false"))
        self.assertEqual(prepared["status"], "blocked_invalid_media")
        self.assertEqual(prepared["entries"][0]["media"]["status"], "invalid")

    def test_recovery_hash_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "gpu-0"
            media = root / "media"
            media.mkdir(parents=True)
            output = media / "clip.mp4"
            output.write_bytes(b"recovered bytes")
            (root / "validation-recovery.json").write_text(json.dumps({
                "output": "/remote/output.mp4", "output_sha256": "0" * 64,
                "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
            }))
            (root / "result.json").write_text(json.dumps({"job": {"id": "vbench-0195-s20260909"}}))
            prepared = prepare_manifest(Path(tmp), MANIFEST)
        self.assertEqual(prepared["status"], "blocked_invalid_media")
        self.assertEqual(prepared["entries"][0]["media"]["status"], "invalid")
        self.assertFalse(prepared["entries"][0]["media"]["hash_match"])

    def test_scoring_profile_consumes_exact_revision_and_metric_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = json.loads((ROOT / "data/stage1/vbench_dev_manifest.json").read_text())
            config = load_eval_config(ROOT / "exp_configs/vbench-eval-scoring.yaml", source, scoring=True)
        self.assertTrue(config["run_scorers"])
        self.assertEqual(config["metrics"], list(DEVELOPMENT_DIMENSIONS))


if __name__ == "__main__":
    unittest.main()
