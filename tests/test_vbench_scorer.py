import json
from pathlib import Path
import tempfile
import unittest

from miniacc_core.vbench_scorer import dimension_metadata, scorer_exit_code, update_scored_ledger


ROOT = Path(__file__).resolve().parents[1]


class VBenchAdapterLedgerTests(unittest.TestCase):
    def ledger(self):
        manifest = json.loads((ROOT / "data/stage1/vbench_dev_manifest.json").read_text())
        prompt = next(p for p in manifest["prompts"]
                      if "overall_consistency" in p["eligible_standard_development_dimensions"])
        return {
            "status": "ready_for_scoring",
            "entries": [{
                "prompt_id": prompt["id"], "seed": 20260909,
                "prompt_en": prompt["prompt_en"], "output": "/tmp/clip.mp4",
                "media": {"status": "validated"},
                "eligible_metrics": {"custom_input": [
                    "subject_consistency", "background_consistency", "motion_smoothness",
                    "dynamic_degree", "aesthetic_quality", "imaging_quality"],
                    "overall_consistency": True},
                "official_metadata_rows": prompt["official_metadata_rows"],
                "scores": {"raw": {}, "normalized": {}},
            }],
        }

    def test_metadata_uses_custom_and_original_standard_rows(self):
        ledger = self.ledger()
        custom = dimension_metadata(ledger, "subject_consistency")
        standard = dimension_metadata(ledger, "overall_consistency")
        self.assertEqual(custom[0]["prompt_en"], ledger["entries"][0]["prompt_en"])
        self.assertEqual(custom[0]["dimension"], ["subject_consistency"])
        self.assertIn("overall_consistency", standard[0]["dimension"])
        self.assertEqual(standard[0]["video_list"], ["/tmp/clip.mp4"])

    def test_non_ready_ledger_is_blocked_without_loading_scorer(self):
        ledger = self.ledger()
        ledger["status"] = "blocked_invalid_media"
        with self.assertRaisesRegex(ValueError, "not ready"):
            dimension_metadata(ledger, "subject_consistency")

    def test_scored_ledger_counts_only_native_metric_values(self):
        ledger = self.ledger()
        entry = ledger["entries"][0]
        entry["scores"]["raw"]["subject_consistency"] = 0.5
        entry["scores"]["normalized"]["subject_consistency"] = 41.0
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "scored-ledger.json"
            update_scored_ledger(ledger, destination,
                                 scoring_profile={"purpose": "official-vbench-stage1-scoring"})
            saved = json.loads(destination.read_text())
        self.assertEqual(saved["status"], "partial_scoring")
        self.assertEqual(saved["scoring_status"]["status"], "partial")
        self.assertEqual(saved["scoring_profile"]["purpose"], "official-vbench-stage1-scoring")
        self.assertEqual(saved["score_counts"], {"raw": 1, "normalized": 1})
        self.assertEqual(saved["scored_dimensions"], ["subject_consistency"])
        self.assertNotIn("quality_index", saved)

    def test_completed_scoring_status_is_reproducible_and_resumable(self):
        ledger = self.ledger()
        entry = ledger["entries"][0]
        for dimension in ("subject_consistency", "background_consistency", "motion_smoothness",
                          "dynamic_degree", "aesthetic_quality", "imaging_quality",
                          "overall_consistency"):
            entry["scores"]["raw"][dimension] = 0.5
            entry["scores"]["normalized"][dimension] = 50.0
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "scored-ledger.json"
            update_scored_ledger(ledger, destination,
                                 preparation_profile={"purpose": "official-vbench-stage1-preparation"},
                                 scoring_profile={"purpose": "official-vbench-stage1-scoring"})
            saved = json.loads(destination.read_text())
        self.assertEqual(saved["status"], "completed_scoring")
        self.assertEqual(saved["scoring_status"]["status"], "completed")
        self.assertEqual(len(saved["scored_dimensions"]), 7)
        self.assertEqual(saved["preparation_profile"]["purpose"], "official-vbench-stage1-preparation")

    def test_partial_scoring_is_a_failed_process_status(self):
        self.assertEqual(scorer_exit_code("completed"), 0)
        self.assertEqual(scorer_exit_code("completed_with_missing"), 1)
        self.assertEqual(scorer_exit_code("failed"), 1)

    def test_prior_scored_status_can_be_used_for_resume(self):
        ledger = self.ledger()
        ledger["status"] = "completed_scoring"
        self.assertEqual(len(dimension_metadata(ledger, "subject_consistency")), 1)


if __name__ == "__main__":
    unittest.main()
