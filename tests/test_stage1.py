"""Offline correctness tests; these are not model/pipeline benchmarks."""

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import miniacc


ROOT = Path(__file__).resolve().parents[1]


class PromptManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = (ROOT / "data/stage1/vbench_full_info.json").read_bytes()
        audit = json.loads((ROOT / "data/stage1/source_audit.json").read_text())
        cls.source = next(source for source in audit["sources"] if source["id"] == "vbench_metadata")
        cls.manifest = miniacc.build_prompt_manifest(cls.raw, cls.source)

    def test_frozen_manifest_reproduces_exactly(self):
        frozen = json.loads((ROOT / "data/stage1/vbench_dev_manifest.json").read_text())
        self.assertEqual(self.manifest, frozen)
        self.assertEqual(self.manifest, miniacc.build_prompt_manifest(self.raw, self.source))

    def test_balanced_nested_subset_and_unique_prompts(self):
        prompts = self.manifest["prompts"]
        self.assertEqual(len(prompts), 64)
        self.assertEqual(len({p["prompt_en"] for p in prompts}), 64)
        self.assertEqual(len({p["id"] for p in prompts}), 64)
        self.assertTrue(all(p["cheap_filter"] for p in prompts[:32]))
        self.assertTrue(all(not p["cheap_filter"] for p in prompts[32:]))
        self.assertEqual(Counter(p["stratum"] for p in prompts), dict.fromkeys(miniacc.STRATA, 16))
        self.assertEqual(Counter(p["stratum"] for p in prompts[:32]), dict.fromkeys(miniacc.STRATA, 8))

    def test_original_metadata_and_auxiliary_labels_preserved(self):
        original = json.loads(self.raw)
        for prompt in self.manifest["prompts"]:
            for index, row in zip(prompt["official_metadata_indices"], prompt["official_metadata_rows"], strict=True):
                self.assertEqual(row, original[index])
                self.assertEqual(prompt["prompt_en"], original[index]["prompt_en"])
        scenes = [p for p in self.manifest["prompts"] if p["stratum"] == "scenes"]
        self.assertTrue(all("auxiliary_info" in p["official_metadata_rows"][0] for p in scenes))

    def test_exact_generation_funnel_and_seed_pairs(self):
        jobs = self.manifest["jobs"]
        self.assertEqual(len(jobs), 128)
        self.assertEqual(len({j["id"] for j in jobs}), 128)
        self.assertEqual(len({j["relative_output_path"] for j in jobs}), 128)
        self.assertEqual(Counter(j["allocation"] for j in jobs), {"cheap_filter": 32, "shortlist_additional": 96})
        for prompt in self.manifest["prompts"]:
            prompt_jobs = [j for j in jobs if j["prompt_id"] == prompt["id"]]
            self.assertEqual([j["seed"] for j in prompt_jobs], list(miniacc.SEEDS))
            self.assertEqual([j["sample_index"] for j in prompt_jobs], [0, 1])
        self.assertEqual({j["seed"] for j in jobs if j["allocation"] == "cheap_filter"}, {miniacc.SEEDS[0]})

    def test_dimension_eligibility_is_not_fabricated(self):
        self.assertEqual(self.manifest["custom_input_dimensions"], list(miniacc.DEVELOPMENT_DIMENSIONS[:-1]))
        self.assertEqual(self.manifest["standard_metadata_dimensions"], ["overall_consistency"])
        counts = Counter()
        for prompt in self.manifest["prompts"]:
            eligible = set(prompt["eligible_standard_development_dimensions"])
            official = {d for row in prompt["official_metadata_rows"] for d in row["dimension"]}
            self.assertEqual(eligible, official & set(miniacc.DEVELOPMENT_DIMENSIONS))
            counts.update(eligible)
        self.assertEqual(counts, dict.fromkeys(miniacc.DEVELOPMENT_DIMENSIONS, 16))

    def test_repeated_text_retains_all_rows_without_duplicate_generation(self):
        rows = json.loads(self.raw)
        prompt = self.manifest["prompts"][0]["prompt_en"]
        extra = {"prompt_en": prompt, "dimension": ["color"], "auxiliary_info": {"test": "preserve"}}
        rows.append(extra)
        raw = json.dumps(rows).encode()
        result = miniacc.build_prompt_manifest(raw, {**self.source, "sha256": miniacc.sha256(raw)})
        selected = [p for p in result["prompts"] if p["prompt_en"] == prompt]
        self.assertEqual(len(selected), 1)
        self.assertIn(extra, selected[0]["official_metadata_rows"])
        self.assertEqual(len(result["jobs"]), 128)

    def test_changed_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hash"):
            miniacc.build_prompt_manifest(self.raw + b" ", self.source)

    def test_insufficient_coverage_is_rejected(self):
        raw = b"[]"
        with self.assertRaisesRegex(ValueError, "requires 16"):
            miniacc.build_prompt_manifest(raw, {**self.source, "sha256": miniacc.sha256(raw)})

    def test_invalid_metadata_is_rejected(self):
        for row in [{"prompt_en": "", "dimension": []}, {"prompt_en": "valid", "dimension": "wrong"}]:
            raw = json.dumps([row]).encode()
            with self.assertRaises(ValueError):
                miniacc.build_prompt_manifest(raw, {**self.source, "sha256": miniacc.sha256(raw)})


class PreflightTests(unittest.TestCase):
    CSV = "0, NVIDIA GeForce RTX 4090, 8.9, 24564, 1496, 22579, 595.84, 500.00\n"

    def test_budget_uses_reported_free_vram(self):
        gpu = miniacc.parse_gpu_inventory(self.CSV)[0]
        self.assertTrue(gpu["is_target_standard_4090"])
        self.assertEqual(gpu["snapshot_incremental_budget_mib"], 22579 - 2048)
        self.assertNotEqual(gpu["snapshot_incremental_budget_mib"], 24564 - 1496 - 2048)

    def test_low_free_memory_never_yields_negative_budget(self):
        gpu = miniacc.parse_gpu_inventory(self.CSV.replace("22579", "1024"))[0]
        self.assertEqual(gpu["snapshot_incremental_budget_mib"], 0)

    def test_4090_d_is_not_the_standard_target(self):
        self.assertFalse(miniacc.parse_gpu_inventory(self.CSV.replace("RTX 4090,", "RTX 4090 D,"))[0]["is_target_standard_4090"])

    def test_bad_csv_and_too_small_reserve_are_rejected(self):
        with self.assertRaises(ValueError):
            miniacc.parse_gpu_inventory("0, RTX 4090")
        with self.assertRaises(ValueError):
            miniacc.parse_gpu_inventory(self.CSV.replace("22579", "99999"))
        with self.assertRaises(ValueError):
            miniacc.parse_gpu_inventory(self.CSV, reserve_mib=0)

    def test_missing_gpu_tools_do_not_claim_readiness(self):
        with patch("miniacc_core.data.command_output", return_value={"status": "not_installed", "stdout": ""}):
            report = miniacc.preflight()
        self.assertEqual(report["gpus"], [])
        self.assertEqual(report["gpu_query_status"], "not_installed")
        self.assertIn("not_a_benchmark", report["kind"])
        self.assertTrue(any("not confirmed" in warning for warning in report["warnings"]))


class FileAndCLITests(unittest.TestCase):
    def test_workload_records_native_shape_without_claiming_stage_completion(self):
        workload = json.loads((ROOT / "data/stage1/workload.json").read_text())
        video = workload["video"]
        self.assertEqual((video["width"], video["height"], video["frames"], video["fps"]), (1344, 768, 124, 24))
        self.assertEqual((video["frames"] - 5) % 17, 0)
        self.assertFalse(workload["stage2_authorized"])
        self.assertEqual(workload["q0"]["clips_cached"], 0)
        self.assertFalse(workload["timing_contract"]["measurements_performed"])

    def test_recorded_artifact_hashes_match(self):
        workload = json.loads((ROOT / "data/stage1/workload.json").read_text())
        for relative_path, expected in workload["artifact_sha256"].items():
            self.assertEqual(miniacc.sha256((ROOT / relative_path).read_bytes()), expected, relative_path)
        audit = json.loads((ROOT / "data/stage1/source_audit.json").read_text())
        for source in audit["sources"]:
            if "local_path" in source:
                self.assertEqual(miniacc.sha256((ROOT / source["local_path"]).read_bytes()), source["sha256"])

    def test_snapshot_overwrite_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            miniacc.write_json(path, {"original": True})
            with self.assertRaises(FileExistsError):
                miniacc.write_json(path, {"original": False})
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_cli_reproduces_manifest_outside_repository_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "miniacc.py"), "prepare-prompts"],
                cwd=directory, capture_output=True, text=True, check=True,
            )
        frozen = json.loads((ROOT / "data/stage1/vbench_dev_manifest.json").read_text())
        self.assertEqual(json.loads(result.stdout), frozen)

    def test_cli_has_no_generation_or_scoring_command(self):
        with self.assertRaises(SystemExit) as result:
            with patch("sys.stderr"):
                miniacc.main(["generate"])
        self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
