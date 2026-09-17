#!/usr/bin/env python3
"""Assemble SGLang H3 request records into a score-ready native AV ledger."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUSTOM = ["subject_consistency", "background_consistency", "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality"]
METRICS = CUSTOM + ["overall_consistency"]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()

def validate(path: Path, ffprobe: Path, ffmpeg: Path) -> dict:
    result = {"path": str(path), "exists": path.is_file()}
    if not path.is_file(): result["status"] = "missing"; return result
    result["sha256"] = sha256(path)
    cmd = [str(ffprobe), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    probe = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=False)
    result["ffprobe_returncode"] = probe.returncode
    if probe.returncode:
        result["status"] = "invalid"; result["stderr"] = probe.stderr[-4000:]; return result
    try: data = json.loads(probe.stdout)
    except json.JSONDecodeError as exc:
        result["status"] = "invalid"; result["stderr"] = str(exc); return result
    result["streams"] = data.get("streams", []); result["format"] = data.get("format", {})
    videos = [s for s in result["streams"] if s.get("codec_type") == "video"]
    audios = [s for s in result["streams"] if s.get("codec_type") == "audio"]
    result["video_valid"] = any(s.get("width") == 1344 and s.get("height") == 768 and s.get("r_frame_rate") == "24/1" for s in videos)
    result["audio_valid"] = any(s.get("sample_rate") == "32000" and s.get("channels") == 2 for s in audios)
    decode = subprocess.run([str(ffmpeg), "-v", "error", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"], capture_output=True, text=True, timeout=180, check=False)
    result["full_decode_returncode"] = decode.returncode
    result["full_decode_stderr"] = decode.stderr[-4000:] if decode.returncode else None
    result["status"] = "validated" if result["video_valid"] and result["audio_valid"] and decode.returncode == 0 else "invalid"
    return result

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ffprobe", type=Path, required=True)
    ap.add_argument("--ffmpeg", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=ROOT / "stage1/eval.yaml")
    args = ap.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    prompts = {p["id"]: p for p in manifest["prompts"]}
    entries = []
    for record_path in sorted((args.root / "requests").glob("*.json")):
        record = json.loads(record_path.read_text())
        prompt_id = record.get("prompt_id")
        prompt = prompts.get(prompt_id)
        jid = record.get("job_id", record_path.stem)
        output = args.root / "media" / f"{jid}.mp4"
        media = validate(output, args.ffprobe, args.ffmpeg)
        entries.append({
            "artifact_root": str(args.root.resolve()), "result": str(record_path.resolve()),
            "job_id": jid, "prompt_id": prompt_id, "seed": record.get("seed"),
            "prompt_en": prompt.get("prompt_en") if prompt else record.get("prompt"),
            "stratum": prompt.get("stratum") if prompt else None, "output": str(output.resolve()),
            "generation_status": record.get("result", "missing_result"), "generation_error": record.get("error"),
            "e2e_seconds": record.get("e2e_seconds"), "expected_forward_count": record.get("expected_forwards"),
            "media": media,
            "official_metadata_rows": prompt.get("official_metadata_rows", []) if prompt else [],
            "eligible_metrics": {"custom_input": CUSTOM, "overall_consistency": bool(prompt and any("overall_consistency" in r.get("dimension", []) for r in prompt.get("official_metadata_rows", [])))}, 
            "scores": {"raw": {}, "normalized": {}}
        })
    accepted = {"validated"}
    eligible = {m: sum(e["media"].get("status") in accepted and (m in CUSTOM or e["eligible_metrics"].get(m) is True) for e in entries) for m in METRICS}
    errors = [{"job_id": e["job_id"], "status": e["media"].get("status"), "generation_status": e["generation_status"], "error": e.get("generation_error")} for e in entries if e["media"].get("status") not in accepted or e["generation_status"] != "success"]
    value = {"schema_version": 1, "family": "sglang-h3", "status": "ready_for_scoring" if entries and not errors else ("no_outputs" if not entries else "blocked_invalid_media"), "source_manifest": str(args.manifest.resolve()), "source_manifest_sha256": sha256(args.manifest), "tiers": [4, 8, 16], "metrics": METRICS, "eligible_counts": eligible, "errors": errors, "score_counts": {"raw": 0, "normalized": 0}, "entries": entries, "limitations": ["No scores are fabricated or zero-filled.", "Five normalized points is an alert only, not statistical rejection.", "No scalar VBench total, formal equivalence, or small-sample p95 is computed."]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": value["status"], "entries": len(entries), "errors": len(errors), "eligible_counts": eligible}))
    return 0 if value["status"] == "ready_for_scoring" else 2
if __name__ == "__main__": raise SystemExit(main())
