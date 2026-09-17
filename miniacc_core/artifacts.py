"""Create-only run namespace and provenance persistence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ArtifactStore:
    """Persist evidence under one isolated run directory without overwriting."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()

    def reserve(self, relative: str) -> Path:
        path = (self.run_dir / relative).resolve()
        if not path.is_relative_to(self.run_dir):
            raise ValueError("artifact path escapes run directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative: str, value: dict) -> Path:
        path = self.reserve(relative)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        return path

    def hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def write_history(self, history: dict) -> Path:
        return self.write_json("history.json", history)
