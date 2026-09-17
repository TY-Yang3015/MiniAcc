"""Thin legacy entrypoint for MiniAcc preparation commands.

Implementation lives in :mod:`miniacc_core`; this module preserves historical imports.
"""
from miniacc_core.config import DEVELOPMENT_DIMENSIONS, STRATA, SEEDS, SELECTION_NAMESPACE
from miniacc_core.data import (build_prompt_manifest, command_output, load_manifest,
    parse_gpu_inventory, prompt_stratum, sha256, write_json)
from miniacc_core import data as _data
from miniacc_core.cli import preparation_main as main

def preflight():
    """Compatibility helper delegating directly to the data service."""
    return _data.preflight()

if __name__ == "__main__":
    main()
