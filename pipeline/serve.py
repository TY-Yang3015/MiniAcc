#!/usr/bin/env python3
"""Run the MiniAcc integrated pipeline.

Default: one timed request (about 2.5 minutes) with the full pipeline
(AdaLN sidecar + INT8 weights via ComfyUI + SageAttention).

  python pipeline/serve.py --output outputs/demo

Options:
  --no-sage   fidelity-first configuration (AdaLN + INT8, baseline-level quality)
  --quality   run a 4-clip quality cohort instead of a single timed request
  --output    campaign output directory (created fresh; must not exist)

Prerequisites: apply the patches in pipeline/patches/, then python pipeline/setup.py.
"""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-sage', action='store_true', help='run without SageAttention (fidelity-first)')
    parser.add_argument('--quality', action='store_true', help='run the quality cohort instead of the timed request')
    parser.add_argument('--output', type=Path, default=Path('outputs/miniacc-demo'), help='fresh output directory')
    args = parser.parse_args()

    arm = 'adaln_kitchen' if args.no_sage else 'adaln_kitchen_sage'
    purpose = 'quality' if args.quality else 'speed'
    spec = importlib.util.spec_from_file_location('miniacc_driver', HERE / 'run_stage4.py')
    driver = importlib.util.module_from_spec(spec)
    sys.modules['miniacc_driver'] = driver
    spec.loader.exec_module(driver)
    sys.argv = ['serve.py', arm, purpose, str(args.output)]
    return driver.main()


if __name__ == '__main__':
    raise SystemExit(main())
