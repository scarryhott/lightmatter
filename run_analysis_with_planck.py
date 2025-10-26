#!/usr/bin/env python3
"""
Helper to download the Planck 857 GHz map, verify it, and run the IVI analysis.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from download_planck_map import download_planck_map
from verify_map import verify_map


OUTPUT_DIR = Path("results/planck_857_analysis")
PLANCK_PATH = Path("data/planck/HFI_SkyMap_857_2048_R2.02_full.fits")
DEFAULT_CONFIG = Path("configs/physical.yaml")
KAPPA_CSV = Path("data/kappa_ext.csv")


def run_analysis() -> int:
    """Execute scripts/run_analysis.py with the Planck radiation map."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_analysis.py",
        "--config",
        str(DEFAULT_CONFIG),
        "--rad-map",
        str(PLANCK_PATH),
        "--auto-I0",
        "--gamma",
        "1.0",
        "--output-dir",
        str(OUTPUT_DIR),
        "--plots",
    ]

    if KAPPA_CSV.exists():
        cmd += ["--kappa-csv", str(KAPPA_CSV)]

    print("[RUN] " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"[DONE] Results saved to {OUTPUT_DIR}")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Analysis failed with exit code {exc.returncode}")
        return exc.returncode


def main() -> int:
    path = download_planck_map()
    if not path:
        return 1

    if not verify_map(path):
        print("[ABORT] Map verification failed; not running analysis.")
        return 2

    return run_analysis()


if __name__ == "__main__":
    sys.exit(main())
