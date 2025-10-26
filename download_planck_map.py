#!/usr/bin/env python3
"""Download the Planck 857 GHz HEALPix intensity map into data/planck/."""

import os
import shutil
from astropy.utils.data import download_file


PLANCK_DIR = os.path.join("data", "planck")
PLANCK_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/"
    "HFI_SkyMap_857_2048_R2.02_full.fits"
)
PLANCK_PATH = os.path.join(PLANCK_DIR, "HFI_SkyMap_857_2048_R2.02_full.fits")


def download_planck_map(force: bool = False) -> str | None:
    """
    Fetch the Planck 857 GHz map.

    Parameters
    ----------
    force : bool
        When True, re-download even if the file already exists.

    Returns
    -------
    str | None
        Path to the downloaded file, or None if the download failed.
    """
    os.makedirs(PLANCK_DIR, exist_ok=True)

    if os.path.exists(PLANCK_PATH) and not force:
        print(f"[INFO] Planck map already present: {PLANCK_PATH}")
        return PLANCK_PATH

    try:
        print(f"[INFO] Downloading Planck 857 GHz map → {PLANCK_PATH}")
        tmp_path = download_file(PLANCK_URL, cache=True, pkgname="lightmatter_planck")
        shutil.copy2(tmp_path, PLANCK_PATH)
        print("[INFO] Download complete.")
        return PLANCK_PATH
    except Exception as exc:  # pragma: no cover - network failures
        print(f"[ERROR] Failed to download Planck map: {exc}")
        print("        You can download manually from:")
        print(f"        {PLANCK_URL}")
        print(f"        and place it at {PLANCK_PATH}")
        return None


if __name__ == "__main__":
    download_planck_map()
