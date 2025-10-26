#!/usr/bin/env python3
"""Simple sanity checks for a HEALPix FITS map."""

import sys
import numpy as np


def verify_map(map_path: str) -> bool:
    """Load a HEALPix map and print basic statistics."""
    import healpy as hp  # defer import so script errors are clearer

    print(f"[CHECK] Reading map: {map_path}")
    m = hp.read_map(map_path, field=0, verbose=False)
    nside = hp.get_nside(m)
    finite = np.isfinite(m)

    if not finite.any():
        print("[ERROR] Map contains no finite pixels.")
        return False

    print(f"[OK]   NSIDE = {nside}")
    print(f"[OK]   Pixels kept: {finite.sum()} / {m.size}")
    print(f"[OK]   Min/Max = {np.nanmin(m):.4g}, {np.nanmax(m):.4g}")
    print(f"[OK]   Median  = {np.nanmedian(m):.4g}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_map.py /path/to/map.fits")
        sys.exit(1)
    ok = verify_map(sys.argv[1])
    sys.exit(0 if ok else 2)
