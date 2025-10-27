#!/usr/bin/env python3
"""
Verify repository-tracked data files against SHA256SUMS.

Usage:
    python scripts/check_data_integrity.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict


def load_reference(file_path: Path) -> Dict[Path, str]:
    mapping: Dict[Path, str] = {}
    with file_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, rel_path = line.split(None, 1)
            mapping[Path(rel_path)] = digest.lower()
    return mapping


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check data file SHA-256 hashes.")
    parser.add_argument("--data-dir", default="data", help="Directory containing SHA256SUMS.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sums_path = data_dir / "SHA256SUMS"
    if not sums_path.exists():
        print(f"[ERROR] Missing checksum file: {sums_path}")
        return 1

    references = load_reference(sums_path)
    missing = []
    mismatched = []

    for rel_path, digest in references.items():
        file_path = data_dir / rel_path
        if not file_path.exists():
            missing.append(rel_path)
            continue
        actual = compute_sha256(file_path)
        if actual != digest:
            mismatched.append((rel_path, digest, actual))

    if missing or mismatched:
        if missing:
            print("[FAIL] Missing files:")
            for rel in missing:
                print(f"  - {rel}")
        if mismatched:
            print("[FAIL] Checksum mismatches:")
            for rel, expected, actual in mismatched:
                print(f"  - {rel}: expected {expected}, got {actual}")
        return 1

    print("[OK] All data files match SHA256SUMS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
