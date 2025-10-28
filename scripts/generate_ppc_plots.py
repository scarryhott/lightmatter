#!/usr/bin/env python3
"""Generate posterior predictive diagnostic plots for IVI channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_ppc(ax, observed, replicated, title, units):
    ax.hist(replicated, bins=30, alpha=0.7, label="replicates")
    ax.axvline(observed, color="red", linestyle="--", label="observed")
    ax.set_xlabel(f"chi2 ({units})")
    ax.set_title(title)
    ax.legend()


def load_ppc_data(results_dir: Path):
    data = json.loads((results_dir / "ivi_publish_results.json").read_text())
    diagnostics = data.get("diagnostics", {}).get("posterior_predictive", {})
    seeds = data.get("provenance", {}).get("rng_draws", {})
    return diagnostics, seeds


def main():
    parser = argparse.ArgumentParser(description="Generate PPC plots for IVI results")
    parser.add_argument("--results-dir", required=True, help="Path to results directory")
    parser.add_argument("--outfile", default="ppc_summary.png", help="Output figure filename")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    diagnostics, _ = load_ppc_data(results_dir)

    channels = ["lensing", "clocks", "pulsars"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, channel in zip(axes, channels):
        diag = diagnostics.get(channel)
        if diag is None:
            ax.text(0.5, 0.5, "NA", ha="center", va="center")
            ax.set_title(channel)
            continue
        observed = diag.get("observed")
        rep_vals = np.asarray(diag.get("rep_values", []), dtype=float)
        if rep_vals.size == 0:
            mean = diag.get("rep_mean", 0.0)
            std = diag.get("rep_std", 1.0)
            rep_vals = np.random.normal(mean, std if std else 1, size=1000)
        plot_ppc(ax, observed, rep_vals, f"PPC {channel}", "chi2")

    fig.tight_layout()
    outfile = results_dir / args.outfile
    fig.savefig(outfile, dpi=150)
    print(f"[WRITE] {outfile}")


if __name__ == "__main__":
    main()
