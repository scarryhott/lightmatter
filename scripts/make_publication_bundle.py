#!/usr/bin/env python3
"""
Create a self-contained publication bundle from IVI time-thickness analysis.

This script:
1. Runs the full analysis pipeline
2. Generates publication-quality figures
3. Creates a README with provenance information
4. Copies the methods appendix

Usage:
    python scripts/make_publication_bundle.py \
      --config configs/default.yaml \
      --tdcosmo-csv data/tdcosmo_time_delays.csv \
      --kappa-csv data/kappa_ext.csv \
      --healpix /path/to/planck_857.fits \
      --nside 2048 --nest \
      --auto-I0 --gamma 1.0 \
      --clock-csv data/clocks_timeseries.csv \
      --pulsar-csv data/pulsar_residuals.csv
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import textwrap
import time

def run(cmd):
    """Run a shell command with output to stdout."""
    print(">>", " ".join(cmd))
    sys.stdout.flush()
    return subprocess.check_call(cmd)

def main():
    # Parse command line arguments
    ap = argparse.ArgumentParser(description='Create a publication bundle for IVI time-thickness results')
    
    # Required arguments
    ap.add_argument("--config", required=True, help="Path to config file")
    ap.add_argument("--tdcosmo-csv", required=True, help="Path to TDCOSMO time delays CSV")
    ap.add_argument("--kappa-csv", required=True, help="Path to kappa_ext values CSV")
    ap.add_argument("--healpix", required=True, help="Path to HEALPix FITS file")
    ap.add_argument("--nside", type=int, required=True, help="HEALPix NSIDE parameter")
    
    # Optional arguments with defaults
    ap.add_argument("--nest", action="store_true", help="Use NESTED HEALPix ordering")
    ap.add_argument("--auto-I0", action="store_true", help="Auto-set I0 from data")
    ap.add_argument("--I0-mjysr", type=float, default=None, help="Reference intensity [MJy/sr]")
    ap.add_argument("--gamma", type=float, default=1.0, help="Radiation field exponent")
    ap.add_argument("--clock-csv", required=True, help="Path to clock residuals CSV")
    ap.add_argument("--pulsar-csv", required=True, help="Path to pulsar residuals CSV")
    ap.add_argument("--kappa-scale", type=float, default=1.0, help="Scale factor for kappa values")
    ap.add_argument("--outdir", help="Output directory (default: results/publish_TIMESTAMP)")
    
    args = ap.parse_args()
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = pathlib.Path(args.outdir or f"results/publish_{timestamp}")
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Creating publication bundle in: {outdir}")
    
    # 1) Run the analysis pipeline
    print("\n[1/3] Running IVI time-thickness analysis...")
    runner = [
        sys.executable, "scripts/run_analysis.py",
        "--config", args.config,
        "--tdcosmo-csv", args.tdcosmo_csv,
        "--kappa-csv", args.kappa_csv,
        "--kappa-scale", str(args.kappa_scale),
        "--healpix", args.healpix,
        "--nside", str(args.nside),
        "--gamma", str(args.gamma),
        "--clock-csv", args.clock_csv,
        "--pulsar-csv", args.pulsar_csv,
        "--output-dir", str(outdir),
        "--no-plots"  # We'll generate publication plots separately
    ]
    
    # Add optional flags
    if args.nest:
        runner.append("--nest")
    if args.auto_I0:
        runner.append("--auto-I0")
    if args.I0_mjysr is not None:
        runner.extend(["--I0-mjysr", str(args.I0_mjysr)])
    
    run(runner)
    
    # 2) Generate publication figures
    print("\n[2/3] Generating publication figures...")
    run([
        sys.executable, "scripts/plot_publication_figures.py",
        "--results-dir", str(outdir)
    ])
    
    # 3) Copy methods appendix if it exists
    print("\n[3/3] Finalizing bundle...")
    docs_src = pathlib.Path("docs/methods_appendix.md")
    if docs_src.exists():
        shutil.copy(docs_src, outdir / "methods_appendix.md")
    
    # 4) Create a README with provenance
    meta = json.loads((outdir / "ivi_publish_results.json").read_text())
    
    readme = textwrap.dedent(f"""
    # IVI Time–Thickness — Publication Bundle

    **Generated**: {timestamp}

    ## Contents
    - `ivi_publish_results.json`  - Parameters, coefficients, χ², and provenance
    - `lensing_fit_points.csv`    - Lensing data points and model predictions
    - `clock_fit_points.csv`      - Clock data points and model predictions
    - `pulsar_fit_points.csv`     - Pulsar data points and model predictions
    - `fig_per_channel.png`       - Figure: Per-channel model fits
    - `fig_residual_hists.png`    - Figure: Residual distributions
    - `fig_text_summary.png`      - Figure: Text summary of results
    - `methods_appendix.md`       - Methods appendix for publication

    ## Provenance
    - **Config file**: `{meta['provenance']['config']}`
    - **TDCOSMO data**: `{meta['provenance']['tdcosmo_csv']}`
    - **Kappa values**: `{meta['provenance']['kappa_csv']}` (scale={meta['provenance']['kappa_scale']})
    - **HEALPix map**: `{meta['provenance']['healpix']}` (nside={meta['provenance']['nside']}, nest={meta['provenance']['nest']})
    - **Radiation field**: I0={meta['provenance']['I0_mjysr']} (auto_I0={meta['provenance']['auto_I0']}), γ={meta['provenance']['gamma']}
    - **Clock data**: `{meta['provenance']['clock_csv']}`
    - **Pulsar data**: `{meta['provenance']['pulsar_csv']}`

    ## Reproduce Results
    ```bash
    # Regenerate figures from saved data
    python scripts/plot_publication_figures.py --results-dir {outdir}
    
    # Rerun full analysis
    python scripts/run_analysis.py \
      --config {meta['provenance']['config']} \
      --tdcosmo-csv {meta['provenance']['tdcosmo_csv']} \
      --kappa-csv {meta['provenance']['kappa_csv']} \
      --healpix {meta['provenance']['healpix']} \
      --nside {meta['provenance']['nside']} \
      --gamma {meta['provenance']['gamma']} \
      --clock-csv {meta['provenance']['clock_csv']} \
      --pulsar-csv {meta['provenance']['pulsar_csv']} \
      --output-dir {outdir}
    """).strip()
    
    # Write README
    (outdir / "README.md").write_text(readme)
    
    print(f"\n[SUCCESS] Publication bundle complete: {outdir}")
    print("\nTo create a ZIP archive for sharing:")
    print(f"  cd {outdir.parent} && zip -r {outdir.name}.zip {outdir.name}")

if __name__ == "__main__":
    main()
