#!/usr/bin/env python3
"""
Run the IVI time–thickness analysis with physical calibration.

Features:
- Sky-varying κ and radiation sampling from HEALPix maps
- Physical kappa_ext from CSV for lens systems
- Standardized outputs with physical units and provenance metadata

Usage:
  python scripts/run_analysis.py --config configs/physical.yaml \
    --kappa-map /path/to/kappa.fits --rad-map /path/to/radiation.fits \
    [--require-sky] [--no-plots] [--n-perm 200] [--no-jackknife]
"""
import argparse
import json
import os
import pathlib
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from ivi_thickness.model import Params
from ivi_thickness.data import DataHub
from ivi_thickness.fit import (
    fit_lensing_channel, 
    fit_clock_channel, 
    fit_pulsar_channel, 
    combined_assessment,
    jackknife_lensing,
    permutation_test
)
from ivi_thickness.plots import scatter_with_fit, residual_hist
from ivi_thickness.diagnostics import posterior_predictive_check

# Set up better plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def parse_args():
    parser = argparse.ArgumentParser(description='Run IVI time-thickness analysis')
    parser.add_argument('--config', default='configs/physical.yaml',
                       help='Path to config file (default: configs/physical.yaml)')
    parser.add_argument('--tdcosmo-csv', help='Path to TDCOSMO/H0LiCOW-style CSV with time-delay data')
    # κ-map parameters
    parser.add_argument('--kappa-map', default=None,
                       help='HEALPix FITS for convergence κ (e.g., Planck CMB lensing).')
    parser.add_argument('--kappa-field', type=int, default=0,
                       help='FITS field index for κ map (default: 0).')
    parser.add_argument('--kappa-nest', action='store_true',
                       help='Set if κ map is in NESTED ordering (default: RING).')
    parser.add_argument('--kappa-smooth-arcmin', type=float, default=None,
                       help='Gaussian smoothing FWHM (arcmin) applied to κ map before sampling.')
    parser.add_argument('--kappa-mask', default=None,
                       help='Optional HEALPix mask to apply to κ map (same NSIDE/order).')
    parser.add_argument('--kappa-mask-field', type=int, default=0,
                       help='FITS field index for κ mask (default: 0).')
    parser.add_argument('--kappa-scale', type=float, default=1.0,
                       help='Global scale factor applied to sampled κ before analysis.')
    
    # Radiation map parameters
    parser.add_argument('--rad-map', default=None,
                       help='HEALPix FITS for radiation field (e.g., Planck 857 GHz intensity).')
    parser.add_argument('--rad-field', type=int, default=0,
                       help='FITS field index for radiation map (default: 0).')
    parser.add_argument('--rad-nest', action='store_true',
                       help='Set if radiation map is in NESTED ordering (default: RING).')
    parser.add_argument('--rad-smooth-arcmin', type=float, default=None,
                       help='Gaussian smoothing FWHM (arcmin) applied to radiation map.')
    parser.add_argument('--rad-mask', default=None,
                       help='Optional HEALPix mask for radiation map (same NSIDE/order).')
    parser.add_argument('--rad-mask-field', type=int, default=0,
                       help='FITS field index for radiation mask (default: 0).')
    parser.add_argument('--auto-I0', action='store_true',
                       help='Normalize G by median intensity across sampled sightlines.')
    parser.add_argument('--I0', type=float, default=None,
                       help='Manual radiation normalization if --auto-I0 is not set.')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Exponent for radiation proxy: G = (I/I0)^gamma.')
    parser.add_argument('--require-sky', action='store_true',
                       help='Fail if κ or radiation sky maps are missing.')
    parser.add_argument('--kappa-csv', help='Path to CSV with kappa_ext values (lens_id,kappa_ext[,sigma_kappa])')
    
    # Clock and Pulsar data options
    parser.add_argument('--clock-csv', help='Path to CSV with optical clock residuals (comparison,epoch_mjd,r,T1_K,T2_K[,w])')
    parser.add_argument('--pulsar-csv', help='Path to CSV with pulsar timing residuals (pulsar,toa_mjd,resid_us,resid_err_us[,distance_kpc,ra_deg,dec_deg])')
    parser.add_argument('--rng-seed', type=int, default=0,
                       help='Seed controlling random draws for diagnostics (default: 0).')
    
    # Plotting options
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument('--plots', action='store_true', dest='make_plots',
                          help='Generate diagnostic plots (default)')
    plot_group.add_argument('--no-plots', action='store_false', dest='make_plots',
                          help='Skip generating plots')
    
    # Jackknife options
    jackknife_group = parser.add_mutually_exclusive_group()
    jackknife_group.add_argument('--jackknife', action='store_true', dest='run_jackknife',
                               help='Enable jackknife resampling (default)')
    jackknife_group.add_argument('--no-jackknife', action='store_false', dest='run_jackknife',
                               help='Disable jackknife resampling')
    
    # Set defaults for the mutually exclusive groups
    parser.set_defaults(make_plots=True, run_jackknife=True)
    parser.add_argument('--n-perm', type=int, default=200,
                       help='Number of permutations for significance testing (default: 200)')
    parser.add_argument('--output-dir', default='output',
                       help='Directory to save output files')
    return parser.parse_args()

def ensure_dir(directory: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def plot_ppc_summary(diagnostics_ppc: Dict[str, Any], output_dir: str) -> Optional[Path]:
    """Create posterior predictive check summary figure if data are available."""
    if not diagnostics_ppc:
        return None

    channels = ["lensing", "clocks", "pulsars"]
    fig, axes = plt.subplots(1, len(channels), figsize=(15, 4))

    for ax, channel in zip(axes, channels):
        diag = diagnostics_ppc.get(channel)
        if not isinstance(diag, dict):
            ax.text(0.5, 0.5, "NA", ha="center", va="center")
            ax.set_title(channel)
            continue

        observed = diag.get("observed")
        rep_vals = np.asarray(diag.get("rep_values", []), dtype=float)
        if rep_vals.size == 0:
            mean = diag.get("rep_mean", 0.0)
            std = diag.get("rep_std", 1.0)
            rep_vals = np.random.normal(mean, std if std else 1.0, size=1000)

        ax.hist(rep_vals, bins=30, alpha=0.7, label="replicates")
        if observed is not None:
            ax.axvline(observed, color="red", linestyle="--", label="observed")
        ax.set_title(f"PPC {channel}")
        ax.set_xlabel("χ²")
        ax.legend()

    fig.tight_layout()
    outfile = Path(output_dir) / "ppc_summary.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile

def fitresult_to_dict(tag: str, fr) -> Dict[str, Any]:
    """Convert a FitResult to a serializable dict with 95% CIs."""
    ci95 = (1.96 * fr.se).tolist()
    return {
        "channel": tag,
        "chi2": float(fr.chi2),
        "dof": int(fr.dof),
        "chi2_reduced": float(fr.chi2 / max(1, fr.dof)),
        "coef_names": fr.design_names,
        "coef": [float(x) for x in fr.beta.tolist()],
        "coef_ci95": [float(x) for x in ci95],
        "scale_info": fr.scale_info,
    }

def save_publication_bundle(
    outdir: pathlib.Path,
    params: Params,
    lens_fit,
    clock_fit,
    puls_fit,
    comb,
    df_lens: pd.DataFrame,
    df_clock: pd.DataFrame,
    df_psr: pd.DataFrame,
    datahub,
    args
) -> None:
    """Save publication-ready results bundle."""
    from ivi_thickness.model import F_kappa, G_temp
    
    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata bundle
    bundle = {
        "params": {
            "epsilon_grain": float(params.epsilon_grain),
            "epsilon_flat":  float(params.epsilon_flat),
            "E0_eV":         float(params.E0_eV),
            "kappa0":        float(params.kappa0),
            "p":             float(params.p),
            "q":             float(params.q),
        },
        "lensing": fitresult_to_dict("lensing", lens_fit),
        "clocks":  fitresult_to_dict("clocks",  clock_fit),
        "pulsars": fitresult_to_dict("pulsars", puls_fit),
        "combined": {
            "chi2_total": float(comb.chi2_total),
            "dof_total":  int(comb.dof_total),
            "chi2_reduced": float(comb.chi2_total / max(1, comb.dof_total)),
        },
        "diagnostics": args.get('diagnostics'),
        "figures": args.get('figures'),
        "provenance": {
            "config": args.get('config_path'),
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "code_version": "0.1.0",
            "rng_seed": args.get('rng_seed'),
            "rng_draws": args.get('rng_draws'),
            "tdcosmo_csv": args.get('tdcosmo_csv'),
            "clock_csv": args.get('clock_csv'),
            "pulsar_csv": args.get('pulsar_csv'),
            "kappa_csv": args.get('kappa_csv'),
            "kappa_map": args.get('kappa_map'),
            "kappa_field": args.get('kappa_field'),
            "kappa_nest": bool(args.get('kappa_nest', False)),
            "kappa_smooth_arcmin": args.get('kappa_smooth_arcmin'),
            "kappa_mask": args.get('kappa_mask'),
            "kappa_mask_field": args.get('kappa_mask_field'),
            "kappa_scale": args.get('kappa_scale'),
            "rad_map": args.get('rad_map'),
            "rad_field": args.get('rad_field'),
            "rad_nest": bool(args.get('rad_nest', False)),
            "rad_smooth_arcmin": args.get('rad_smooth_arcmin'),
            "rad_mask": args.get('rad_mask'),
            "rad_mask_field": args.get('rad_mask_field'),
            "auto_I0": bool(args.get('auto_I0', False)),
            "I0_input": args.get('I0_input'),
            "I0_used": args.get('I0_used'),
            "gamma": args.get('gamma', 1.0),
            "require_sky": bool(args.get('require_sky', False)),
            "map_provenance": args.get('map_provenance')
        }
    }
    
    # Write JSON
    (outdir / "ivi_publish_results.json").write_text(json.dumps(bundle, indent=2))
    
    # ---- Save per-channel fit points ----
    # Lensing
    R = (df_lens["dt_obs"] - df_lens["dt_gr"]) / df_lens["dt_gr"]
    if "G_proxy" in df_lens.columns and df_lens["G_proxy"].notna().any():
        GT_raw = df_lens["G_proxy"].to_numpy(float)
    elif "rad_proxy" in df_lens.columns and df_lens["rad_proxy"].notna().any():
        GT_raw = df_lens["rad_proxy"].to_numpy(float)
    else:
        GT_raw = np.array([datahub.los_radiation_proxy(ra, dec)
                          for ra, dec in zip(df_lens["ra_deg"], df_lens["dec_deg"])], float)

    if "kappa_ext" in df_lens.columns and df_lens["kappa_ext"].notna().any():
        kappa_vals = df_lens["kappa_ext"].to_numpy(float)
    elif "kappa_map" in df_lens.columns and df_lens["kappa_map"].notna().any():
        kappa_vals = df_lens["kappa_map"].to_numpy(float)
    elif "kappa_ext_scaled" in df_lens.columns and df_lens["kappa_ext_scaled"].notna().any():
        kappa_vals = df_lens["kappa_ext_scaled"].to_numpy(float)
    else:
        kappa_vals = np.array([datahub.env_kappa_proxy(dm) for dm in df_lens["dm_level"]], float)

    FK_raw = F_kappa(kappa_vals, params)

    X_L = np.column_stack([
        np.ones_like(R), 
        (GT_raw - np.mean(GT_raw))/max(np.std(GT_raw),1e-12),
        (FK_raw - np.mean(FK_raw))/max(np.std(FK_raw),1e-12)
    ])
    yhat_L = X_L @ lens_fit.beta
    
    lens_points = pd.DataFrame({
        "lens_id": df_lens["lens_id"],
        "pair_id": df_lens["pair_id"],
        "z_lens": df_lens["z_lens"], 
        "z_src": df_lens["z_src"],
        "ra_deg": df_lens["ra_deg"], 
        "dec_deg": df_lens["dec_deg"],
        "residual_R": R, 
        "yhat_R": yhat_L,
        "G_proxy": GT_raw,
        "F_kappa": FK_raw
    })
    if "G_raw" in df_lens.columns:
        lens_points["G_raw"] = df_lens["G_raw"]
    if "rad_map" in df_lens.columns:
        lens_points["rad_map"] = df_lens["rad_map"]
    if "kappa_map" in df_lens.columns:
        lens_points["kappa_map"] = df_lens["kappa_map"]
    if "kappa_ext" in df_lens.columns:
        lens_points["kappa_ext"] = df_lens["kappa_ext"]
    lens_points.to_csv(outdir / "lensing_fit_points.csv", index=False)

    # Clocks
    dG = 0.5*(G_temp(df_clock["T1"], params)-G_temp(df_clock["T2"], params))
    X_C = np.column_stack([
        np.ones_like(dG), 
        (dG - np.mean(dG))/max(np.std(dG),1e-12)
    ])
    yhat_C = X_C @ clock_fit.beta
    
    pd.DataFrame({
        "comparison": df_clock["comparison"], 
        "t_mjd": df_clock["t"],
        "r": df_clock["r"], 
        "dG": dG, 
        "yhat_r": yhat_C
    }).to_csv(outdir / "clock_fit_points.csv", index=False)

    # Pulsars
    if "kappa_ext" in df_psr.columns and df_psr["kappa_ext"].notna().any():
        kappa_psr = df_psr["kappa_ext"].to_numpy(float)
    elif "kappa_map" in df_psr.columns and df_psr["kappa_map"].notna().any():
        kappa_psr = df_psr["kappa_map"].to_numpy(float)
    else:
        kappa_psr = np.array([datahub.env_kappa_proxy(dm) for dm in df_psr["dm_level"]], float)
    FK_psr = F_kappa(kappa_psr, params)
    xP = df_psr["distance_kpc"].to_numpy(float) * FK_psr
    X_P = np.column_stack([
        np.ones_like(xP), 
        (xP - np.mean(xP))/max(np.std(xP),1e-12)
    ])
    yhat_P = X_P @ puls_fit.beta
    
    pulsar_points = pd.DataFrame({
        "pulsar": df_psr["pulsar"], 
        "distance_kpc": df_psr["distance_kpc"],
        "rms_residual_us": df_psr["rms_residual_us"],
        "x_distance_Fk": xP, 
        "yhat_rms": yhat_P
    })
    if "kappa_map" in df_psr.columns:
        pulsar_points["kappa_map"] = df_psr["kappa_map"]
    if "kappa_ext" in df_psr.columns:
        pulsar_points["kappa_ext"] = df_psr["kappa_ext"]
    if "G_proxy" in df_psr.columns:
        pulsar_points["G_proxy"] = df_psr["G_proxy"]
    if "G_raw" in df_psr.columns:
        pulsar_points["G_raw"] = df_psr["G_raw"]
    pulsar_points.to_csv(outdir / "pulsar_fit_points.csv", index=False)
    
    print(f"\n[WRITE] Results → {outdir}")
    print(f"        - ivi_publish_results.json")
    print(f"        - lensing_fit_points.csv")
    print(f"        - clock_fit_points.csv")
    print(f"        - pulsar_fit_points.csv")

def run_analysis(
    config_path: str,
    n_perm: int = 200,
    run_jackknife: bool = True,
    make_plots: bool = True,
    output_dir: str = 'output',
    tdcosmo_csv: str = None,
    clock_csv: str = None,
    pulsar_csv: str = None,
    kappa_csv: str = None,
    kappa_scale: float = 1.0,
    kappa_map: str = None,
    kappa_field: int = 0,
    kappa_nest: bool = False,
    kappa_smooth_arcmin: float = None,
    kappa_mask: str = None,
    kappa_mask_field: int = 0,
    rad_map: str = None,
    rad_field: int = 0,
    rad_nest: bool = False,
    rad_smooth_arcmin: float = None,
    rad_mask: str = None,
    rad_mask_field: int = 0,
    auto_I0: bool = False,
    I0: float = None,
    gamma: float = 1.0,
    require_sky: bool = False,
    rng_seed: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """Run the full IVI time-thickness analysis pipeline."""
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Initialize model parameters
    params = Params(
        epsilon_grain=cfg["params"]["epsilon_grain"],
        epsilon_flat=cfg["params"]["epsilon_flat"],
        E0_eV=cfg["params"]["E0_eV"],
        kappa0=cfg["params"]["kappa0"],
        p=cfg["params"]["p"],
        q=cfg["params"]["q"]
    )
    
    base_seed = int(rng_seed)
    rng_master = np.random.default_rng(base_seed)
    rng_draws: Dict[str, int] = {}
    
    # Initialize data hub
    datahub = DataHub(cfg['io']['data_dir'])
    
    # Load datasets
    if tdcosmo_csv:
        print(f"[INFO] Loading TDCOSMO time-delay data from {tdcosmo_csv}")
        df_lens = datahub.load_tdcosmo_csv(tdcosmo_csv)
        print(f"[INFO] Loaded {len(df_lens)} time-delay pairs")
        
        # Validate data quality
        if (df_lens["sig_gr"] <= 0).any():
            bad = df_lens.loc[df_lens["sig_gr"] <= 0, ["lens_id","pair_id","sig_gr"]]
            raise RuntimeError(f"Non-positive sig_gr in pairs:\n{bad}")
        if not np.isfinite(df_lens["ra_deg"]).all() or not np.isfinite(df_lens["dec_deg"]).all():
            raise RuntimeError("Non-finite RA/Dec in lens table.")
    else:
        print("[INFO] Using built-in H0LiCOW-like lensing dataset")
        df_lens = datahub.load_h0licow_like()
    
    # Load clock data
    if clock_csv:
        print(f"[INFO] Loading clock data from {clock_csv}")
        try:
            df_clock = datahub.load_clocks_csv(clock_csv)
            print(f"[INFO] Loaded {len(df_clock)} clock measurements from {clock_csv}")
        except Exception as e:
            print(f"[WARN] Failed to load clock data: {str(e)}")
            print("  Falling back to synthetic clock data")
            df_clock = datahub.load_clock_like()
    else:
        print("[INFO] Using synthetic clock dataset (demo)")
        df_clock = datahub.load_clock_like()
    
    # Load pulsar data
    if pulsar_csv:
        print(f"[INFO] Loading pulsar data from {pulsar_csv}")
        try:
            df_psr = datahub.load_pulsar_residuals_csv(pulsar_csv)
            print(f"[INFO] Loaded {len(df_psr)} pulsars from {pulsar_csv}")
            
            # Check for missing distances and warn
            if df_psr['distance_kpc'].isna().any():
                n_missing = df_psr['distance_kpc'].isna().sum()
                print(f"[WARN] {n_missing} pulsars have no distance - using default 1.0 kpc")
                
        except Exception as e:
            print(f"[WARN] Failed to load pulsar data: {str(e)}")
            print("  Falling back to NANOGrav-like dataset")
            df_psr = datahub.load_nanograv_like()
    else:
        print("[INFO] Using NANOGrav-like dataset (demo)")
        df_psr = datahub.load_nanograv_like()
    
    # Apply physical calibration if available
    if kappa_csv:
        print(f"Loading kappa_ext values from {kappa_csv}")
        df_lens = datahub.load_kappa_ext_csv(df_lens, kappa_csv, scale_to_kappa=kappa_scale)
        has_kappa = df_lens['kappa_ext'].notna()
        n_with = int(has_kappa.sum())
        print(f"[INFO] Loaded kappa_ext from {kappa_csv} (n={n_with}/{len(df_lens)})")
        
        if kappa_scale is not None:
            print(f"[INFO] Applied kappa_scale = {kappa_scale:g}")
            if 'kappa_ext_scaled' in df_lens.columns:
                print("  Using column 'kappa_ext_scaled' for analysis")
        
        # Use scaled kappa if available
        if 'kappa_ext_scaled' in df_lens.columns and kappa_scale is not None:
            df_lens['kappa_ext'] = df_lens['kappa_ext_scaled']
            print(f"  Applied kappa_scale = {kappa_scale} to published kappa_ext")
    
    # Enforce map availability if requested
    if require_sky and (not kappa_map or not rad_map):
        raise SystemExit("ERROR: --require-sky set but κ or radiation map path is missing.")

    map_provenance = {}
    I0_used = I0
    I0_used_psr = None

    # κ map sampling
    if kappa_map:
        print(f"[INFO] Sampling κ map from {kappa_map}")
        df_lens, good_kappa_lens, kappa_stats_lens = datahub.fill_kappa_from_map_for_lenses(
            df_lens,
            kappa_map_path=kappa_map,
            kappa_field=kappa_field,
            nest=kappa_nest,
            smooth_fwhm_arcmin=kappa_smooth_arcmin,
            mask_path=kappa_mask,
            mask_field=kappa_mask_field,
            kappa_scale=kappa_scale
        )
        df_lens = df_lens.loc[good_kappa_lens].reset_index(drop=True)

        df_psr, good_kappa_psr, kappa_stats_psr = datahub.fill_kappa_from_map_for_pulsars(
            df_psr,
            kappa_map_path=kappa_map,
            kappa_field=kappa_field,
            nest=kappa_nest,
            smooth_fwhm_arcmin=kappa_smooth_arcmin,
            mask_path=kappa_mask,
            mask_field=kappa_mask_field,
            kappa_scale=kappa_scale
        )
        df_psr = df_psr.loc[good_kappa_psr].reset_index(drop=True)

        kappa_scale_val = 1.0 if kappa_scale is None else float(kappa_scale)
        map_provenance["kappa"] = {
            "map_path": kappa_map,
            "field": int(kappa_field),
            "nest": bool(kappa_nest),
            "smooth_fwhm_arcmin": kappa_smooth_arcmin,
            "mask_path": kappa_mask,
            "mask_field": int(kappa_mask_field),
            "scale": kappa_scale_val,
            "lens_stats": kappa_stats_lens,
            "pulsar_stats": kappa_stats_psr
        }
    elif require_sky:
        raise SystemExit("ERROR: κ-map required but not provided.")
    else:
        print("[WARN] No κ map provided → falling back to published/env proxies.")
        map_provenance["kappa"] = None

    # Radiation map sampling
    if rad_map:
        print(f"[INFO] Sampling radiation map from {rad_map}")
        df_lens, good_rad_lens, I0_used, rad_stats_lens = datahub.fill_radiation_from_map_for_lenses(
            df_lens,
            rad_map_path=rad_map,
            rad_field=rad_field,
            nest=rad_nest,
            smooth_fwhm_arcmin=rad_smooth_arcmin,
            mask_path=rad_mask,
            mask_field=rad_mask_field,
            auto_I0=auto_I0,
            I0=I0,
            gamma=gamma
        )
        df_lens = df_lens.loc[good_rad_lens].reset_index(drop=True)

        df_psr, good_rad_psr, I0_used_psr, rad_stats_psr = datahub.fill_radiation_from_map_for_pulsars(
            df_psr,
            rad_map_path=rad_map,
            rad_field=rad_field,
            nest=rad_nest,
            smooth_fwhm_arcmin=rad_smooth_arcmin,
            mask_path=rad_mask,
            mask_field=rad_mask_field,
            auto_I0=auto_I0,
            I0=I0_used,
            gamma=gamma
        )
        df_psr = df_psr.loc[good_rad_psr].reset_index(drop=True)

        map_provenance["radiation"] = {
            "map_path": rad_map,
            "field": int(rad_field),
            "nest": bool(rad_nest),
            "smooth_fwhm_arcmin": rad_smooth_arcmin,
            "mask_path": rad_mask,
            "mask_field": int(rad_mask_field),
            "auto_I0": bool(auto_I0),
            "I0_used": float(I0_used),
            "I0_used_pulsars": float(I0_used_psr),
            "gamma": float(gamma),
            "lens_stats": rad_stats_lens,
            "pulsar_stats": rad_stats_psr
        }
    elif require_sky:
        raise SystemExit("ERROR: radiation map required but not provided.")
    else:
        print("[WARN] No radiation map provided → NOT using sky-varying G. (Consider --require-sky.)")
        map_provenance["radiation"] = None
    
    # Run main fits
    print("\n[1/4] Fitting lensing data...")
    lens_fit = fit_lensing_channel(
        df_lens, 
        params, 
        datahub,
        use_cov=cfg.get('lensing', {}).get('use_cov', True),
        rho_intra_lens=cfg.get('lensing', {}).get('rho_intra_lens', 0.5),
        standardize=cfg.get('lensing', {}).get('standardize', True)
    )
    
    print("\n[2/4] Fitting clock data...")
    clock_fit = fit_clock_channel(df_clock, params)
    
    print("\n[3/4] Fitting pulsar data...")
    puls_fit = fit_pulsar_channel(df_psr, params, datahub)
    
    # Combine results
    print("\n[4/4] Combining results...")
    comb = combined_assessment(lens_fit, clock_fit, puls_fit)
    
    # Run diagnostics
    results = {
        'params': {k: getattr(params, k) for k in ['epsilon_grain', 'epsilon_flat', 'E0_eV', 'kappa0', 'p', 'q']},
        'lens_fit': {
            **lens_fit.__dict__,
            'n_lenses': len(df_lens['lens_id'].unique()),
            'n_pairs': len(df_lens),
            'has_kappa_ext': 'kappa_ext' in df_lens.columns and df_lens['kappa_ext'].notna().any(),
            'has_sky_G': 'G_proxy' in df_lens.columns and df_lens['G_proxy'].notna().any(),
            'has_kappa_map': map_provenance.get("kappa") is not None
        },
        'clock_fit': clock_fit.__dict__,
        'pulsar_fit': puls_fit.__dict__,
        'combined': comb.__dict__,
        'config': cfg,
        'physical_calibration': {
            'kappa_csv': os.path.basename(kappa_csv) if kappa_csv else None,
            'maps': map_provenance,
            'require_sky': bool(require_sky)
        }
    }
    results['rng'] = {
        'base_seed': base_seed,
        'draws': dict(rng_draws)
    }
    results['physical_calibration'].setdefault('rng', results['rng'])
    results.setdefault('diagnostics', {})
    
    # Posterior predictive checks
    diagnostics_ppc = {}
    try:
        sigma_lens = np.array(lens_fit.scale_info.get('sigma_obs', {}).get('values', []), dtype=float)
        if sigma_lens.size == len(df_lens):
            R = (df_lens["dt_obs"].to_numpy(float) - df_lens["dt_gr"].to_numpy(float)) / df_lens["dt_gr"].to_numpy(float)
            yhat_lens = R - lens_fit.resid
            lens_ppc_seed = int(rng_master.integers(2**32))
            rng_draws['ppc_lensing'] = lens_ppc_seed
            ppc_rng = np.random.default_rng(lens_ppc_seed)
            diagnostics_ppc["lensing"] = posterior_predictive_check(R, yhat_lens, sigma_lens, ppc_rng)
    except Exception as exc:  # pragma: no cover - diagnostics failure should not halt run
        diagnostics_ppc["lensing_error"] = {"error": str(exc)}

    try:
        sigma_clock = np.array(clock_fit.scale_info.get('sigma_obs', {}).get('values', []), dtype=float)
        if sigma_clock.size == len(df_clock):
            r = df_clock["r"].to_numpy(float)
            yhat_clock = r - clock_fit.resid
            clock_ppc_seed = int(rng_master.integers(2**32))
            rng_draws['ppc_clocks'] = clock_ppc_seed
            ppc_rng = np.random.default_rng(clock_ppc_seed)
            diagnostics_ppc["clocks"] = posterior_predictive_check(r, yhat_clock, sigma_clock, ppc_rng)
    except Exception as exc:  # pragma: no cover
        diagnostics_ppc["clocks_error"] = {"error": str(exc)}

    try:
        sigma_psr = np.array(puls_fit.scale_info.get('sigma_obs', {}).get('values', []), dtype=float)
        if sigma_psr.size == len(df_psr):
            y_psr = df_psr["rms_residual_us"].to_numpy(float)
            yhat_psr = y_psr - puls_fit.resid
            pulsar_ppc_seed = int(rng_master.integers(2**32))
            rng_draws['ppc_pulsars'] = pulsar_ppc_seed
            ppc_rng = np.random.default_rng(pulsar_ppc_seed)
            diagnostics_ppc["pulsars"] = posterior_predictive_check(y_psr, yhat_psr, sigma_psr, ppc_rng)
    except Exception as exc:  # pragma: no cover
        diagnostics_ppc["pulsars_error"] = {"error": str(exc)}

    if diagnostics_ppc:
        results.setdefault('diagnostics', {})
        results['diagnostics']['posterior_predictive'] = diagnostics_ppc
        printable = {k: v for k, v in diagnostics_ppc.items() if isinstance(v, dict) and "two_tailed" in v}
        if printable:
            print("\n[DIAGNOSTICS] Posterior predictive checks (χ² two-tailed p-values):")
            for channel, info in printable.items():
                print(f"  {channel}: p = {info['two_tailed']:.3f}")
    # Run jackknife analysis for lensing
    if run_jackknife and len(df_lens['lens_id'].unique()) > 1:
        print("\n[DIAGNOSTICS] Running jackknife analysis...")
        jk_results = jackknife_lensing(df_lens, params, datahub)
        
        # Get the number of coefficients (excluding intercept)
        n_coeffs = len(jk_results[0]["beta"]) - 1 if jk_results and "beta" in jk_results[0] else 0
        
        # Initialize summary with None values
        jk_summary = {
            'n_jackknife': len(jk_results),
            'n_coefficients': n_coeffs,
            'all_betas': []
        }
        
        # Only try to compute statistics if we have coefficients
        if n_coeffs > 0:
            # Extract coefficients (excluding intercept)
            jk_betas = np.array([r["beta"][1:] for r in jk_results if r.get("beta") is not None])
            
            # Only proceed if we have valid results
            if len(jk_betas) > 0 and jk_betas.size > 0:
                jk_summary['all_betas'] = jk_betas.tolist()
                
                # Add coefficient statistics for each coefficient
                for i in range(n_coeffs):
                    if i < jk_betas.shape[1]:  # Check if we have this many coefficients
                        coeff_name = f'coeff_{i}'
                        jk_summary[f'{coeff_name}_mean'] = float(np.mean(jk_betas[:, i]))
                        jk_summary[f'{coeff_name}_std'] = float(np.std(jk_betas[:, i], ddof=1))
        
        # For backward compatibility, add b and c coefficients if we have 2+ coefficients
        if n_coeffs >= 2:
            jk_summary.update({
                'b_mean': jk_summary.get('coeff_0_mean', np.nan),
                'b_std': jk_summary.get('coeff_0_std', np.nan),
                'c_mean': jk_summary.get('coeff_1_mean', np.nan),
                'c_std': jk_summary.get('coeff_1_std', np.nan)
            })
        results.setdefault('diagnostics', {})['jackknife'] = jk_summary
        
        print(f"  Jackknife stability (n={jk_summary['n_jackknife']}, n_coeffs={jk_summary['n_coefficients']}):")
        
        # Print coefficient statistics if available
        for i in range(jk_summary.get('n_coefficients', 0)):
            coeff_name = f'coeff_{i}'
            if f'{coeff_name}_mean' in jk_summary and f'{coeff_name}_std' in jk_summary:
                mean = jk_summary[f'{coeff_name}_mean']
                std = jk_summary[f'{coeff_name}_std']
                print(f"  {coeff_name}: mean = {mean:.2e} ± {std:.1e}")
        
        # Print b and c coefficients if available (for backward compatibility)
        if 'b_mean' in jk_summary and 'c_mean' in jk_summary:
            print(f"  b (radiation): mean = {jk_summary['b_mean']:.2e} ± {jk_summary['b_std']:.1e}")
            print(f"  c (grain):     mean = {jk_summary['c_mean']:.2e} ± {jk_summary['c_std']:.1e}")
    
    # Run permutation test for lensing
    if n_perm > 0:
        print(f"\n[DIAGNOSTICS] Running permutation test (n={n_perm})...")
        results.setdefault('diagnostics', {})
        try:
            perm_seed = int(rng_master.integers(2**32))
            rng_draws['permutation'] = perm_seed
            perm_rng = np.random.default_rng(perm_seed)
            perm_results = permutation_test(df_lens, params, datahub, n_perm=n_perm, rng=perm_rng)
            results['diagnostics']['permutation'] = perm_results
            
            print(f"  Permutation p-values (n={n_perm}):")
            
            # Get the number of coefficients from the jackknife results if available
            n_coeffs = results.get('diagnostics', {}).get('jackknife', {}).get('n_coefficients', 0)
            
            # Print p-values for each coefficient
            if 'p_values' in perm_results and len(perm_results['p_values']) > 0:
                for i, p_val in enumerate(perm_results['p_values']):
                    if i == 0 and n_coeffs >= 1:
                        print(f"  b (radiation): p = {p_val:.4f}")
                    elif i == 1 and n_coeffs >= 2:
                        print(f"  c (grain):     p = {p_val:.4f}")
                    else:
                        print(f"  coeff_{i}:     p = {p_val:.4f}")
            else:
                print("  No p-values available from permutation test")
                
        except Exception as e:
            print(f"  Warning: Permutation test failed with error: {str(e)}")
            results['diagnostics']['permutation'] = {
                'error': str(e),
                'n_perm': n_perm
            }
    
    # Generate plots if requested
    if make_plots:
        print("\n[PLOTS] Generating diagnostic plots...")
        plot_results({
            'lens_fit': lens_fit,
            'clock_fit': clock_fit,
            'pulsar_fit': puls_fit,
            'combined': comb
        }, df_lens, df_clock, df_psr, datahub, params, output_dir)

    ppc_path = plot_ppc_summary(results.get('diagnostics', {}).get('posterior_predictive', {}), output_dir)
    if ppc_path:
        print(f"[WRITE] {ppc_path}")
        results.setdefault('figures', {})['posterior_predictive'] = str(ppc_path)
    
    # Save publication bundle
    outdir = Path(output_dir)
    save_publication_bundle(
        outdir=outdir,
        params=params,
        lens_fit=lens_fit,
        clock_fit=clock_fit,
        puls_fit=puls_fit,
        comb=comb,
        df_lens=df_lens,
        df_clock=df_clock,
        df_psr=df_psr,
        datahub=datahub,
        args={
            'config_path': config_path,
            'n_perm': n_perm,
            'run_jackknife': run_jackknife,
            'make_plots': make_plots,
            'output_dir': output_dir,
            'kappa_csv': kappa_csv,
            'kappa_map': kappa_map,
            'kappa_field': kappa_field,
            'kappa_nest': kappa_nest,
            'kappa_smooth_arcmin': kappa_smooth_arcmin,
            'kappa_mask': kappa_mask,
            'kappa_mask_field': kappa_mask_field,
            'kappa_scale': kappa_scale,
            'rad_map': rad_map,
            'rad_field': rad_field,
            'rad_nest': rad_nest,
            'rad_smooth_arcmin': rad_smooth_arcmin,
            'rad_mask': rad_mask,
            'rad_mask_field': rad_mask_field,
            'auto_I0': auto_I0,
            'I0_used': I0_used,
            'I0_input': I0,
            'gamma': gamma,
            'require_sky': require_sky,
            'map_provenance': map_provenance,
            'tdcosmo_csv': tdcosmo_csv,
            'clock_csv': clock_csv,
            'pulsar_csv': pulsar_csv,
            'diagnostics': results.get('diagnostics'),
            'figures': results.get('figures'),
            'rng_seed': base_seed,
            'rng_draws': dict(rng_draws)
        }
    )
    
    # Save results
    results_file = os.path.join(output_dir, 'ivi_results.pkl')
    import pickle
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n[COMPLETE] Results saved to {results_file}")
    
    return results

def plot_results(results: Dict[str, Any], df_lens: pd.DataFrame, df_clock: pd.DataFrame,
                df_psr: pd.DataFrame, datahub: DataHub, params: Params, output_dir: str) -> None:
    """Generate diagnostic plots for the analysis."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Lensing fit results
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1])
    
    # 1a. Time delay ratio vs. radiation proxy
    ax1 = fig.add_subplot(gs[0, 0])
    GT = (df_lens['dt_obs'] - df_lens['dt_gr']) / df_lens['dt_gr']
    
    # Handle scale_info safely
    if hasattr(results['lens_fit'], 'scale_info') and results['lens_fit'].scale_info is not None:
        GT_std = (GT - results['lens_fit'].scale_info['mu_GT']) / results['lens_fit'].scale_info['sd_GT']
        y_label = 'Standardized Time Delay Residuals'
    else:
        GT_std = GT  # Fallback to unscaled values
        y_label = 'Time Delay Residuals (Δt_obs - Δt_GR) / Δt_GR'
    
    # Create scatter plot
    x_proxy = None
    x_label = 'Lens Index'
    if 'G_proxy' in df_lens:
        x_proxy = df_lens['G_proxy']
        x_label = 'G_proxy (I/I0)^γ'
    elif 'rad_proxy' in df_lens:
        x_proxy = df_lens['rad_proxy']
        x_label = 'Radiation Proxy (I/I0)'
    if x_proxy is None:
        x_proxy = np.arange(len(GT_std))

    scatter = ax1.scatter(
        x_proxy,
        GT_std,
        c=df_lens['kappa_ext'] if 'kappa_ext' in df_lens else 'b',
        cmap='viridis' if 'kappa_ext' in df_lens else None,
        alpha=0.7,
        edgecolors='w'
    )
    
    # Add a simple linear fit if we have enough points
    if len(GT_std) > 2:
        x = np.asarray(x_proxy, dtype=float)
        mask = np.isfinite(x) & np.isfinite(GT_std)
        if mask.sum() > 2:
            coeffs = np.polyfit(x[mask], GT_std[mask], 1)
            x_vals = np.linspace(np.min(x[mask]), np.max(x[mask]), 100)
            ax1.plot(x_vals, np.polyval(coeffs, x_vals), 'r--', alpha=0.7,
                     label=f'Slope: {coeffs[0]:.2e}')
    
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title('Lensing: Time Delay Residuals')
    ax1.legend()
    
    # Add colorbar if kappa_ext is available
    if 'kappa_ext' in df_lens:
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('$\\kappa_{\\mathrm{ext}}$')
    
    # 1b. Residual histogram
    ax2 = fig.add_subplot(gs[0, 1])
    if hasattr(results['lens_fit'], 'resid'):
        ax2.hist(results['lens_fit'].resid, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Count')
        ax2.set_title('Residual Distribution')
    
    # 2. Parameter constraints
    ax3 = fig.add_subplot(gs[1, :])
    if 'lens_fit' in results and hasattr(results['lens_fit'], 'params'):
        params = results['lens_fit'].params
        param_names = ["b (rad)", "c (grain)"]
        param_vals = [params.get('b', np.nan), params.get('c', np.nan)]
        
        if hasattr(results['lens_fit'], 'param_errors'):
            errors = results['lens_fit'].param_errors
            y_err = [[errors.get('b', 0)], [errors.get('c', 0)]]
            ax3.errorbar(param_vals, range(len(param_vals)), xerr=y_err, fmt='o', 
                        capsize=5, markersize=8, color='darkgreen')
            ax3.set_yticks(range(len(param_names)))
            ax3.set_yticklabels(param_names)
        else:
            ax3.plot(param_vals, range(len(param_names)), 'o', color='darkgreen')
            ax3.set_yticks(range(len(param_names)))
            ax3.set_yticklabels(param_names)
        
        ax3.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Parameter Value')
        ax3.set_title('Parameter Constraints')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'diagnostic_plot.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - Saved diagnostic plot to {plot_file}")
    
    # If we have clock data with the required columns, create a separate plot for it
    if len(df_clock) > 0 and all(col in df_clock.columns for col in ['year', 'residual', 'sigma']):
        try:
            plt.figure(figsize=(10, 5))
            plt.errorbar(df_clock['year'], df_clock['residual'], 
                        yerr=df_clock['sigma'], fmt='o')
            plt.axhline(0, color='k', linestyle='--', alpha=0.3)
            plt.xlabel('Year')
            plt.ylabel('Clock Residual (s)')
            plt.title('Clock Residuals vs. Time')
            clock_plot_file = os.path.join(output_dir, 'clock_residuals.png')
            plt.savefig(clock_plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  - Saved clock residuals plot to {clock_plot_file}")
        except Exception as e:
            print(f"  - Could not generate clock residuals plot: {str(e)}")
    
    # If we have clock data with temperature columns, create temperature-dependent plot
    if (len(df_clock) > 0 and all(col in df_clock.columns for col in ['T1', 'T2', 'r']) and 
        'clock_fit' in results and hasattr(results['clock_fit'], 'scale_info')):
        try:
            dG = 0.5 * (G_temp(df_clock["T1"], params) - G_temp(df_clock["T2"], params))
            dG_std = (dG - results['clock_fit'].scale_info['dG_mean']) / results['clock_fit'].scale_info['dG_std']
            X_clock = np.column_stack([np.ones_like(dG_std), dG_std])
            y_clock_pred = X_clock @ results['clock_fit'].beta
            
            plt.figure(figsize=(10, 5))
            plt.scatter(dG, df_clock["r"], alpha=0.6, label='Data')
            plt.plot(dG, y_clock_pred, 'r-', label='Fit')
            plt.xlabel('0.5*(G(T1) - G(T2))')
            plt.ylabel('Clock Residual (s)')
            plt.title('Clock Residuals vs. Temperature Difference')
            plt.legend()
            temp_plot_file = os.path.join(output_dir, 'clock_temperature_fit.png')
            plt.savefig(temp_plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  - Saved clock temperature fit plot to {temp_plot_file}")
        except Exception as e:
            print(f"  - Could not generate clock temperature plot: {str(e)}")
    plt.ylabel('Clock Residual (fractional)')
    plt.title('Clock Residuals vs ΔG')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    
    clock_plot_file = os.path.join(output_dir, 'clock_diagnostics.png')
    plt.savefig(clock_plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Plots saved to {output_dir}/")

def main():
    args = parse_args()
    
    print("="*60)
    print("IVI TIME-THICKNESS ANALYSIS")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Plots: {'Enabled' if args.make_plots else 'Disabled'}")
    print(f"Permutations: {args.n_perm}")
    print(f"Jackknife: {'Enabled' if args.run_jackknife else 'Disabled'}")
    print(f"Output directory: {args.output_dir}")
    print(f"TDCosmo CSV: {args.tdcosmo_csv}")
    print(f"Require sky maps: {'Yes' if args.require_sky else 'No'}")
    print("-"*60)
    
    # Run the analysis with physical calibration
    results = run_analysis(
        args.config,
        n_perm=args.n_perm,
        run_jackknife=args.run_jackknife,
        make_plots=args.make_plots,
        output_dir=args.output_dir,
        tdcosmo_csv=args.tdcosmo_csv,
        clock_csv=args.clock_csv,
        pulsar_csv=args.pulsar_csv,
        kappa_csv=args.kappa_csv,
        kappa_scale=args.kappa_scale,
        kappa_map=args.kappa_map,
        kappa_field=args.kappa_field,
        kappa_nest=args.kappa_nest,
        kappa_smooth_arcmin=args.kappa_smooth_arcmin,
        kappa_mask=args.kappa_mask,
        kappa_mask_field=args.kappa_mask_field,
        rad_map=args.rad_map,
        rad_field=args.rad_field,
        rad_nest=args.rad_nest,
        rad_smooth_arcmin=args.rad_smooth_arcmin,
        rad_mask=args.rad_mask,
        rad_mask_field=args.rad_mask_field,
        auto_I0=args.auto_I0,
        I0=args.I0,
        gamma=args.gamma,
        require_sky=args.require_sky,
        rng_seed=args.rng_seed
    )
    
    # Print summary of physical calibration
    if 'physical_calibration' in results:
        pc = results['physical_calibration']
        print("\n=== Sky Map Summary ===")
        maps = pc.get('maps') or {}
        kappa_info = maps.get('kappa')
        rad_info = maps.get('radiation')
        kappa_path = kappa_info.get('map_path') if kappa_info else None
        rad_path = rad_info.get('map_path') if rad_info else None
        print(f"κ map: {kappa_path or 'None'}")
        if kappa_info:
            print(f"  field={kappa_info['field']} nest={kappa_info['nest']} smooth={kappa_info['smooth_fwhm_arcmin']}")
        print(f"G map: {rad_path or 'None'}")
        if rad_info:
            print(f"  field={rad_info['field']} nest={rad_info['nest']} smooth={rad_info['smooth_fwhm_arcmin']}")
            print(f"  gamma={rad_info['gamma']} I0_used={rad_info.get('I0_used')}")
        print(f"kappa_ext CSV: {pc.get('kappa_csv') or 'None'}")
        print(f"Require sky: {pc.get('require_sky')}")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(results['combined'])
    
    # Print clock constraint if available
    if 'clock_fit' in results:
        clock_info = results['clock_fit']
        if isinstance(clock_info, dict) and 'scale_info' in clock_info:
            scale_info = clock_info['scale_info']
            print("\n=== CLOCK CONSTRAINT ===")
            if 'eps_flat_est' in scale_info and 'eps_flat_se' in scale_info:
                print(f"ε_flat = {scale_info['eps_flat_est']:.2e} ± {1.96*scale_info['eps_flat_se']:.1e} (95% CL)")
            if 'eps_flat_upper_95' in scale_info:
                print(f"95% upper limit: |ε_flat| < {scale_info['eps_flat_upper_95']:.1e}")
        else:
            print("\n[INFO] No clock constraint information available")
    else:
        print("\n[INFO] No clock fit results available")
    
    # Print lensing coefficients with physical interpretation
    lens_info = results['lens_fit']
    
    print("\n=== LENSING COEFFICIENTS ===")
    
    # Handle case where lens_info is a dictionary
    if isinstance(lens_info, dict):
        if 'beta' in lens_info and len(lens_info['beta']) > 2:
            beta = lens_info['beta']
            print(f"b (radiation flattening): {beta[1]:.2e}")
            print(f"c (grain thickening):     {beta[2]:.2e}")
            
            # Try to get confidence intervals if available
            if 'ci' in lens_info and len(lens_info['ci']) > 2:
                b_ci = lens_info['ci'][1]
                c_ci = lens_info['ci'][2]
                if isinstance(b_ci, (list, tuple)) and len(b_ci) == 2:
                    print(f"  b 95% CI: [{b_ci[0]:.2e}, {b_ci[1]:.2e}]")
                if isinstance(c_ci, (list, tuple)) and len(c_ci) == 2:
                    print(f"  c 95% CI: [{c_ci[0]:.2e}, {c_ci[1]:.2e}]")
            
            # Check for parameter errors if CIs aren't available
            elif 'param_errors' in lens_info:
                errors = lens_info['param_errors']
                if 'b' in errors and 'c' in errors:
                    print(f"  b error: ±{errors['b']:.2e}")
                    print(f"  c error: ±{errors['c']:.2e}")
        else:
            print("Could not extract lensing coefficients from results")
            print("Available keys:", list(lens_info.keys()))
    
    # Handle case where lens_info is an object with methods
    elif hasattr(lens_info, 'get_ci') and hasattr(lens_info, 'beta'):
        try:
            b_ci_lo, b_ci_hi = lens_info.get_ci(1)  # b coefficient (radiation)
            c_ci_lo, c_ci_hi = lens_info.get_ci(2)  # c coefficient (grain)
            print(f"b (radiation flattening): {lens_info.beta[1]:.2e}  [95% CI: {b_ci_lo:.2e}, {b_ci_hi:.2e}]")
            print(f"c (grain thickening):     {lens_info.beta[2]:.2e}  [95% CI: {c_ci_lo:.2e}, {c_ci_hi:.2e}]")
        except Exception as e:
            print(f"Error extracting lensing coefficients: {str(e)}")
    else:
        print("Unsupported lens_info type:", type(lens_info))

if __name__ == "__main__":
    main()
