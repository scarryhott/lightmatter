#!/usr/bin/env python3
"""
Generate publication-quality figures from IVI time-thickness analysis results.

This script reads the output from run_analysis.py and generates:
1. Per-channel scatter plots with model fits
2. Residual histograms
3. Text summary of key results

Usage:
    python scripts/plot_publication_figures.py --results-dir path/to/results
"""
import json, argparse, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def panel(ax, x, y, yhat, xlabel, ylabel, title):
    """Create a scatter plot with model fit line."""
    ax.scatter(x, y, s=14, alpha=0.8)
    order = np.argsort(x)
    ax.plot(np.array(x)[order], np.array(yhat)[order], lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def hist(ax, resid, title):
    """Create a histogram of residuals."""
    ax.hist(resid, bins=20, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("residual")
    ax.set_ylabel("count")

def main():
    # Parse command line arguments
    ap = argparse.ArgumentParser(description='Generate publication figures from IVI analysis results')
    ap.add_argument("--results-dir", required=True, help="Folder produced by run_analysis.py")
    args = ap.parse_args()
    
    # Resolve paths
    d = pathlib.Path(args.results_dir)
    
    # Load data
    meta = json.loads((d/"ivi_publish_results.json").read_text())
    dfL = pd.read_csv(d/"lensing_fit_points.csv")
    dfC = pd.read_csv(d/"clock_fit_points.csv")
    dfP = pd.read_csv(d/"pulsar_fit_points.csv")

    # Calculate residuals
    rL = dfL["residual_R"].to_numpy() - dfL["yhat_R"].to_numpy()
    rC = dfC["r"].to_numpy() - dfC["yhat_r"].to_numpy()
    rP = dfP["rms_residual_us"].to_numpy() - dfP["yhat_rms"].to_numpy()

    # ---- Figure 1: per-channel scatter with fit ----
    fig1, axs = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    
    # Lensing panel
    if "G_proxy" in dfL.columns:
        lens_x = dfL["G_proxy"]
        lens_xlabel = "G_proxy (I/I0)^γ"
    elif "rad_G" in dfL.columns:
        lens_x = dfL["rad_G"]
        lens_xlabel = "G(T) proxy"
    else:
        lens_x = np.arange(len(dfL))
        lens_xlabel = "Lens index"
    panel(axs[0], lens_x, dfL["residual_R"], dfL["yhat_R"],
          lens_xlabel, "Lensing residual R", "Lensing")
    
    # Clocks panel
    panel(axs[1], dfC["dG"], dfC["r"], dfC["yhat_r"],
          "ΔG(T)", "Δν/ν residual", "Clocks")
    
    # Pulsars panel
    panel(axs[2], dfP["x_distance_Fk"], dfP["rms_residual_us"], dfP["yhat_rms"],
          "distance × F(κ)", "RMS residual (μs)", "Pulsars")
    
    # Save figure 1
    fig1.savefig(d/"fig_per_channel.png", dpi=220)

    # ---- Figure 2: residual histograms ----
    fig2, axs2 = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    
    # Lensing residuals
    hist(axs2[0], rL, "Lensing residuals")
    
    # Clock residuals
    hist(axs2[1], rC, "Clock residuals")
    
    # Pulsar residuals
    hist(axs2[2], rP, "Pulsar residuals")
    
    # Save figure 2
    fig2.savefig(d/"fig_residual_hists.png", dpi=220)

    # ---- Figure 3: sky map histograms ----
    fig3 = None
    if "kappa_map" in dfL.columns or "G_raw" in dfL.columns:
        fig3, axs3 = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        # κ histogram
        if "kappa_map" in dfL.columns:
            data_kappa = dfL["kappa_map"].dropna().to_numpy()
            if data_kappa.size:
                axs3[0].hist(data_kappa, bins=25, alpha=0.85)
                axs3[0].set_title("κ map samples (lenses)")
                axs3[0].set_xlabel("κ_map")
                axs3[0].set_ylabel("count")
            else:
                axs3[0].text(0.5, 0.5, "No κ samples", ha="center", va="center")
                axs3[0].axis("off")
        else:
            axs3[0].text(0.5, 0.5, "κ map not available", ha="center", va="center")
            axs3[0].axis("off")

        # G_raw histogram
        if "G_raw" in dfL.columns:
            data_G = dfL["G_raw"].dropna().to_numpy()
            if data_G.size:
                axs3[1].hist(data_G, bins=25, alpha=0.85, color="tab:orange")
                axs3[1].set_title("G_raw samples (lenses)")
                axs3[1].set_xlabel("G_raw")
                axs3[1].set_ylabel("count")
            else:
                axs3[1].text(0.5, 0.5, "No G_raw samples", ha="center", va="center")
                axs3[1].axis("off")
        else:
            axs3[1].text(0.5, 0.5, "G_raw not available", ha="center", va="center")
            axs3[1].axis("off")

        fig3.savefig(d/"fig_sky_histograms.png", dpi=220)

    # ---- Figure 4: residuals vs sky proxies ----
    fig4 = None
    if "G_proxy" in dfL.columns and "F_kappa" in dfL.columns:
        fig4, axs4 = plt.subplots(1, 2, figsize=(12, 3.6), constrained_layout=True)
        # Residual vs G_proxy
        g_vals = dfL["G_proxy"].to_numpy()
        order_g = np.argsort(g_vals)
        axs4[0].scatter(g_vals, dfL["residual_R"], s=18, alpha=0.8)
        axs4[0].plot(g_vals[order_g], dfL["yhat_R"].to_numpy()[order_g], color="tab:red", lw=2)
        axs4[0].set_xlabel("G_proxy (I/I0)^γ")
        axs4[0].set_ylabel("Residual R")
        axs4[0].set_title("R vs G_proxy")

        # Residual vs F(kappa)
        fk_vals = dfL["F_kappa"].to_numpy()
        order_fk = np.argsort(fk_vals)
        axs4[1].scatter(fk_vals, dfL["residual_R"], s=18, alpha=0.8, color="tab:green")
        axs4[1].plot(fk_vals[order_fk], dfL["yhat_R"].to_numpy()[order_fk], color="k", lw=2)
        axs4[1].set_xlabel("F(κ)")
        axs4[1].set_ylabel("Residual R")
        axs4[1].set_title("R vs F(κ)")

        fig4.savefig(d/"fig_residual_vs_sky.png", dpi=220)

    # ---- Figure 5: mask retention counts ----
    maps = meta.get("provenance", {}).get("map_provenance")
    if maps is None:
        maps = meta.get("provenance", {}).get("maps", {})
    fig5 = None
    if isinstance(maps, dict) and maps:
        fig5 = plt.figure(figsize=(6.0, 3.2))
        ax5 = fig5.add_subplot(111)
        ax5.axis("off")
        lines = ["Sky map retention counts:"]
        kappa_info = maps.get("kappa") or {}
        rad_info = maps.get("radiation") or {}
        if kappa_info.get("lens_stats"):
            ls = kappa_info["lens_stats"]
            lines.append(f"κ map (lenses): {ls['n_kept']} / {ls['n_total']} kept")
        else:
            lines.append("κ map (lenses): n/a")
        if kappa_info.get("pulsar_stats"):
            ps = kappa_info["pulsar_stats"]
            lines.append(f"κ map (pulsars): {ps['n_kept']} / {ps['n_total']} kept")
        if rad_info.get("lens_stats"):
            ls = rad_info["lens_stats"]
            lines.append(f"G map (lenses): {ls['n_kept']} / {ls['n_total']} kept")
        if rad_info.get("pulsar_stats"):
            ps = rad_info["pulsar_stats"]
            lines.append(f"G map (pulsars): {ps['n_kept']} / {ps['n_total']} kept")
        ax5.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace")
        fig5.savefig(d/"fig_mask_counts.png", dpi=220, bbox_inches='tight')

    # ---- Figure 6: text summary (simple) ----
    fig6 = plt.figure(figsize=(7, 4))
    ax6 = fig6.add_subplot(111)
    
    # Format text summary
    txt = []
    txt.append("IVI time–thickness: combined fit")
    txt.append(f"chi2_red = {meta['combined']['chi2_reduced']:.3f}   dof = {meta['combined']['dof_total']}")
    
    # Add coefficients for each channel
    for ch in ["lensing", "clocks", "pulsars"]:
        ci = np.array(meta[ch]["coef_ci95"])
        beta = np.array(meta[ch]["coef"])
        names = meta[ch]["coef_names"]
        txt.append(f"\n[{ch.upper()}]")
        for n, b, c in zip(names, beta, ci):
            txt.append(f"  {n:22s} = {b:+.3e}  ± {c:.3e} (95% CI)")
    
    # Add text to figure
    ax6.axis("off")
    ax6.text(0.02, 0.98, "\n".join(txt), va="top", family="monospace")
    
    # Save figure 6
    fig6.savefig(d/"fig_text_summary.png", dpi=220, bbox_inches='tight')

    print(f"[WRITE] {d/'fig_per_channel.png'}")
    print(f"[WRITE] {d/'fig_residual_hists.png'}")
    if fig3 is not None:
        print(f"[WRITE] {d/'fig_sky_histograms.png'}")
    if fig4 is not None:
        print(f"[WRITE] {d/'fig_residual_vs_sky.png'}")
    if fig5 is not None:
        print(f"[WRITE] {d/'fig_mask_counts.png'}")
    print(f"[WRITE] {d/'fig_text_summary.png'}")

if __name__ == "__main__":
    main()
