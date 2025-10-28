# Reproducing Figures

This guide explains how to regenerate the figures shipped with the IVI
publish bundle.

## Quick sanity run

```bash
make quick SEED=123   # uses the committed data/quick/ subset
```

Outputs (in `results/quick_run/`):
- `fig_per_channel.png`
- `fig_residual_hists.png`
- `fig_residual_vs_sky.png`
- `fig_mask_counts.png`
- `fig_text_summary.png`
- `ppc_summary.png`

## Full real-data run

```bash
make full \
  RAD_MAP=/path/to/HFI_SkyMap_857_R3.00_full.fits \
  KAPPA_MAP=/path/to/COM_CompMap_Lensing_4096_R3.00_kappa.fits \
  SEED=123
```

Outputs (in `results/full_run/`): same as above plus any additional diagnostics.
