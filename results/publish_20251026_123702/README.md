# IVI Time–Thickness — Publication Bundle

**Generated**: 20251026_123702

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
- **Config file**: `configs/physical.yaml`
- **TDCOSMO data**: `data/tdcosmo_time_delays.csv`
- **Kappa values**: `data/kappa_ext.csv` (scale=1.0)
- **HEALPix map**: `/path/to/planck_857.fits` (nside=2048, nest=True)
- **Radiation field**: I0=None (auto_I0=True), γ=1.0
- **Clock data**: `data/clocks_timeseries.csv`
- **Pulsar data**: `data/pulsar_residuals.csv`

## Reproduce Results
```bash
# Regenerate figures from saved data
python scripts/plot_publication_figures.py --results-dir results/publish_20251026_123702

# Rerun full analysis
python scripts/run_analysis.py       --config configs/physical.yaml       --tdcosmo-csv data/tdcosmo_time_delays.csv       --kappa-csv data/kappa_ext.csv       --healpix /path/to/planck_857.fits       --nside 2048       --gamma 1.0       --clock-csv data/clocks_timeseries.csv       --pulsar-csv data/pulsar_residuals.csv       --output-dir results/publish_20251026_123702