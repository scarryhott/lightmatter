# IVI Time-Thickness Analysis with Physical Calibration

This document describes how to use the physical calibration features for the IVI time-thickness analysis, including HEALPix map support for radiation fields and kappa_ext values from external catalogs.

## Overview

The physical calibration pipeline adds two key features:

1. **Radiation Field from HEALPix Maps**
   - Use full-sky intensity maps (e.g., Planck 857 GHz, IRIS 100 μm) as a proxy for the radiation field
   - Convert intensities to dimensionless G(T) using a power-law scaling
   - Properly handle coordinate systems and interpolation

2. **External Convergence (κ_ext) from Catalogs**
   - Import published κ_ext values for lens systems
   - Fall back to environment-based proxies when κ_ext is not available
   - Scale and transform κ_ext to match the IVI model's expectations

## Setup

### Dependencies

```bash
# Required for HEALPix support
pip install healpy astropy

# Optional: for better performance with large maps
pip install astropy-healpix
```

### Data Preparation

1. **HEALPix Map**
   - Download a full-sky intensity map (e.g., [Planck 857 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/COM_CompMap_dust-commrul_2048_R2.00.fits))
   - The map should be in FITS format with standard HEALPix structure
   - Supported coordinate systems: Galactic, Ecliptic, or Equatorial

2. **Kappa_ext Catalog**
   - Prepare a CSV file with columns: `lens_id,kappa_ext[,sigma_kappa]`
   - Example:
     ```csv
     lens_id,kappa_ext,sigma_kappa
     B1608+656,0.03,0.01
     HE0435-1223,0.04,0.02
     ```

## Configuration

Edit `configs/physical.yaml` to set up the physical calibration:

```yaml
# Physical calibration configuration
io:
  data_dir: "./time_thickness_data"

# Default parameter values
params:
  epsilon_grain: 1.0e-18
  epsilon_flat: -1.0e-19
  E0_eV: 10.0
  kappa0: 1.0e20
  p: 1.0
  q: 1.0

# Radiation field configuration
radiation:
  # Reference intensity [MJy/sr] - set to median of your lens sightlines
  I0_mjysr: 1.0
  # Exponent for intensity scaling
  gamma: 1.0

# Lensing configuration
lensing:
  use_cov: true
  rho_intra_lens: 0.5  # Correlation coefficient for time-delay pairs
  standardize: true    # Standardize predictors (recommended)
```

## Running the Analysis

### Basic Usage

```bash
python scripts/run_analysis.py \
  --config configs/physical.yaml \
  --healpix path/to/planck_857.fits \
  --kappa-csv path/to/kappa_ext.csv \
  --output-dir output/physical_calibration
```

### Advanced Options

- `--nside NSIDE`: Manually set HEALPix resolution (default: auto-detect)
- `--nest`: Use NESTED ordering for HEALPix map (default: RING)
- `--n-perm 1000`: Increase number of permutations for significance testing
- `--no-jackknife`: Skip the jackknife analysis for faster execution
- `--no-plots`: Skip generating diagnostic plots

## Output

The analysis produces several outputs in the specified output directory:

- `results.json`: Complete analysis results in JSON format
- `corner.png`: Corner plot of parameter constraints
- `lensing_fit.png`: Lensing residual vs. model prediction
- `clock_fit.png`: Clock comparison results
- `pulsar_fit.png`: Pulsar timing residuals
- `jackknife/`: Directory with jackknife analysis results

## Interpreting Results

### Radiation Field (G(T))
- The radiation proxy is defined as `G(T) = (I / I0)^γ`, where the intensity `I`
  is read from the HEALPix map in units of MJy/sr. We assume the Planck HFI maps
  are beam-corrected; optional smoothing parameters applied in
  `fill_radiation_from_map_for_*` are logged in the provenance. Use
  `--auto-I0` to normalise by the median lens sightline intensity, or specify
  `--I0` explicitly. The exponent `γ` defaults to 1 but can be tuned to mimic a
  more physical spectral response.

- Because the Planck intensity is not a direct thermometer, treat `G` as a
  dimensionless proxy. Regression outputs are reported in σ-units, and the
  provenance JSON captures `I0_used`, smoothing, masks, and map filenames.

### Convergence (κ)
- Published κ_ext values (e.g., H0LiCOW) are ingested from CSV via
  `fill_kappa_from_map_for_*`. The model uses `F(κ) = (κ/κ0)^p`; the default
  `κ0` comes from `configs/physical.yaml` and should match the scaling of the
  input catalogue.

- When κ_ext is absent, the pipeline falls back to environment buckets
  (`low`, `medium`, `high`), mapping to fiducial κ values (`1e19`, `5e19`,
  `1e20`). These defaults live in `DataHub.env_kappa_proxy`; update them to
  match the lens sample under study and document any changes here.

- If a HEALPix κ map is provided (`--kappa-map`), sampled values are stored in
  `kappa_map` columns and logged in the provenance (`nside`, smoothing, masks).
  Global scaling via `--kappa-scale` is also captured in the output JSON.

### Parameter Constraints
- `epsilon_flat`: Constrained by the clock comparison data
- `epsilon_grain`: Constrained by the lensing and pulsar data
- `E0`, `p`, `q`: Shape parameters (may be fixed or varied in the fit)

## Troubleshooting

### HEALPix Map Issues
- **Error**: "Could not load HEALPix map"
  - Check that the file exists and is readable
  - Verify that healpy is installed: `pip install healpy`
  - Try a different NSIDE or ordering scheme (RING vs NESTED)

### Kappa_ext Issues
- **Warning**: "No kappa_ext values found for some lenses"
  - This is expected if some lenses don't have published κ_ext values
  - The analysis will fall back to environment-based proxies

### Memory Issues
- For large HEALPix maps (NSIDE > 1024), you may need to increase Python's memory limit
- Try using a lower NSIDE map if memory is a concern

## Example

```bash
# Run with Planck 857 GHz map and H0LiCOW kappa_ext values
python scripts/run_analysis.py \
  --config configs/physical.yaml \
  --healpix data/planck_857_2048.fits \
  --kappa-csv data/h0licow_kappa_ext.csv \
  --n-perm 1000 \
  --output-dir output/planck_physical
```

## References

1. Planck Collaboration 2016, A&A, 594, A13 (2016) - Planck 2015 results. XIII.
2. M. Millea et al. 2020, ApJ, 901, 4 - H0LiCOW XIII. A 2.4% measurement of H0
3. G. F. Lewis 2016, MNRAS, 463, 288 - The trouble with Hubble: Local versus global expansion rates
