# IVI Architecture Overview

This document summarizes how data, models, and scripts interact inside the
lightmatter repository. The intent is to help reviewers and collaborators trace
exactly how inputs propagate to published results.

```
          ┌────────────────────┐
          │   configs/*.yaml   │
          └────────┬───────────┘
                   │ (model hyperparameters, IO paths)
                   ▼
┌────────────────────────────────────────────────────────────────────┐
│                          scripts/run_analysis.py                   │
│                                                                    │
│ 1. Load params & seed RNG                                          │
│ 2. Instantiate DataHub                                             │
│ 3. Load lens/clock/pulsar tables                                   │
│ 4. Sample κ-map / radiation map (ivi_thickness/data.py + maps.py)  │
│ 5. Fit channels (ivi_thickness/fit.py)                             │
│ 6. Diagnostics (jackknife, permutation, posterior predictive)      │
│ 7. Save publish bundle + figures                                   │
└────────────────────────────────────────────────────────────────────┘
                   │                         │
                   │                         │
                   │                         │
                   ▼                         ▼
       ┌───────────────────────┐   ┌──────────────────────────┐
       │  ivi_thickness/data.py│   │  ivi_thickness/fit.py    │
       └──────────┬────────────┘   └──────────┬───────────────┘
                  │                           │
                  │ HEALPix sampling helpers   │ Channel-specific WLS/GLS
                  │ CSV loaders                │ diagnostics & scale info
                  ▼                           ▼
       ┌───────────────────────┐   ┌──────────────────────────┐
       │  ivi_thickness/maps.py│   │  ivi_thickness/model.py  │
       └──────────┬────────────┘   └──────────┬───────────────┘
                  │                           │
                  │ HEALPix utilities         │ Core IVI functions
                  ▼                           ▼
          External FITS maps           Shape functions F(κ), G(T),
                                       lapse law, predictors
```

## Key components

### Configurations
- `configs/*.yaml` define physical parameters (`epsilon_flat`, `p`, `q`, …) and
  IO defaults (data directory, analysis options).

### Data ingestion (`ivi_thickness/data.py`)
- CSV loaders validate schema and units for lenses, clocks, and pulsars.
- Map sampling helpers wrap `ivi_thickness/maps.py` to add `G_proxy`,
  `kappa_map`, normalization statistics, and masks.
- Synthetic fallbacks live here for demo/quick runs; provenance records which
  path was taken.

### Map utilities (`ivi_thickness/maps.py`)
- Thin layer over `healpy` for loading, smoothing, masking, and sampling
  HEALPix maps.
- Shared by κ and radiation proxies.

### Modeling (`ivi_thickness/model.py`)
- Contains the theoretical shape functions and predicted observables:
  - `F_kappa(κ)` for grain thickening,
  - `G_temp(T)` for radiation flattening,
  - Lapse deformation helpers used by Bayesian modules.

### Fitting & diagnostics (`ivi_thickness/fit.py`, `ivi_thickness/diagnostics.py`)
- Weighted least squares / GLS for each observational channel.
- Jackknife, permutation tests, ridge fallback, and posterior predictive checks.
- All observers’ σ values and RNG seeds logged in `scale_info` and results JSON.

### Runners & publication artifacts (`scripts/…`)
- `run_analysis.py`: orchestrates end-to-end analysis; accepts `--rng-seed`,
  map paths, and other CLI overrides.
- `plot_publication_figures.py`: turns fit tables into diagnostic figures.
- `make_publication_bundle.py` (future work): packaging for submission.
- `download_planck_map.py`, `verify_map.py`: provenance helpers.

### Automation
- `Makefile` introduces `make quick`, `make full`, `make data`, and checksum
  verification targets.
- `.github/workflows/ci.yml` runs lint + pytest on every push/PR.

## Data flow at a glance
1. **Inputs**: CSVs in `data/`, HEALPix FITS (Planck κ, 857 GHz).
2. **Sampling**: `DataHub.fill_*_from_map*` builds per-object predictors
   (`G_proxy`, `kappa_ext`, masks, medians).
3. **Fitting**:
   - Lensing: residual regression on `G_proxy` and `F_kappa`.
   - Clocks: slope on ΔG with seeds recorded.
   - Pulsars: distance × `F_kappa` bounds.
4. **Diagnostics**: jackknife, permutation tests, posterior predictive checks
   (all seeded).
5. **Outputs**: `results/<label>/ivi_publish_results.json`, CSV fit tables,
   reproducibility metadata, optional figures (`scripts/plot_publication_figures.py`).

## Reproducibility hooks
- SHA256 checksums (`scripts/check_data_integrity.py`).
- RNG seed logging in both `results[...]` dictionary and publish bundle
  provenance (`rng_seed`, `rng_draws`).
- `Makefile` targets to run quick/full workflows deterministically.
- CI ensures code compilation and tests remain green.
