# Lightmatter IVI Analysis

Inference and validation infrastructure (IVI) for testing grain/flat
deformations of the weak-field lapse function via three observational
channels:

- Strong-lens time delays (TDCOSMO / H0LiCOW)
- Optical atomic clocks
- Pulsar timing residuals

The codebase provides:

- Reusable data loaders with HEALPix map sampling for sky-varying predictors
- Weighted least-squares fits with jackknife & permutation diagnostics
- Bayesian joint inference (emcee) for global parameters
- “Publish bundle” generation (results JSON, fit tables, provenance)

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock      # exact versions for reproducibility
# or: pip install -e .[dev]           # editable install with lint/test extras
python scripts/run_analysis.py --help
# tip: pass --rng-seed to make permutation and posterior predictive diagnostics reproducible

### Make targets

```bash
make data                # download Planck map (if needed) and verify checksums
make quick SEED=123      # fast run using data/quick/* (no jackknife/permutation)
make full RAD_MAP=path/to/planck.fits KAPPA_MAP=path/to/kappa.fits SEED=123
```

The `full` target expects both a radiation (`RAD_MAP`) and κ (`KAPPA_MAP`) HEALPix map.
Pass additional options such as `KAPPA_FIELD=0` or `RAD_FIELD=0` as needed.
```

### Reproducing the sky analysis

```bash
python run_analysis_with_planck.py
```

This helper downloads the Planck 857 GHz map, verifies it, and runs the
analysis with plots enabled, depositing outputs in
`results/planck_857_analysis`. Posterior predictive diagnostics are written to
`ppc_summary.png` alongside the other figures.

## Data provenance

See [DATA_SOURCES.md](DATA_SOURCES.md) for detailed origins, licenses, and
transformations applied to each dataset. Update the table whenever new data are
ingested. To confirm integrity run:

```bash
python scripts/check_data_integrity.py
```

which validates every tracked file against `data/SHA256SUMS`.

## Documentation

- `docs/methods_appendix.md` – mathematical background
- `docs/physical_calibration.md` – mapping observational quantities to IVI
  predictors
- `docs/ARCHITECTURE.md` – high-level data & module flow
- `docs/ROADMAP.md` – remaining rigor tasks / TODOs
- `docs/RELEASE.md` – release checklist and DOI registration steps
- `docs/FIGURE_REPRODUCTION.md` – regenerate figures from saved results
- `results/.../README.md` – auto-generated bundle notes

## Publication bundle automation

Use `scripts/make_publication_bundle.py` to run the full pipeline, render
figures, and collect artifacts into a timestamped directory. Example:

```bash
python scripts/make_publication_bundle.py \
  --config configs/default.yaml \
  --tdcosmo-csv data/tdcosmo_time_delays.csv \
  --kappa-csv data/kappa_ext.csv \
  --healpix /path/to/HFI_SkyMap_857_R3.00_full.fits \
  --nside 2048 \
  --clock-csv data/quick/clocks_quick.csv \
  --pulsar-csv data/quick/pulsars_quick.csv \
  --auto-I0
```

The command assembles `ivi_publish_results.json`, derived CSVs, figures
(`fig_*`, `ppc_summary.png`), and the methods appendix under `results/publish_*`.

## Contributing

1. Fork and create a feature branch.
2. Run `python -m compileall ivi_thickness scripts` before submitting.
3. Add tests for new behaviour (synthetic regression suite coming soon).
4. Open a pull request referencing relevant issues and datasets.

## License

This project is licensed under the terms of the MIT License
([LICENSE](LICENSE)).

## Citation

If you use this codebase, please cite it using the metadata in
[CITATION.cff](CITATION.cff) (Zenodo DOI placeholder: 10.5281/zenodo.0000000).
