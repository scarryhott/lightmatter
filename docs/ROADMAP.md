# Rigor Roadmap

This document tracks the remaining high-priority items required to bring the
lightmatter IVI pipeline to publication-grade reproducibility. The checklist is
living; update it as tasks land.

## Blocking (before journal submission)

- [x] **Data availability bundle**  
  Added `data/quick/*` with checksums and wired `make quick` to use it.

- [x] **κ/G unit narrative & uncertainty model**  
  Extend docs/physical_calibration.md to spell out unit conversions for the
  Planck map → G(T) proxy, κ_ext scaling, and the environment bucket mapping.
  Document the uncertainty model choices per channel (lensing covariance,
  pulsar upper bounds, clock systematics).

- [x] **Posterior predictive visualisation**  
  Export PPC histograms/QQ plots to the publish bundle; reference them in the
  results README.

- [x] **Figure reproduction guide**  
  Add a walkthrough (“reproduce Figure 1–3”) in README or a dedicated doc,
  referencing `make full` and figure filenames.

## Important (should land soon)

- [x] **CI artifact upload**  
  Extend `.github/workflows/ci.yml` to run `make quick` on a small dataset and
  upload the resulting JSON/PNGs as build artifacts.

- [x] **MCMC / ablation seeds**  
  When Bayesian or ablation scripts are committed, ensure their RNG seeds are
  surfaced in results and CLI (similar to `--rng-seed` in run_analysis).

- [ ] **Zenodo DOI**  
  Replace the placeholder DOI in `CITATION.cff` once an official release is
  archived.

## Nice to have

- [ ] **Docker image**  
  Provide a Dockerfile (or uv/conda environment) that runs `make full` out of
  the box.

- [x] **Automated publication bundle**  
  `scripts/make_publication_bundle.py` runs the analysis and assembles the publish directory.

## Completed

- [x] Provenance + licensing (`DATA_SOURCES.md`, MIT license)
- [x] Environment pinning (`pyproject.toml`, `requirements.lock`)
- [x] CI (lint + pytest) and synthetic regression tests
- [x] Posterior predictive checks and RNG logging
- [x] Makefile targets for data/quick/full workflows
- [x] Architecture overview (`docs/ARCHITECTURE.md`)
