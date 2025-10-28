# Data Sources & Licensing

This repository depends on a mix of public survey products and user-supplied
observational catalogs. The table below records the origin, license, and any
transformations applied before ingest. Please update this document whenever a
new dataset is introduced. All repository-tracked files are checksummed; run
`python scripts/check_data_integrity.py` to verify they match
`data/SHA256SUMS`.

| Dataset | Local Path | Source & DOI/URL | License | Notes / Processing |
|---------|------------|------------------|---------|--------------------|
| Planck 857 GHz intensity map (HFI, R2.02) | `data/planck/HFI_SkyMap_857_2048_R2.02_full.fits` | Planck Collaboration (2015), *Planck 2015 results. X. Diffuse component separation*, ESA Planck Legacy Archive: <https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/> | ESA Planck data policy (public domain with acknowledgement) | Used as the sky-varying radiation proxy `G_proxy`. Internally smoothed with optional Gaussian kernel (specified at runtime) and normalized by the median sightline intensity when `--auto-I0` is set. |
| Planck CMB lensing convergence (κ) map (R3.00) | `data/planck/COM_CompMap_Lensing_4096_R3.00_kappa.fits` *(user-supplied path)* | Planck Collaboration (2018), *Planck 2018 results. VIII. Gravitational lensing*, ESA Planck Legacy Archive | ESA Planck data policy | Samples κ along lens / pulsar sightlines to build `F_kappa`. Masking and smoothing parameters are logged in `ivi_publish_results.json`. |
| TDCOSMO/H0LiCOW time-delay measurements | `data/tdcosmo_time_delays.csv` | Wong et al. (2020), *H0LiCOW XIII*, DOI: [10.1093/mnras/stz3094](https://doi.org/10.1093/mnras/stz3094) | TDCOSMO collaboration data release (CC BY 4.0) | CSV columns are harmonised to the internal schema (`lens_id`, `pair_id`, delays, uncertainties, sky positions). No additional filtering beyond numeric sanity checks. |
| TDCOSMO quick subset | `data/quick/tdcosmo_quick.csv` | Derived subset of the above (rows for B1608+656, RXJ1131-1231) | CC BY 4.0 | Used for `make quick` smoke test. Checksummed in `data/SHA256SUMS`. |
| External convergence catalog | `data/kappa_ext.csv`, `data/h0licow_kappa_ext.csv` | Suyu et al. (2013), Rusu et al. (2020), DOI: [10.1088/0004-637X/766/2/70](https://doi.org/10.1088/0004-637X/766/2/70) | CC BY 3.0 (where released) | Provides published κ_ext estimates for lenses. Optional global rescaling via `--kappa-scale` is recorded in published bundles. |
| κ_ext quick subset | `data/quick/kappa_ext_quick.csv` | Derived subset of the above | CC BY 3.0 | Matches lenses in the quick dataset; checksummed. |
| Quick clock residuals | `data/quick/clocks_quick.csv` | This repository | MIT License | Small committed set for `make quick`; based on published temperature ranges. |
| Quick pulsar residuals | `data/quick/pulsars_quick.csv` | This repository | MIT License | Minimal pulsar TOA subset for `make quick`. |
| Synthetic clock residuals | generated in `ivi_thickness/data.py` | This repository | MIT License | Placeholder dataset used when `--clock-csv` is omitted. Replace with real optical clock comparisons; record provenance here. |
| Synthetic pulsar residuals | generated in `ivi_thickness/data.py` | This repository | MIT License | Placeholder dataset used when `--pulsar-csv` is omitted. Replace with real PTA catalogues; record provenance here. |

### User-supplied catalogues

- **Optical clocks:** Provide a CSV with columns `comparison,epoch_mjd,r,T1_K,T2_K[,w]`.
  Record the instrument, campaign, and license in this document when publishing.
- **Pulsar residuals:** Provide a CSV with `pulsar,toa_mjd,resid_us,resid_err_us[,distance_kpc,ra_deg,dec_deg]`.
  Note the PTA data release and license.

### Citation requirements

Analyses using this repository must cite the Planck Collaboration for sky maps,
the TDCOSMO/H0LiCOW consortium for time-delay and κ_ext measurements, and any
additional data releases listed here. Include acknowledgement text mandated by
each data provider.
