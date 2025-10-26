# Methods Appendix — IVI Time–Thickness Analysis

## Model Overview

We test the IVI prediction that local "graininess" (κ) thickens time and radiation energy density (G(T)) flattens time, via a weak-field lapse deformation:

$$
g_{00} = -\left[ 1 + \frac{2\Phi}{c^2} - \varepsilon_{\rm grain}\,F(\kappa) + \varepsilon_{\rm flat}\,G(T)\right]
$$

with dimensionless shape functions $F(\kappa)=(\kappa/\kappa_0)^p$ and $G(T)=[k_B T/E_0]^q$.
On the IVI "sheet" the algebraic relation $j\,\ell^2=1$ yields the local time law $t=\sqrt{m}\,\ell^2$ and the logarithmic inverse $i(t)=\frac{2}{3}m^{3/2}\ln|t|+C$; these encode the curvature-flattening direction.

## Channels and Observables

### Gravitational Lensing (Cosmic)
For each image-pair time delay, we form a fractional residual $R=(\Delta t_{\rm obs}-\Delta t_{\rm GR})/\Delta t_{\rm GR}$. To first order, $R \approx a + b\,G(T_{\rm LOS}) + c\,F(\kappa_{\rm LOS})$. We regress $R$ on $[1,\,G,\,F]$.

### Optical Clocks (Local)
Fractional frequency residuals $r_k$ (post standard corrections) satisfy $r_k \approx a + \tfrac{1}{2}\varepsilon_{\rm flat}\,[G(T_{1,k})-G(T_{2,k})]$. We regress $r$ on $[1,\,\Delta G]$ to bound $\varepsilon_{\rm flat}$.

### Pulsar Timing (Galactic)
As a conservative achromatic summary, per-pulsar RMS residuals satisfy $\mathrm{RMS} \approx a + \alpha\,(\mathrm{distance}\times F(\kappa))$. We regress RMS on $[1,\,\mathrm{distance}\,F(\kappa)]$ to constrain grain thickening.

## Data and Calibrations

### Time Delays (TDCOSMO/H0LiCOW)
We ingest a CSV with one row per image pair:
```
lens_id, pair_id, dt_obs_days, sig_obs_days, dt_gr_days, sig_gr_days, z_lens, z_src, ra_deg, dec_deg
```

### External Convergence (κ_ext)
We merge a per-lens κ table:
```
lens_id, kappa_ext[, sigma_kappa]
```
A global scale factor can be applied for unit normalization.

### Radiation Field (Planck 857 GHz)
We sample a HEALPix map (MJy/sr) along each line-of-sight to obtain intensity $I$. A dimensionless proxy is constructed as $G = (I/I_0)^\gamma$, with either fixed $I_0$ or the median across lenses (auto-$I_0$).

### Clocks
We ingest time-series residuals in:
```
comparison, epoch_mjd, r, T1_K, T2_K[, w]
```
We use weights $w$ if provided.

### Pulsars
We ingest achromatic residuals per TOA and aggregate to per-PSR RMS:
```
pulsar, toa_mjd, resid_us, resid_err_us[, distance_kpc, ra_deg, dec_deg]
```

## Inference

We use weighted least squares (WLS) with propagated uncertainties per channel. Predictors are standardized (zero mean, unit variance) before regression; reported coefficients include 95% confidence intervals and the scaling info to recover physical units. We provide leave-one-lens-out jackknife and sky-coordinate permutation tests for robustness. Full Bayesian MCMC is available as an optional step, initialized by WLS.

## Outputs

The runner writes:
- `ivi_publish_results.json`: parameters, coefficients, CIs, χ², provenance.
- `*_fit_points.csv`: per-channel scatter data with model predictions.
- PNG figures: per-channel fits and residual histograms.

## Reproducibility

All code paths and calibrations (κ scaling, HEALPix ordering, $I_0$, $\gamma$) are recorded in the JSON. The figures are regenerated from the saved CSVs, decoupled from the internal design matrix.
