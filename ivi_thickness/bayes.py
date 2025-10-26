"""
Bayesian joint inference for the IVI time–thickness model.

Parameters (theta)
------------------
theta = [
  log10_eps_grain,     # ε_grain > 0, modeled in log10
  log10_eps_flat,      # ε_flat  can be < 0; we sample its log10 of |ε_flat|,
                       #   and carry a fixed sign from prior (see priors below)
  p, q,                # exponents in F(κ) and G(T)
  log10_E0_eV,         # scale in G(T)
  a_lens,              # lensing intercept
  a_clock,             # clock intercept
  a_puls,              # pulsar intercept
  ln_s_lens,           # extra scatter (log std) lensing
  ln_s_clock,          # extra scatter (log std) clock
  ln_s_puls            # extra scatter (log std) pulsar
]

Model (likelihood)
------------------
Lensing:
  R_i ~ Normal( a_lens + 0.5*(eps_flat*G_i + eps_grain*F_i),  sqrt(sigR_i^2 + s_lens^2) )
  (Optional GLS: Σ_i plus s_lens^2 I within per-lens blocks)

Clocks:
  r_k ~ Normal( a_clock + 0.5*eps_flat*(G(T1_k) - G(T2_k)), sqrt(sig_k^2 + s_clock^2) )

Pulsars (achromatic proxy):
  y_j ~ Normal( a_puls + (eps_grain/c)*L_j*F_j (in μs units folded into slope),
                sqrt(sig_j^2 + s_puls^2) )
  Here we absorb L/c and unit conversions into design x_j created by data layer.

Priors (default)
----------------
- log10_eps_grain ~ Uniform[-20, -14]
- log10_eps_flat  ~ Uniform[-22, -16] and sign_flat ∈ {+1,-1} configured externally
- p ~ Uniform[0.3, 2.0]
- q ~ Uniform[0.3, 2.0]
- log10_E0_eV ~ Uniform[-1, 2]
- a_* ~ Normal(0, 0.1)
- ln_s_* ~ Uniform[log(1e-6), log(1e-1)]

You can change priors in `make_priors()`.

Usage
-----
See scripts/run_mcmc.py for a complete example.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from numpy.typing import ArrayLike

from .model import Params, F_kappa, G_temp, C_LIGHT

# ---------------------------
# Helpers: safe probabilities
# ---------------------------

def _norm_logpdf(resid: np.ndarray, sigma: np.ndarray) -> float:
    s2 = sigma * sigma
    return -0.5 * (np.sum(np.log(2.0*np.pi*s2)) + np.sum((resid*resid)/s2))

def _chol_solve_logquad(resid: np.ndarray, Sigma: np.ndarray) -> float:
    # for GLS: log-likelihood term -0.5*(log|Σ| + r^T Σ^{-1} r)
    from scipy.linalg import cho_factor, cho_solve
    eps = 1e-12
    c, lower = cho_factor(Sigma + eps*np.eye(Sigma.shape[0]), overwrite_a=False, check_finite=True)
    alpha = cho_solve((c, lower), resid)
    logdet = 2.0*np.sum(np.log(np.diag(c)))
    quad = resid @ alpha
    return -0.5 * (logdet + quad + resid.size*np.log(2.0*np.pi))

# ---------------------------
# Data container
# ---------------------------

@dataclass
class LensingBlock:
    R: np.ndarray             # residuals (dt_obs - dt_gr)/dt_gr
    sigR: np.ndarray          # propagated measurement sigma for R
    GT: np.ndarray            # G(T) predictor (already G_temp(...) evaluated)
    FK: np.ndarray            # F(kappa) predictor (already F_kappa(...) evaluated)
    lens_ids: np.ndarray      # per-row lens id (for block cov)
    use_cov: bool = False
    rho_intra: float = 0.3

    def Sigma(self, s_extra: float) -> Optional[np.ndarray]:
        if not self.use_cov:
            return None
        sig = np.sqrt(self.sigR**2 + s_extra**2)
        S = np.diag(sig**2)
        if self.rho_intra != 0.0:
            for L in np.unique(self.lens_ids):
                idx = np.where(self.lens_ids == L)[0]
                if len(idx) > 1:
                    blk = self.rho_intra * np.outer(sig[idx], sig[idx])
                    np.fill_diagonal(blk, 0.0)
                    S[np.ix_(idx, idx)] += blk
        return S

@dataclass
class ClockBlock:
    r: np.ndarray             # residual fractional frequency shift (after standard corr.)
    sig: np.ndarray           # measurement sigma per epoch
    G1: np.ndarray            # G(T1)
    G2: np.ndarray            # G(T2)

@dataclass
class PulsarBlock:
    y: np.ndarray             # RMS μs (proxy upper bound)
    sig: np.ndarray           # adopted sigma
    x: np.ndarray             # design factor proportional to distance*F_kappa (unitless scale)

@dataclass
class JointData:
    lens: LensingBlock
    clock: ClockBlock
    puls: PulsarBlock

# ---------------------------
# Priors
# ---------------------------

@dataclass
class Priors:
    # bounds: tuples (min, max) for uniforms
    log10_eps_grain: Tuple[float,float] = (-20.0, -14.0)
    log10_eps_flat:  Tuple[float,float] = (-22.0, -16.0)
    p: Tuple[float,float]              = (0.3, 2.0)
    q: Tuple[float,float]              = (0.3, 2.0)
    log10_E0_eV: Tuple[float,float]    = (-1.0, 2.0)
    # normal priors for intercepts a_*
    a_sigma: float = 0.1
    # uniform on log extra scatters
    ln_s_bounds: Tuple[float,float] = (np.log(1e-6), np.log(1e-1))
    # sign of eps_flat (theory expects <0); set -1 or +1
    sign_eps_flat: float = -1.0

def make_priors(**kw) -> Priors:
    P = Priors()
    for k,v in kw.items():
        setattr(P,k,v)
    return P

# ---------------------------
# Log prior
# ---------------------------

def log_prior(theta: np.ndarray, P: Priors) -> float:
    (lg_eg, lg_ef, p, q, lg_E0,
     a_l, a_c, a_p, ln_s_l, ln_s_c, ln_s_p) = theta

    # uniforms
    def U(x, lo, hi):
        return 0.0 if (lo <= x <= hi) else -np.inf

    lp = 0.0
    lp += U(lg_eg, *P.log10_eps_grain)
    lp += U(lg_ef, *P.log10_eps_flat)
    lp += U(p, *P.p)
    lp += U(q, *P.q)
    lp += U(lg_E0, *P.log10_E0_eV)
    lp += U(ln_s_l, *P.ln_s_bounds)
    lp += U(ln_s_c, *P.ln_s_bounds)
    lp += U(ln_s_p, *P.ln_s_bounds)

    if not np.isfinite(lp):
        return -np.inf

    # normals for intercepts
    a_sig2 = P.a_sigma**2
    lp += -0.5*(a_l*a_l + a_c*a_c + a_p*a_p)/a_sig2 - 3.0*np.log(np.sqrt(2.0*np.pi)*P.a_sigma)
    return lp

# ---------------------------
# Log-likelihood
# ---------------------------

def log_likelihood(theta: np.ndarray, D: JointData, P: Priors) -> float:
    (lg_eg, lg_ef, p, q, lg_E0,
     a_l, a_c, a_p, ln_s_l, ln_s_c, ln_s_p) = theta

    eps_grain = 10.0**lg_eg
    eps_flat  = P.sign_eps_flat * (10.0**lg_ef)
    E0_eV     = 10.0**lg_E0
    s_l = np.exp(ln_s_l)
    s_c = np.exp(ln_s_c)
    s_p = np.exp(ln_s_p)

    # lensing
    R    = D.lens.R
    GT   = D.lens.GT  # already G_temp
    FK   = D.lens.FK  # already F_kappa
    mu_R = a_l + 0.5*(eps_flat*GT + eps_grain*FK)

    if D.lens.use_cov:
        Sigma = D.lens.Sigma(s_l)
        ll_l = _chol_solve_logquad(R - mu_R, Sigma)
    else:
        sig = np.sqrt(D.lens.sigR**2 + s_l*s_l)
        ll_l = _norm_logpdf(R - mu_R, sig)

    # clocks
    r    = D.clock.r
    dG   = (D.clock.G1 - D.clock.G2) * 0.5
    mu_c = a_c + eps_flat * dG
    sigc = np.sqrt(D.clock.sig**2 + s_c*s_c)
    ll_c = _norm_logpdf(r - mu_c, sigc)

    # pulsars (achromatic proxy)
    y    = D.puls.y
    x    = D.puls.x   # proportional to distance*F_kappa
    # absorb unit factors into x outside; here linear in eps_grain
    mu_p = a_p + eps_grain * x
    sigp = np.sqrt(D.puls.sig**2 + s_p*s_p)
    ll_p = _norm_logpdf(y - mu_p, sigp)

    return ll_l + ll_c + ll_p

# ---------------------------
# Posterior
# ---------------------------

def log_posterior(theta: np.ndarray, D: JointData, P: Priors) -> float:
    lp = log_prior(theta, P)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, D, P)
    return lp + ll

# ---------------------------
# Build JointData from your DataHub frames
# ---------------------------

def build_joint_data(df_lens, df_clock, df_psr, params_for_maps: Params,
                     use_cov: bool = False, rho_intra: float = 0.3,
                     clock_sigma: Optional[np.ndarray] = None,
                     puls_frac_unc: float = 0.2):
    """Prepare numeric blocks used in the Bayesian likelihood."""
    # lensing inputs
    dt_obs = df_lens["dt_obs"].to_numpy(float)
    dt_gr  = df_lens["dt_gr"].to_numpy(float)
    sig_obs = df_lens["sig_obs"].to_numpy(float)
    sig_gr  = df_lens["sig_gr"].to_numpy(float)
    R = (dt_obs - dt_gr) / dt_gr
    sigR = np.hypot(sig_obs/dt_gr, (sig_gr*dt_obs)/(dt_gr*dt_gr))

    # predictors: prefer sky-varying G_proxy/kappa_ext if present
    from .model import G_temp, F_kappa
    if "G_proxy" in df_lens.columns and df_lens["G_proxy"].notna().any():
        GT = df_lens["G_proxy"].to_numpy(float)
    elif "rad_proxy" in df_lens.columns and df_lens["rad_proxy"].notna().any():
        GT = df_lens["rad_proxy"].to_numpy(float)
    else:
        GT = np.full_like(R, 0.5, dtype=float)

    if "kappa_ext" in df_lens.columns and df_lens["kappa_ext"].notna().any():
        kappa = df_lens["kappa_ext"].to_numpy(float)
    else:
        # fallback proxy by environment bucket
        from .data import DataHub
        dm = df_lens["dm_level"].to_list()
        mapping = {"low":1e19, "medium":5e19, "high":1e20}
        kappa = np.array([mapping.get(str(d).lower(), 5e19) for d in dm], dtype=float)

    FK = F_kappa(kappa, params_for_maps)

    lens_ids = df_lens["lens_id"].to_numpy(str)
    lens_block = LensingBlock(R=R, sigR=sigR, GT=GT, FK=FK, lens_ids=lens_ids,
                              use_cov=use_cov, rho_intra=rho_intra)

    # clock inputs
    T1 = df_clock["T1"].to_numpy(float)
    T2 = df_clock["T2"].to_numpy(float)
    r  = df_clock["r"].to_numpy(float)
    if clock_sigma is not None:
        sig = clock_sigma
    else:
        sig = df_clock["w"].to_numpy(float) if "w" in df_clock.columns else np.full_like(r, 1e-18)
    G1 = G_temp(T1, params_for_maps)
    G2 = G_temp(T2, params_for_maps)
    clock_block = ClockBlock(r=r, sig=sig, G1=G1, G2=G2)

    # pulsar inputs
    # x = distance_kpc * F_kappa(dm-level proxy) / C_LIGHT (converted to μs scale outside)
    dist = df_psr["distance_kpc"].to_numpy(float)
    if "kappa_ext" in df_psr.columns and df_psr["kappa_ext"].notna().any():
        kap_psr = df_psr["kappa_ext"].to_numpy(float)
    else:
        dm_level = df_psr["dm_level"].to_list()
        mapping = {"low":1e19, "medium":5e19, "high":1e20}
        kap_psr = np.array([mapping.get(str(d).lower(), 5e19) for d in dm_level], dtype=float)
    F_psr = F_kappa(kap_psr, params_for_maps)
    pc_to_cm = 3.085677581e18
    L_cm = dist * 1e3 * pc_to_cm
    # achromatic delay ~ (eps_grain / c) * L * F → in μs:
    x_phys = (L_cm / C_LIGHT) * F_psr * 1e6  # μs per unit eps_grain
    y = df_psr["rms_residual_us"].to_numpy(float)
    sig_p = puls_frac_unc * y
    puls_block = PulsarBlock(y=y, sig=sig_p, x=x_phys)

    return JointData(lens=lens_block, clock=clock_block, puls=puls_block)

# ---------------------------
# Emcee driver
# ---------------------------

def run_emcee(logpost, theta0: np.ndarray, nwalkers: int = 24, nsteps: int = 4000, burn: int = 1000,
              progress: bool = True, random_state: Optional[int] = 42):
    import emcee
    rng = np.random.default_rng(random_state)
    ndim = theta0.size
    # initialize walkers with small Gaussian ball
    p0 = theta0 + 1e-3 * rng.normal(size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost)
    sampler.run_mcmc(p0, nsteps, progress=progress)
    chain = sampler.get_chain(discard=burn, flat=True)
    lnp   = sampler.get_log_prob(discard=burn, flat=True)
    return {"chain": chain, "log_prob": lnp, "sampler": sampler}
