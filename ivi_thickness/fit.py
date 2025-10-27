import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
from scipy import stats
from scipy.linalg import LinAlgError, cho_factor, cho_solve

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# -----------------
# Core Functions
# -----------------

def weighted_ols(X, y, w):
    """Weighted least squares."""
    W = np.diag(w)
    XT_W = X.T @ W
    XT_W_X = XT_W @ X
    XT_W_y = XT_W @ y
    beta = np.linalg.solve(XT_W_X, XT_W_y)
    yhat = X @ beta
    resid = y - yhat
    dof = max(1, len(y) - X.shape[1])
    sigma2 = (resid.T @ W @ resid) / dof
    cov = np.linalg.inv(XT_W_X) * sigma2
    se = np.sqrt(np.diag(cov))
    chi2 = (resid**2 * w).sum()
    return beta, se, resid, chi2, dof

def gls(X, y, Sigma):
    """
    Generalized least squares: minimize (y - Xb)^T Σ^{-1} (y - Xb).
    Uses Cholesky whitening.
    """
    # Σ must be SPD; add tiny jitter if needed
    eps = 1e-12
    Sigma = Sigma + eps * np.eye(Sigma.shape[0])
    c, lower = cho_factor(Sigma, overwrite_a=False, check_finite=True)
    y_w = cho_solve((c, lower), y)
    X_w = cho_solve((c, lower), X)
    XT_X = X.T @ X_w
    XT_y = X.T @ y_w
    beta = np.linalg.solve(XT_X, XT_y)
    yhat = X @ beta
    resid = y - yhat
    # GLS covariance:
    cov = np.linalg.inv(XT_X)
    se = np.sqrt(np.diag(cov))
    dof = max(1, len(y) - X.shape[1])
    chi2 = resid.T @ cho_solve((c, lower), resid)
    return beta, se, resid, chi2, dof

def _standardize(x):
    """Standardize data to zero mean and unit variance."""
    x = np.asarray(x, float)
    mu = np.nanmean(x)
    sd = np.nanstd(x) if np.nanstd(x) > 0 else 1.0
    return (x - mu) / sd, mu, sd
from .model import Params, F_kappa, G_temp, predicted_clock_residual_delta_nu_over_nu, \
                   predicted_lens_residual, predicted_pulsar_achromatic_delay

# -----------------
# Data Classes
# -----------------

@dataclass
class FitResult:
    """Container for fit results."""
    beta: np.ndarray
    se: np.ndarray
    chi2: float
    dof: int
    resid: np.ndarray
    design_names: List[str]
    scale_info: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    @property
    def chi2_red(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else np.nan

    def __str__(self) -> str:
        lines = ["Fit results:", "-" * 40]
        for name, b, se in zip(self.design_names, self.beta, self.se):
            lines.append(f"{name:>15s}: {b:+.3f} ± {se:.3f}")
        lines.append(f"\nχ²/dof = {self.chi2:.2f}/{self.dof} = {self.chi2_red:.2f}")
        return "\n".join(lines)

@dataclass
class CombinedResult:
    """Container for combined fit results across channels."""
    lensing: Optional[FitResult] = None
    clocks: Optional[FitResult] = None
    pulsars: Optional[FitResult] = None
    p_value: Optional[float] = None

    def __str__(self) -> str:
        lines = ["IVI Time-Thickness Analysis Results", "=" * 50]
        
        if self.lensing:
            lines.extend(["", "Lensing Channel:", str(self.lensing)])
        if self.clocks:
            lines.extend(["", "Clocks Channel:", str(self.clocks)])
        if self.pulsars:
            lines.extend(["", "Pulsars Channel:", str(self.pulsars)])
        
        if self.p_value is not None:
            lines.append(f"\nGlobal p-value = {self.p_value:.3f}")
        
        return "\n".join(lines)

# -----------------------
# Channel-specific fits
# -----------------------

def fit_lensing_channel(
    df_pairs: pd.DataFrame, 
    params: Params, 
    datahub,
    use_cov: bool = True,
    rho_intra_lens: float = 0.5,
    standardize: bool = True
) -> FitResult:
    """
    Fit lensing time delay residuals with IVI model.
    
    Args:
        df_pairs: DataFrame with columns:
            - dt_obs, sig_obs: Observed time delay and uncertainty
            - dt_gr, sig_gr: GR-predicted delay and uncertainty
            - ra_deg, dec_deg: Sky coordinates
            - lens_id: Unique identifier for each lens system
            - kappa_ext: Optional external convergence (uses dm_level as fallback)
            - dm_level: Dust map level (low/medium/high)
            - rad_proxy: Optional radiation field proxy (from HEALPix map)
        params: IVI model parameters
        datahub: Data provider instance
        use_cov: If True, use GLS with intra-lens covariance
        rho_intra_lens: Correlation coefficient for time-delay pairs from same lens
        standardize: If True, standardize predictors to zero mean and unit variance
        
    Returns:
        FitResult with coefficients and scaling info
    """
    """
    Fit lensing time delay residuals with IVI model.
    
    Args:
        df_pairs: DataFrame with columns:
            - dt_obs, sig_obs: Observed time delay and uncertainty
            - dt_gr, sig_gr: GR-predicted delay and uncertainty
            - ra_deg, dec_deg: Sky coordinates
            - lens_id: Unique identifier for each lens system
            - kappa_ext: Optional external convergence (uses dm_level as fallback)
            - dm_level: Dust map level (low/medium/high)
        params: IVI model parameters
        datahub: Data provider instance
        use_cov: If True, use GLS with intra-lens covariance
        rho_intra_lens: Correlation coefficient for time-delay pairs from same lens
        standardize: If True, standardize predictors to zero mean and unit variance
        
    Returns:
        FitResult with coefficients and scaling info
    """
    # Ensure required columns are present
    required_cols = {"dt_obs", "dt_gr", "sig_obs", "sig_gr", "ra_deg", "dec_deg", "lens_id"}
    missing = required_cols - set(df_pairs.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Residual and uncertainty
    dt_obs = df_pairs["dt_obs"].to_numpy(float)
    dt_gr = df_pairs["dt_gr"].to_numpy(float)
    sig_obs = df_pairs["sig_obs"].to_numpy(float)
    sig_gr = df_pairs["sig_gr"].to_numpy(float)
    lens_ids = df_pairs["lens_id"].values

    # Calculate fractional residual R = (dt_obs - dt_gr)/dt_gr
    R = (dt_obs - dt_gr) / dt_gr
    sigR = np.hypot(sig_obs/dt_gr, (sig_gr*dt_obs)/(dt_gr**2))
    
    # Prefer sky-varying G_proxy if present; fall back to legacy or proxy.
    if "G_proxy" in df_pairs.columns and df_pairs["G_proxy"].notna().any():
        G_vec = df_pairs["G_proxy"].to_numpy(float)
        G_source = "sky_map"
    elif "rad_proxy" in df_pairs.columns and df_pairs["rad_proxy"].notna().any():
        G_vec = df_pairs["rad_proxy"].to_numpy(float)
        G_source = "legacy_map"
    else:
        G_vec = np.array(
            [datahub.los_radiation_proxy(ra, dec) for ra, dec in zip(df_pairs["ra_deg"], df_pairs["dec_deg"])],
            dtype=float
        )
        G_source = "declination_proxy"
    
    # Handle kappa: use provided kappa_ext or fall back to proxy
    if "kappa_ext" in df_pairs.columns and df_pairs["kappa_ext"].notna().any():
        kappa_los = df_pairs["kappa_ext"].fillna(np.nan).to_numpy(float)
        mask_na = np.isnan(kappa_los)
        if "dm_level" in df_pairs.columns and mask_na.any():
            kappa_los[mask_na] = np.array([datahub.env_kappa_proxy(dm)
                                         for dm in df_pairs.loc[mask_na, "dm_level"]],
                                         dtype=float)
        kappa_source = "physical"
    elif "dm_level" in df_pairs.columns:
        kappa_los = np.array([datahub.env_kappa_proxy(dm)
                            for dm in df_pairs["dm_level"]],
                            dtype=float)
        kappa_source = "proxy"
    else:
        raise ValueError("Either 'kappa_ext' or 'dm_level' must be provided")

    # Calculate shape functions
    if G_source in ("sky_map", "legacy_map"):
        # Map-based G already normalized to (I/I0)^gamma
        GT = G_vec  # Use directly without G_temp transformation
    else:
        GT = G_temp(G_vec, params)  # Use standard G_temp for proxy-based G
        
    FK = F_kappa(kappa_los, params)
    
    # Standardize predictors if requested
    sigma_summary = {
        "values": sigR.tolist(),
        "median": float(np.median(sigR)),
        "mean": float(np.mean(sigR)),
        "min": float(np.min(sigR)),
        "max": float(np.max(sigR))
    }

    if standardize:
        GT_std, mu_GT, sd_GT = _standardize(GT)
        FK_std, mu_FK, sd_FK = _standardize(FK)
        scale_info = {
            'G_temp': {
                'mean': mu_GT,
                'std': sd_GT,
                'source': G_source,
                'n_obs': len(GT)
            },
            'F_kappa': {
                'mean': mu_FK,
                'std': sd_FK,
                'source': kappa_source,
                'n_obs': len(FK)
            }
        }
        scale_info["sigma_obs"] = sigma_summary
        X = np.column_stack([np.ones_like(R), GT_std, FK_std])
        names = ["intercept", "G_temp", "F_kappa"]
    else:
        X = np.column_stack([np.ones_like(R), GT, FK])
        names = ["intercept", "G_temp", "F_kappa"]
        scale_info = {}
        mu_GT = float(np.mean(GT))
        sd_GT = float(np.std(GT))
        mu_FK = float(np.mean(FK))
        sd_FK = float(np.std(FK))
        scale_info["sigma_obs"] = sigma_summary
        scale_info["G_temp"] = {
            "mean": mu_GT,
            "std": sd_GT,
            "source": G_source,
            "n_obs": len(GT)
        }
        scale_info["F_kappa"] = {
            "mean": mu_FK,
            "std": sd_FK,
            "source": kappa_source,
            "n_obs": len(FK)
        }
    
    # Check for constant columns in the design matrix (excluding intercept)
    X_no_intercept = X[:, 1:] if X.shape[1] > 1 else X
    col_std = np.std(X_no_intercept, axis=0)
    constant_cols = np.where(col_std < 1e-10)[0]
    
    if len(constant_cols) > 0:
        warnings.warn(f"Found {len(constant_cols)} constant columns in design matrix. "
                    f"Dropping columns: {constant_cols}")
        # Keep intercept and non-constant columns
        keep_cols = [0] + [i+1 for i in range(X.shape[1]-1) if i not in constant_cols]
        X = X[:, keep_cols]
        names = [names[i] for i in keep_cols]
    
    # Add small ridge to diagonal of X'X for stability
    def ridge_regression(X, y, w, alpha=1e-6):
        X_w = X * np.sqrt(w)[:, None]
        y_w = y * np.sqrt(w)
        XTX = X_w.T @ X_w
        XTy = X_w.T @ y_w
        
        # Add ridge penalty
        ridge = alpha * np.eye(XTX.shape[0])
        # Don't regularize the intercept
        if XTX.shape[0] > 0:
            ridge[0, 0] = 0
            
        beta = np.linalg.solve(XTX + ridge, XTy)
        
        # Calculate residuals and chi2
        y_pred = X @ beta
        resid = y - y_pred
        chi2 = np.sum(w * resid**2)
        dof = len(y) - X.shape[1]
        
        # Calculate standard errors
        XTX_inv = np.linalg.inv(XTX + ridge)
        se = np.sqrt(np.diag(XTX_inv))
        
        return beta, se, resid, chi2, dof
    
    # Fit model with ridge regression as fallback
    try:
        if use_cov and len(np.unique(lens_ids)) < len(lens_ids):
            # Build intra-lens covariance matrix
            n = len(R)
            Sigma = np.diag(sigR**2)
            
            # Add off-diagonal terms for pairs from same lens
            for lens_id in np.unique(lens_ids):
                idx = np.where(lens_ids == lens_id)[0]
                if len(idx) > 1:  # Only if multiple images per lens
                    for i in idx:
                        for j in idx:
                            if i != j:
                                Sigma[i,j] = rho_intra_lens * sigR[i] * sigR[j]
            
            # Try GLS first
            try:
                beta, se, resid, chi2, dof = gls(X, R, Sigma)
            except (LinAlgError, ValueError) as e:
                warnings.warn(f"GLS failed with error: {str(e)}. Falling back to WLS.")
                w = 1.0 / np.clip(sigR**2, 1e-24, None)
                beta, se, resid, chi2, dof = ridge_regression(X, R, w)
        else:
            # Standard weighted least squares with ridge
            w = 1.0 / np.clip(sigR**2, 1e-24, None)
            beta, se, resid, chi2, dof = ridge_regression(X, R, w)
            
    except np.linalg.LinAlgError as e:
        # If still failing, try with more regularization
        warnings.warn(f"Standard regression failed: {str(e)}. Using stronger ridge regularization.")
        w = 1.0 / np.clip(sigR**2, 1e-24, None)
        beta, se, resid, chi2, dof = ridge_regression(X, R, w, alpha=1e-3)
    
    # Store scaling information for interpretation
    scale_info.update({
        "mu_GT": mu_GT,
        "sd_GT": sd_GT,
        "mu_FK": mu_FK,
        "sd_FK": sd_FK,
        "y_mean": float(np.mean(R)),
        "y_std": float(np.std(R)),
        "n_obs": len(R)
    })
    
    return FitResult(
        beta=beta,
        se=se,
        chi2=chi2,
        dof=dof,
        resid=resid,
        design_names=names,
        scale_info=scale_info
    )

def fit_clock_channel(df_clock: pd.DataFrame, params: Params) -> FitResult:
    """
    Fit clock comparison data to constrain ε_flat.
    
    Model: r_k ≈ a + ε_flat * 0.5*(G(T1_k) - G(T2_k))
    
    Args:
        df_clock: DataFrame with columns:
            - T1, T2: Temperatures of the two clocks
            - r: Observed fractional frequency residual
            - w: Optional weights (default: 1.0)
        params: IVI model parameters
        
    Returns:
        FitResult with ε_flat estimate and physical scaling
    """
    T1 = df_clock["T1"].to_numpy(float)
    T2 = df_clock["T2"].to_numpy(float)
    r = df_clock["r"].to_numpy(float)
    w = df_clock["w"].to_numpy(float) if "w" in df_clock.columns else np.ones_like(r)
    
    # Calculate ΔG and standardize
    G1 = G_temp(T1, params)
    G2 = G_temp(T2, params)
    dG = 0.5 * (G1 - G2)
    dG_std, mu_dG, sd_dG = _standardize(dG)
    
    # Build design matrix
    X = np.column_stack([np.ones_like(dG_std), dG_std])
    names = ["a (intercept)", "eps_flat (slope)"]
    
    # Fit model
    beta, se, resid, chi2, dof = weighted_ols(X, r, w)
    
    # Calculate physical ε_flat (convert from standardized coefficient)
    slope_scale = sd_dG if sd_dG != 0 else 1.0
    eps_flat_est = beta[1] / slope_scale  # Convert back to original scale
    eps_flat_se = se[1] / slope_scale
    eps_flat_upper = eps_flat_est + 1.96 * eps_flat_se  # 95% upper limit

    sigma_obs = np.clip(1.0 / np.sqrt(np.clip(w, 1e-24, None)), 1e-18, None)
    sigma_summary = {
        "values": sigma_obs.tolist(),
        "median": float(np.median(sigma_obs)),
        "mean": float(np.mean(sigma_obs)),
        "min": float(np.min(sigma_obs)),
        "max": float(np.max(sigma_obs))
    }
    
    # Store scaling information
    scale_info = {
        "dG_mean": mu_dG,
        "dG_std": sd_dG,
        "eps_flat_est": eps_flat_est,
        "eps_flat_se": eps_flat_se,
        "eps_flat_upper_95": eps_flat_upper,
        "n_obs": len(r),
        "sigma_obs": sigma_summary
    }
    
    return FitResult(beta, se, chi2, dof, resid, names, scale_info)

def fit_pulsar_channel(df_psr: pd.DataFrame, params: Params, datahub):
    """
    Compare predicted achromatic delays with observed RMS as a conservative bound.
    Fit a simple regression:
      r_psr ≈ a + alpha * F(κ) * (L/c) * eps_grain   (we let alpha be absorbed in slope)
    Use proxy κ from dm_level unless provided.
    """
    dm_level = df_psr["dm_level"].to_list()
    kappa = np.array([datahub.env_kappa_proxy(dm) for dm in dm_level], dtype=float)
    dist = df_psr["distance_kpc"].to_numpy(float)
    rms  = df_psr["rms_residual_us"].to_numpy(float)

    FK = F_kappa(kappa, params)
    # predictor proportional to L/c * F(κ); units folded into slope
    x = dist * FK
    y = rms
    w = 1.0 / np.clip((0.2 * y)**2, 1e-12, None)  # 20% fractional uncertainty as placeholder

    X = np.column_stack([np.ones_like(x), x])
    names = ["a (intercept)", "scale * eps_grain"]

    beta, se, resid, chi2, dof = weighted_ols(X, y, w)
    sigma_obs = np.clip(1.0 / np.sqrt(np.clip(w, 1e-24, None)), 1e-18, None)
    scale_info = {
        "sigma_obs": {
            "values": sigma_obs.tolist(),
            "median": float(np.median(sigma_obs)),
            "mean": float(np.mean(sigma_obs)),
            "min": float(np.min(sigma_obs)),
            "max": float(np.max(sigma_obs))
        },
        "predictor": {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "n_obs": len(x)
        }
    }
    return FitResult(beta, se, chi2, dof, resid, names, scale_info)

# -----------------------
# Combined assessment
# -----------------------

@dataclass
class CombinedResult:
    """Container for combined results from all data channels."""
    lens: FitResult
    clock: FitResult
    pulsar: FitResult
    chi2_total: float
    dof_total: int
    p_value: float = field(default=np.nan)
    
    def __str__(self) -> str:
        """Human-readable summary of combined results."""
        lines = [
            "=== COMBINED RESULTS ===",
            f"Total χ² = {self.chi2_total:.3f}, dof = {self.dof_total}, χ²_red = {self.chi2_total/max(1, self.dof_total):.3f}",
            f"Combined p-value = {self.p_value:.4f}",
            "\n=== LENSING ===",
            str(self.lens),
            "\n=== CLOCKS ===",
            str(self.clock),
            "\n=== PULSARS ===",
            str(self.pulsar)
        ]
        return "\n".join(lines)

def jackknife_lensing(df_pairs: pd.DataFrame, params: Params, datahub, 
                      min_lens_obs: int = 3) -> List[Dict[str, Any]]:
    """
    Perform leave-one-lens-out jackknife analysis.
    
    Args:
        df_pairs: DataFrame with lensing data
        params: IVI model parameters
        datahub: Data provider instance
        min_lens_obs: Minimum number of observations per lens to include
        
    Returns:
        List of dicts with jackknife results for each lens
    """
    from scipy.linalg import LinAlgError
    
    results = []
    lens_ids = df_pairs['lens_id'].unique()
    n_skipped = 0
    
    for lens_id in lens_ids:
        # Skip if we'd have too few observations
        mask = df_pairs['lens_id'] != lens_id
        if mask.sum() < min_lens_obs:
            n_skipped += 1
            continue
            
        try:
            # Fit model without this lens
            fit = fit_lensing_channel(df_pairs[mask], params, datahub)
            results.append({
                'lens_id': lens_id,
                'beta': fit.beta,
                'se': fit.se,
                'chi2': fit.chi2,
                'dof': fit.dof,
                'success': True
            })
        except (LinAlgError, ValueError) as e:
            # Handle singular matrix or other numerical issues
            n_skipped += 1
            results.append({
                'lens_id': lens_id,
                'beta': None,
                'se': None,
                'chi2': None,
                'dof': None,
                'success': False,
                'error': str(e)
            })
    
    # Filter out failed fits
    valid_results = [r for r in results if r.get('success', False)]
    
    if n_skipped > 0:
        print(f"  Warning: Skipped {n_skipped} lenses due to insufficient data or numerical issues")
    
    if len(valid_results) < 2:
        raise RuntimeError(f"Insufficient valid jackknife samples (got {len(valid_results)}, need at least 2)")
        
    return valid_results

def permutation_test(df_pairs: pd.DataFrame, params: Params, datahub,
                    n_perm: int = 200, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """
    Permutation test to assess significance of lensing signals.
    
    Args:
        df_pairs: DataFrame with lensing data
        params: IVI model parameters
        datahub: Data provider instance
        n_perm: Number of permutations
        
    Returns:
        Dictionary with test results
    """
    # Get real fit
    real_fit = fit_lensing_channel(df_pairs, params, datahub)
    real_beta = real_fit.beta[1:]  # Exclude intercept
    
    # Initialize storage
    null_dist = np.zeros((n_perm, len(real_beta)))
    df_perm = df_pairs.copy()
    
    rng = np.random.default_rng() if rng is None else rng

    # Run permutations
    for i in range(n_perm):
        # Shuffle sky positions while keeping other columns fixed
        idx = rng.permutation(len(df_perm))
        df_perm["ra_deg"] = df_perm["ra_deg"].iloc[idx].values
        df_perm["dec_deg"] = df_perm["dec_deg"].iloc[idx].values
        
        # Fit permuted data
        perm_fit = fit_lensing_channel(df_perm, params, datahub)
        null_dist[i] = perm_fit.beta[1:]  # b,c coefficients only
    
    # Calculate two-tailed p-values
    p_values = 2 * np.minimum(
        np.mean(null_dist >= real_beta, axis=0),
        np.mean(null_dist <= real_beta, axis=0)
    )
    
    return {
        "real_beta": real_beta,
        "null_dist": null_dist,
        "p_values": p_values,
        "perm_means": np.mean(null_dist, axis=0),
        "perm_stds": np.std(null_dist, axis=0, ddof=1),
        "n_perm": n_perm
    }

def combined_assessment(lens_fit: FitResult, clock_fit: FitResult, 
                       pulsar_fit: FitResult) -> 'CombinedResult':
    """Combine results from all channels for joint assessment."""
    chi2_total = lens_fit.chi2 + clock_fit.chi2 + pulsar_fit.chi2
    dof_total = lens_fit.dof + clock_fit.dof + pulsar_fit.dof
    
    # Calculate combined p-value
    from scipy.stats import chi2 as chi2_dist
    p_value = 1.0 - chi2_dist.cdf(chi2_total, dof_total) if dof_total > 0 else np.nan
    
    return CombinedResult(
        lens=lens_fit,
        clock=clock_fit,
        pulsar=pulsar_fit,
        chi2_total=chi2_total,
        dof_total=dof_total,
        p_value=p_value
    )
