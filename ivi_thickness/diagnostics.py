from __future__ import annotations

import numpy as np
from typing import Dict, Any


def posterior_predictive_check(
    y: np.ndarray,
    yhat: np.ndarray,
    sigma: np.ndarray,
    rng: np.random.Generator,
    n_draws: int = 2048,
) -> Dict[str, Any]:
    """
    Perform a simple posterior predictive check using a χ² statistic.

    Parameters
    ----------
    y : array
        Observed data values.
    yhat : array
        Model predictions (same shape as y).
    sigma : array
        Observational uncertainties for each datum.
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    n_draws : int
        Number of replicated datasets to draw.

    Returns
    -------
    dict
        Summary containing observed statistic, replicated statistic moments,
        and posterior predictive p-values.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    sigma = np.clip(np.asarray(sigma, dtype=float), 1e-18, None)

    resid = y - yhat
    chi2_obs = np.sum((resid / sigma) ** 2)

    draws = rng.normal(loc=yhat, scale=sigma, size=(n_draws, y.size))
    chi2_rep = np.sum(((draws - yhat) / sigma) ** 2, axis=1)

    p_greater = float(np.mean(chi2_rep >= chi2_obs))
    p_less = float(np.mean(chi2_rep <= chi2_obs))
    two_tailed = float(2 * min(p_greater, p_less))

    return {
        "statistic": "chi2",
        "observed": float(chi2_obs),
        "p_greater": p_greater,
        "p_less": p_less,
        "two_tailed": two_tailed,
        "rep_mean": float(np.mean(chi2_rep)),
        "rep_std": float(np.std(chi2_rep, ddof=1)),
        "rep_q05": float(np.quantile(chi2_rep, 0.05)),
        "rep_q50": float(np.quantile(chi2_rep, 0.5)),
        "rep_q95": float(np.quantile(chi2_rep, 0.95)),
        "n_draws": int(n_draws),
        "rep_values": chi2_rep.tolist(),
    }
