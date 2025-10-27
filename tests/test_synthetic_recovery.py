import numpy as np
import pandas as pd

from ivi_thickness.data import DataHub
from ivi_thickness.fit import fit_lensing_channel, fit_clock_channel
from ivi_thickness.model import Params, predicted_clock_residual_delta_nu_over_nu


def _synthetic_lens_dataframe(beta0=0.01, beta_G=0.07, beta_F=-0.05):
    # Construct five synthetic time-delay pairs with known coefficients
    GT = np.array([0.15, 0.65, 0.25, 0.85, 0.45])
    FK = np.array([0.3, 0.1, 0.7, 0.4, 0.9])
    n = GT.size
    dt_gr = np.full(n, 120.0)
    residual = beta0 + beta_G * GT + beta_F * FK
    dt_obs = dt_gr * (1.0 + residual)

    df = pd.DataFrame(
        {
            "lens_id": [f"L{i}" for i in range(n)],
            "pair_id": np.arange(n),
            "dt_obs": dt_obs,
            "sig_obs": np.full(n, 0.5),
            "dt_gr": dt_gr,
            "sig_gr": np.full(n, 0.5),
            "ra_deg": np.linspace(10.0, 50.0, n),
            "dec_deg": np.linspace(-20.0, 20.0, n),
            "G_proxy": GT,
            "kappa_ext": FK,
        }
    )
    return df


def test_lensing_coefficients_recover_truth():
    params = Params(epsilon_grain=0.1, epsilon_flat=-0.05, kappa0=1.0, E0_eV=1.0, p=1.0, q=1.0)
    df_lens = _synthetic_lens_dataframe()
    datahub = DataHub("./tmp_synth")

    fit = fit_lensing_channel(
        df_lens,
        params,
        datahub,
        use_cov=False,
        standardize=False,
    )

    expected = np.array([0.01, 0.07, -0.05])
    np.testing.assert_allclose(fit.beta, expected, atol=1e-10)
    assert fit.dof == len(df_lens) - len(expected)
    assert "sigma_obs" in fit.scale_info
    assert len(fit.scale_info["sigma_obs"]["values"]) == len(df_lens)


def test_clock_channel_recovers_epsilon_flat():
    params = Params(epsilon_grain=0.0, epsilon_flat=-4.2e-24, E0_eV=1.0, kappa0=1.0, p=1.0, q=1.0)
    T1 = np.array([300.0, 295.0, 310.0, 280.0])
    T2 = np.array([80.0, 85.0, 75.0, 90.0])
    r = predicted_clock_residual_delta_nu_over_nu(T1, T2, params)

    df_clock = pd.DataFrame(
        {
            "comparison": ["synthetic"] * len(T1),
            "t": np.arange(len(T1)),
            "T1": T1,
            "T2": T2,
            "r": r,
            "w": np.ones_like(r),
        }
    )

    fit = fit_clock_channel(df_clock, params)
    est = fit.scale_info["eps_flat_est"]
    np.testing.assert_allclose(est, params.epsilon_flat, rtol=1e-6)
    assert "sigma_obs" in fit.scale_info
