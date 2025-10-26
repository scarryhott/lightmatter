import numpy as np

# Physical constants (cgs where needed)
C_LIGHT = 2.99792458e10  # cm/s

class Params:
    """
    Shared model parameters (tunable).
    Use signs consistent with theory:
      eps_grain > 0  → thickens time (adds lapse)
      eps_flat  < 0  → flattens time (reduces lapse)
    """
    def __init__(self,
                 epsilon_grain=1e-18,
                 epsilon_flat=-1e-19,
                 E0_eV=10.0,
                 kappa0=1e20,
                 p=1.0,
                 q=1.0):
        self.epsilon_grain = float(epsilon_grain)
        self.epsilon_flat  = float(epsilon_flat)
        self.E0_eV         = float(E0_eV)
        self.kappa0        = float(kappa0)
        self.p             = float(p)
        self.q             = float(q)

    def as_dict(self):
        return dict(
            epsilon_grain=self.epsilon_grain,
            epsilon_flat=self.epsilon_flat,
            E0_eV=self.E0_eV,
            kappa0=self.kappa0,
            p=self.p,
            q=self.q
        )


# -------------------------------
# IVI core shape functions & maps
# -------------------------------

def F_kappa(kappa, params: Params):
    """Grain density shape function F(κ). Dimensionless, monotone."""
    kappa = np.asarray(kappa, dtype=float)
    val = np.power(np.clip(kappa, 0, None) / params.kappa0, params.p)
    return np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

def G_temp(T_K, params: Params, k_B_eV=8.617333262e-5):
    """
    Temperature/radiation shape function G(T).
    Here we use a normalized energy scale: (k_B T / E0)^q.
    """
    T_K = np.asarray(T_K, dtype=float)
    ratio = (k_B_eV * np.clip(T_K, 0, None)) / params.E0_eV
    val = np.power(ratio, params.q)
    return np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

def sheet_delta(j, ell):
    """
    δ = 1 - j * ell^2  → zero on the i-flat sheet (curvature flattening of time).
    j := m * ell^2 / t^2
    """
    return 1.0 - (np.asarray(j) * (np.asarray(ell) ** 2))

def t_local_from_sheet(m, ell):
    """
    From j * ell^2 = 1 with j = m*ell^2 / t^2 → t = sqrt(m) * ell^2
    (algebraic local time law on the sheet).
    """
    m = np.asarray(m, dtype=float)
    ell = np.asarray(ell, dtype=float)
    return np.sqrt(np.clip(m, 0, None)) * (ell ** 2)

def i_of_t(m, t, C=0.0):
    """
    Logarithmic inverse map:
      i(t) = (2/3) m^(3/2) ln|t| + C
    """
    m = np.asarray(m, dtype=float)
    t = np.asarray(t, dtype=float)
    return (2.0/3.0) * (np.power(np.clip(m, 0, None), 1.5)) * np.log(np.abs(t)) + C

def lapse_W(Phi_over_c2, kappa, T_K, params: Params):
    """
    Return W = -g00 (lapse scalar) in weak field:
      g00 = -[ 1 + 2 Φ/c^2 - eps_grain F(κ) + eps_flat G(T) ]
      W   =  1 + 2 Φ/c^2 - eps_grain F(κ) + eps_flat G(T)
    """
    Fg = F_kappa(kappa, params)
    Gt = G_temp(T_K, params)
    return 1.0 + 2.0 * Phi_over_c2 - params.epsilon_grain * Fg + params.epsilon_flat * Gt

def predicted_clock_residual_delta_nu_over_nu(T1, T2, params: Params):
    """
    Clock channel (local): residual fractional frequency shift after standard corrections.
      r ≈ (1/2) * eps_flat * ( G(T1) - G(T2) )
    """
    G1 = G_temp(T1, params)
    G2 = G_temp(T2, params)
    return 0.5 * params.epsilon_flat * (G1 - G2)

def predicted_lens_residual(dt_obs, dt_gr, kappa_los, T_los, params: Params):
    """
    Lensing channel (cosmic): first-order fractional residual
      R ≈ a + b G(T) + c F(κ),    with a ≈ 0 at theory level.
    Here we return only the model piece (without intercept):
      R_model = 0.5 * eps_grain * F(κ) + 0.5 * eps_flat * G(T)
    You can fit a free intercept in the regression.
    """
    Fg = F_kappa(kappa_los, params)
    Gt = G_temp(T_los, params)
    return 0.5 * (params.epsilon_grain * Fg + params.epsilon_flat * Gt)

def predicted_pulsar_achromatic_delay(distance_kpc, kappa_los, params: Params, pc_to_cm=3.085677581e18):
    """
    Pulsar channel: achromatic (non-ISM) delay component scaling like (ε_grain / c) ∫ F(κ) dℓ
    Here we approximate a line-of-sight of L ≈ distance (conservative upper bound).
    Return microseconds.
    """
    L_cm = float(distance_kpc) * 1e3 * pc_to_cm
    Fg = float(F_kappa(kappa_los, params))
    delay_s = (params.epsilon_grain / C_LIGHT) * L_cm * Fg
    return delay_s * 1e6  # μs
