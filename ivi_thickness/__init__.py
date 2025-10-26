"""
IVI Time–Thickness package.

Implements:
- Sheet condition j * ell^2 = 1  (from j * ell^3 / ell = 1)
- Local time law t = sqrt(m) * ell^2
- Logarithmic inverse map i(t) = (2/3) m^(3/2) ln|t| + C
- Weak-field lapse deformation:
    g00 = -[ 1 + 2Φ/c^2 - eps_grain * F(kappa) + eps_flat * G(T) ]

Provides:
- Model utilities (ivi_thickness.model)
- Data loaders/proxies (ivi_thickness.data)
- Regression & optimization (ivi_thickness.fit)
- Diagnostic plotting (ivi_thickness.plots)
"""
__all__ = ["model", "data", "fit", "plots"]
__version__ = "0.1.0"
