from __future__ import annotations
import os
import numpy as np
from typing import Tuple

class HealpixNotAvailable(RuntimeError):
    pass

def _import_healpy():
    try:
        import healpy as hp  # type: ignore
        return hp
    except Exception as e:
        raise HealpixNotAvailable(
            "healpy is required for HEALPix map sampling. "
            "Install with: pip install healpy"
        ) from e

class HealpixSampler:
    """
    Minimal HEALPix sampler for Planck/IRIS intensity maps (MJy/sr).
    Handles RING or NEST ordering and vectorized RA/Dec sampling.

    Parameters
    ----------
    path : str
        FITS filename for HEALPix map (e.g., Planck 857 GHz).
    nside : int
        NSIDE of the map (e.g., 2048).
    nest : bool
        True if map is NESTED ordered, else False (RING).
    field : int
        FITS field index (0 for default scalar map).
    """
    def __init__(self, path: str, nside: int, nest: bool, field: int = 0):
        hp = _import_healpy()
        if not os.path.exists(path):
            raise FileNotFoundError(f"HEALPix map not found: {path}")
        self.hp = hp
        self.path = path
        self.nside = int(nside)
        self.nest = bool(nest)
        self.field = int(field)
        self.map = hp.read_map(path, field=field, dtype=float, verbose=False)
        # Sanity check NSIDE
        m_nside = hp.get_nside(self.map)
        if m_nside != self.nside:
            raise ValueError(f"Map NSIDE={m_nside} does not match requested nside={self.nside}")
        # Units: assume MJy/sr for Planck 857 GHz (HFI), which is what we want.

    def sample_mjysr(self, ra_deg, dec_deg) -> np.ndarray:
        """
        Sample map intensity at given coordinates (deg). Vectorized.
        Returns MJy/sr as float array (same shape as broadcasted inputs).
        """
        ra = np.asarray(ra_deg, float)
        dec = np.asarray(dec_deg, float)
        # Convert to theta, phi in radians (healpy convention):
        # theta = colatitude = 90° - dec, phi = ra
        theta = np.deg2rad(90.0 - dec)
        phi   = np.deg2rad(ra)
        pix = self.hp.ang2pix(self.nside, theta, phi, nest=self.nest)
        vals = self.map[pix]
        return np.asarray(vals, float)


def load_healpix_map(path: str, field: int = 0, nest: bool = False) -> Tuple[np.ndarray, int, bool]:
    """
    Load a HEALPix map from FITS file.

    Returns
    -------
    map_data : np.ndarray
        The map values for the requested field.
    nside : int
        NSIDE of the map.
    is_nest : bool
        True if the map is NEST ordered.
    """
    hp = _import_healpy()
    if not os.path.exists(path):
        raise FileNotFoundError(f"HEALPix map not found: {path}")
    m = hp.read_map(path, field=field, dtype=float, nest=nest, verbose=False)
    nside = hp.get_nside(m)
    return np.asarray(m, dtype=float), int(nside), bool(nest)


def smooth_map(m: np.ndarray, smooth_fwhm_arcmin: float = None) -> np.ndarray:
    """
    Optionally smooth a HEALPix map with a Gaussian kernel.
    """
    if smooth_fwhm_arcmin in (None, 0):
        return np.asarray(m, dtype=float)
    hp = _import_healpy()
    fwhm_rad = np.deg2rad(float(smooth_fwhm_arcmin) / 60.0)
    smoothed = hp.smoothing(np.asarray(m, dtype=float), fwhm=fwhm_rad, verbose=False)
    return np.asarray(smoothed, dtype=float)


def sample_kappa_at_radec(
    m: np.ndarray,
    nside: int,
    ra_deg,
    dec_deg,
    nest: bool = False,
    fill_value: float = np.nan
) -> np.ndarray:
    """
    Sample a HEALPix map at the given RA/Dec coordinates.
    """
    hp = _import_healpy()
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(np.mod(ra, 360.0))
    pix = hp.ang2pix(int(nside), theta, phi, nest=bool(nest))
    vals = np.asarray(m, dtype=float)[pix]
    unseen = ~np.isfinite(vals) | (vals == hp.UNSEEN)
    if np.any(unseen):
        vals = vals.copy()
        vals[unseen] = fill_value
    return vals


def apply_mask(
    vals: np.ndarray,
    mask_map: np.ndarray,
    nside: int,
    ra_deg,
    dec_deg,
    nest: bool = False,
    fill_value: float = np.nan,
    mask_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a HEALPix mask to sampled values.

    Returns the masked values (with fill_value applied where mask < threshold)
    and a boolean array indicating which samples are kept.
    """
    mask_vals = sample_kappa_at_radec(
        mask_map, nside, ra_deg, dec_deg, nest=nest, fill_value=0.0
    )
    good = np.isfinite(mask_vals) & (mask_vals > mask_threshold)
    out = np.asarray(vals, dtype=float)
    if np.any(~good):
        out = out.copy()
        out[~good] = fill_value
    return out, good
def radiation_G_from_map(I_mjysr: np.ndarray, I0_mjysr: float, gamma: float = 1.0) -> np.ndarray:
    """
    Build a dimensionless radiation proxy G(T) from intensity:
        G = (I / I0)^gamma
    where I0 is a chosen normalization (e.g., median LOS intensity across lenses).
    """
    I = np.asarray(I_mjysr, float)
    I0 = float(I0_mjysr) if I0_mjysr else 1.0
    # Protect against zero or negative normals:
    I0 = max(I0, 1e-12)
    ratio = np.clip(I / I0, 0.0, np.inf)
    G = np.power(ratio, float(gamma))
    # Replace NaNs/Infs with zeros to be robust:
    G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    return G

def median_I0_for_lenses(sampler: HealpixSampler, ra_deg: np.ndarray, dec_deg: np.ndarray) -> float:
    """
    Compute median LOS intensity (MJy/sr) across the lens set for normalization.
    """
    I = sampler.sample_mjysr(ra_deg, dec_deg)
    I = I[np.isfinite(I)]
    if I.size == 0:
        raise ValueError("All sampled intensities are non-finite; check map, nside, or coordinates.")
    return float(np.median(I))
