import os, json
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any

from .maps import (
    HealpixSampler,
    radiation_G_from_map,
    median_I0_for_lenses,
    load_healpix_map,
    smooth_map,
    sample_kappa_at_radec,
    apply_mask,
)

class DataHub:
    """
    Collection of loaders & proxies, now with:
      - HEALPix sampling for radiation proxy
      - κ_ext CSV ingestion for physical F(kappa)
    """
    def __init__(self, data_dir="./time_thickness_data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def load_kappa_ext_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load a per-lens kappa_ext table.
        REQUIRED: lens_id, kappa_ext
        OPTIONAL: sigma_kappa, source
        Returns DataFrame with ['lens_id','kappa_ext','sigma_kappa'].
        """
        df = pd.read_csv(csv_path, comment="#")
        required = {"lens_id", "kappa_ext"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"[kappa_ext CSV] Missing columns: {sorted(missing)}")

        # basic type/finite checks
        if not np.isfinite(df["kappa_ext"]).all():
            bad = df.loc[~np.isfinite(df["kappa_ext"]), ["lens_id","kappa_ext"]]
            raise ValueError(f"[kappa_ext CSV] Non-finite kappa_ext values:\n{bad.head()}")

        out = pd.DataFrame({
            "lens_id": df["lens_id"].astype(str),
            "kappa_ext": df["kappa_ext"].astype(float)
        })
        if "sigma_kappa" in df.columns:
            if np.issubdtype(df["sigma_kappa"].dtype, np.number):
                out["sigma_kappa"] = df["sigma_kappa"].astype(float)
            else:
                warnings.warn("[kappa_ext CSV] sigma_kappa not numeric; dropping.")
        return out

    # -----------------------
    # Lensing data loaders
    # -----------------------
    def load_tdcosmo_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load fully real time-delay pairs from a TDCOSMO/H0LiCOW-style CSV.

        REQUIRED columns:
          lens_id, pair_id, dt_obs_days, sig_obs_days, dt_gr_days, sig_gr_days,
          z_lens, z_src, ra_deg, dec_deg

        OPTIONAL:
          kappa_ext, sigma_kappa, theta_E_arcsec, sigma_v_km_s, ...

        Returns a DataFrame normalized to the internal schema expected by the fitter:
          [lens_id, pair_id, z_lens, z_src, dt_obs, sig_obs, dt_gr, sig_gr,
           ra_deg, dec_deg, kappa_ext (optional)]
        """
        df = pd.read_csv(csv_path, comment="#")
        required = {
            "lens_id", "pair_id", "dt_obs_days", "sig_obs_days",
            "dt_gr_days", "sig_gr_days", "z_lens", "z_src", "ra_deg", "dec_deg"
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"[TDCOSMO CSV] Missing required columns: {sorted(missing)}")

        # Basic type/finite checks
        for col in ["dt_obs_days", "sig_obs_days", "dt_gr_days", "sig_gr_days", "ra_deg", "dec_deg", "z_lens", "z_src"]:
            if not np.isfinite(df[col]).all():
                bad = df[~np.isfinite(df[col])]
                raise ValueError(f"[TDCOSMO CSV] Non-finite values in '{col}'\n{bad.head()}")

        # Enforce pair_id int
        try:
            df["pair_id"] = df["pair_id"].astype(int)
        except Exception:
            raise ValueError("[TDCOSMO CSV] 'pair_id' must be integer.")

        # Normalize to internal names
        out = pd.DataFrame({
            "lens_id":   df["lens_id"].astype(str),
            "pair_id":   df["pair_id"].astype(int),
            "z_lens":    df["z_lens"].astype(float),
            "z_src":     df["z_src"].astype(float),
            "dt_obs":    df["dt_obs_days"].astype(float),
            "sig_obs":   np.clip(df["sig_obs_days"].astype(float), 1e-6, None),
            "dt_gr":     df["dt_gr_days"].astype(float),
            "sig_gr":    np.clip(df["sig_gr_days"].astype(float), 1e-6, None),
            "ra_deg":    df["ra_deg"].astype(float),
            "dec_deg":   df["dec_deg"].astype(float),
            "dm_level":  "medium"  # Default for backward compatibility
        })

        # Optional κ_ext passthrough
        if "kappa_ext" in df.columns:
            if not np.isfinite(df["kappa_ext"]).all():
                warnings.warn("[TDCOSMO CSV] Non-finite kappa_ext values; rows will be NaN.")
            out["kappa_ext"] = df["kappa_ext"]
            
            # Also pass through sigma_kappa if present
            if "sigma_kappa" in df.columns:
                out["sigma_kappa"] = df["sigma_kappa"]

        return out

    def load_h0licow_like(self):
        """
        Returns DataFrame columns:
          lens_id, pair_id, z_lens, z_src,
          dt_obs, sig_obs, dt_gr, sig_gr,
          ra_deg, dec_deg,
          dm_level, kappa_ext (optional), rad_proxy (optional)
        """
        path = os.path.join(self.data_dir, "h0licow_lenses.json")
        if not os.path.exists(path):
            # Minimal, curated example (compatible with your earlier dict)
            lens_data = {
                "B1608+656": {
                    "measured_delays_days":[31.5, 36.0, 77.0],
                    "uncertainties_days":[1.5, 1.5, 1.5],
                    "h0licow_model_delays_days":[31.2, 35.8, 76.5],
                    "z_lens":0.630, "z_source":1.394,
                    "ra":242.275, "dec":65.706,
                    "dm_column":"high"
                },
                "RXJ1131-1231": {
                    "measured_delays_days":[91.0],
                    "uncertainties_days":[1.5],
                    "h0licow_model_delays_days":[90.3],
                    "z_lens":0.295, "z_source":0.654,
                    "ra":172.958, "dec":-12.532,
                    "dm_level":"medium"
                },
                "HE0435-1223": {
                    "measured_delays_days":[8.0, 2.1, 14.4],
                    "uncertainties_days":[0.8, 0.8, 0.8],
                    "h0licow_model_delays_days":[7.8, 2.0, 14.2],
                    "z_lens":0.454, "z_source":1.693,
                    "ra":69.676, "dec":-12.381,
                    "dm_column":"high"
                }
            }
            with open(path, "w") as f:
                json.dump(lens_data, f, indent=2)
        else:
            lens_data = json.load(open(path))

        rows = []
        for lens_id, d in lens_data.items():
            arr = zip(d["measured_delays_days"], d["uncertainties_days"], d["h0licow_model_delays_days"])
            for pair_id, (dt_obs, sig_obs, dt_gr) in enumerate(arr):
                rows.append(dict(
                    lens_id=lens_id, pair_id=pair_id,
                    z_lens=d["z_lens"], z_src=d["z_source"],
                    dt_obs=dt_obs, sig_obs=sig_obs,
                    dt_gr=dt_gr, sig_gr=max(0.1*sig_obs, 0.2),  # placeholder GR uncertainty
                    ra_deg=d["ra"], dec_deg=d["dec"],
                    dm_level=d.get("dm_column","medium"),
                    kappa_ext=np.nan,  # fill later if available
                    rad_proxy=np.nan
                ))
        return pd.DataFrame(rows)

    # ---------------
    # Pulsar metadata
    # ---------------
    def load_nanograv_like(self):
        """
        Returns a DataFrame with columns:
          pulsar, distance_kpc, dm_level, rms_residual_us, ra_deg, dec_deg
        """
        path = os.path.join(self.data_dir, "nanograv_pulsars.json")
        if not os.path.exists(path):
            pulsar_data = {
                "J0030+0451": {"distance_kpc":0.3, "dm_level":"low", "rms_residual_us":0.12, "ra":7.5, "dec": 4.85},
                "J0613-0200": {"distance_kpc":1.2, "dm_level":"medium","rms_residual_us":0.85,"ra":93.25,"dec":-2.0},
                "J1713+0747": {"distance_kpc":1.1, "dm_level":"low","rms_residual_us":0.065,"ra":258.25,"dec":7.78},
                "J1909-3744": {"distance_kpc":1.5, "dm_level":"low","rms_residual_us":0.055,"ra":287.25,"dec":-37.73}
            }
            with open(path, "w") as f:
                json.dump({"pulsars":pulsar_data}, f, indent=2)
        else:
            pulsar_data = json.load(open(path))["pulsars"]

        rows = []
        for name, d in pulsar_data.items():
            rows.append(dict(
                pulsar=name,
                distance_kpc=d["distance_kpc"],
                dm_level=d["dm_level"],
                rms_residual_us=d["rms_residual_us"],
                ra_deg=d["ra"], dec_deg=d["dec"]
            ))
        return pd.DataFrame(rows)

    # --------------
    # Clock datasets
    # --------------
    def load_clock_like(self):
        """
        Returns a DataFrame with columns:
          comparison, t (index/epoch), r (residual fractional freq),
          T1, T2, w (optional weights)
        Here we generate a tiny synthetic dataset consistent with the model
        to exercise the WLS pipeline; replace with real time series to bound eps_flat.
        """
        # Synthetic 100 points around room temp vs cryo
        n = 100
        T1 = np.full(n, 300.0)           # K
        T2 = np.where(np.arange(n)%2==0, 300.0, 77.0)
        # Simulate tiny residuals ~ 1e-18 (noise dominated)
        rng = np.random.default_rng(42)
        r_noise = rng.normal(0.0, 5e-19, size=n)
        return pd.DataFrame({
            "comparison": ["Yb_vs_Yb"] * n,
            "t": np.arange(n),
            "r": r_noise,
            "T1": T1,
            "T2": T2,
            "w": np.ones(n)  # uniform weights
        })
        
    def load_clocks_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load real optical clock residual time series.

        REQUIRED:
          comparison, epoch_mjd, r, T1_K, T2_K
        OPTIONAL:
          w  (weights, inverse variance)

        Returns columns: comparison, t (MJD), r, T1, T2, w
        """
        df = pd.read_csv(csv_path, comment="#")
        need = {"comparison", "epoch_mjd", "r", "T1_K", "T2_K"}
        miss = need.difference(df.columns)
        if miss:
            raise ValueError(f"[clocks CSV] Missing columns: {sorted(miss)}")
            
        for col in ["epoch_mjd", "r", "T1_K", "T2_K"]:
            if not np.isfinite(df[col]).all():
                bad = df[~np.isfinite(df[col])]
                raise ValueError(f"[clocks CSV] Non-finite in '{col}':\n{bad.head()}")
                
        out = pd.DataFrame({
            "comparison": df["comparison"].astype(str),
            "t": df["epoch_mjd"].astype(float),
            "r": df["r"].astype(float),
            "T1": df["T1_K"].astype(float),
            "T2": df["T2_K"].astype(float),
            "w": df["w"].astype(float) if "w" in df.columns else np.ones(len(df), float)
        })
        return out
        
    def load_pulsar_residuals_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load achromatic pulsar timing residuals (per TOA).

        REQUIRED:
          pulsar, toa_mjd, resid_us, resid_err_us
        OPTIONAL:
          distance_kpc, ra_deg, dec_deg

        Returns a compact per-PSR summary for the current WLS:
          pulsar, distance_kpc, rms_residual_us, ra_deg, dec_deg
        """
        df = pd.read_csv(csv_path, comment="#")
        need = {"pulsar", "toa_mjd", "resid_us", "resid_err_us"}
        miss = need.difference(df.columns)
        if miss:
            raise ValueError(f"[pulsar CSV] Missing columns: {sorted(miss)}")

        # Basic checks
        for col in ["toa_mjd", "resid_us", "resid_err_us"]:
            if not np.isfinite(df[col]).all():
                bad = df[~np.isfinite(df[col])]
                raise ValueError(f"[pulsar CSV] Non-finite in '{col}':\n{bad.head()}")

        # Aggregate per pulsar (RMS of achromatic residuals)
        rows = []
        for psr, g in df.groupby("pulsar"):
            rms = float(np.sqrt(np.mean((g["resid_us"].to_numpy(float))**2)))
            # pick distance / RA/Dec if present (take median across TOAs)
            dist = float(np.median(g["distance_kpc"])) if "distance_kpc" in g.columns else np.nan
            ra   = float(np.median(g["ra_deg"])) if "ra_deg" in g.columns else np.nan
            dec  = float(np.median(g["dec_deg"])) if "dec_deg" in g.columns else np.nan
            rows.append(dict(
                pulsar=psr, distance_kpc=dist, rms_residual_us=rms,
                ra_deg=ra, dec_deg=dec, dm_level="medium"  # dm_level unused when real κ/rad used
            ))
        out = pd.DataFrame(rows)
        # Fill missing distances with a conservative default (1.0 kpc)
        if out["distance_kpc"].isna().any():
            out["distance_kpc"] = out["distance_kpc"].fillna(1.0)
        return out

    # -----------------------
    # Proxies for G(T) and F(kappa)
    # -----------------------
    def los_radiation_proxy(self, ra_deg: float, dec_deg: float) -> float:
        """
        Placeholder LOS radiation proxy.
        Replace with HEALPix sampling of Planck/IRIS (MJy/sr → normalized).
        """
        val = 0.5 + 0.5 * np.tanh(np.deg2rad(dec_deg)/1.5)
        return float(np.clip(val, 0, 1))

    def env_kappa_proxy(self, dm_level: str) -> float:
        """
        Fallback κ proxy from environment bucket (low/medium/high).
        Replace with published kappa_ext or galaxy-count based κ estimates.
        """
        mapping = {"low": 1e19, "medium": 5e19, "high": 1e20}
        return float(mapping.get(str(dm_level).lower(), 5e19))

    # -------------------------------
    # Physical kappa from CSV
    # -------------------------------
    def load_kappa_ext_csv(self, df_lens: pd.DataFrame, 
                          csv_path: str,
                          scale_to_kappa: Optional[float] = None) -> pd.DataFrame:
        """
        Merge published kappa_ext values into the lens frame.

        Required columns:
          - lens_id
          - kappa_ext

        Optional:
          - sigma_kappa  (used only for bookkeeping at this stage)

        If `scale_to_kappa` is given, we create column `kappa_ext_scaled = scale_to_kappa * kappa_ext` 
        and use that for modeling (original kappa_ext is preserved).
        """
        import os
        import numpy as np
        import pandas as pd
        
        # Input validation
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"kappa_ext CSV not found: {csv_path}")
            
        if not isinstance(df_lens, pd.DataFrame):
            raise TypeError(f"df_lens must be a pandas DataFrame, got {type(df_lens)}")
            
        if 'lens_id' not in df_lens.columns:
            raise ValueError("df_lens must contain 'lens_id' column")
        
        # Read and validate CSV
        try:
            kdf = pd.read_csv(csv_path)
            print(f"[INFO] Loaded kappa_ext from {os.path.basename(csv_path)} with {len(kdf)} entries")
        except Exception as e:
            raise IOError(f"Failed to read kappa_ext CSV {csv_path}: {str(e)}")
        
        # Check required columns
        required_columns = {"lens_id", "kappa_ext"}
        missing_columns = required_columns - set(kdf.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in {csv_path}: {', '.join(sorted(missing_columns))}")

        # Data validation
        if kdf.empty:
            raise ValueError(f"kappa_ext CSV {csv_path} is empty")
            
        if not kdf["lens_id"].is_unique:
            duplicates = kdf[kdf.duplicated("lens_id", keep=False)]["lens_id"].unique()
            raise ValueError(f"Duplicate lens_id values found in {csv_path}: {duplicates}")
            
        if not np.isfinite(kdf["kappa_ext"]).all():
            bad_rows = kdf[~np.isfinite(kdf["kappa_ext"])]
            raise ValueError(f"Non-finite kappa_ext values found in rows: {bad_rows.index.tolist()}")

        # Merge with lens data
        try:
            # Make a copy to avoid modifying the original
            merged = df_lens.copy()
            
            # Get the original number of rows for validation
            original_count = len(merged)
            
            # Create a clean copy of kappa_df with consistent lens_id formatting
            kdf_clean = kdf.copy()
            
            # Ensure lens_id is a string and strip any whitespace or quotes
            kdf_clean['lens_id'] = kdf_clean['lens_id'].astype(str).str.strip("\"' ").str.strip()
            
            # Create a temporary column for matching with consistent formatting
            merged_lens_ids = merged['lens_id'].astype(str).str.strip("\"' ").str.strip()
            kdf_lens_ids = kdf_clean['lens_id'].astype(str).str.strip("\"' ").str.strip()
            
            # Create a mapping from clean lens_id to kappa values
            kappa_map = dict(zip(kdf_lens_ids, kdf_clean['kappa_ext']))
            
            # Map kappa values to the merged dataframe using the clean lens_id
            merged['kappa_ext'] = merged_lens_ids.map(kappa_map)
            
            # If sigma_kappa is available, map it as well
            if 'sigma_kappa' in kdf_clean.columns:
                sigma_map = dict(zip(kdf_lens_ids, kdf_clean['sigma_kappa']))
                merged['sigma_kappa'] = merged_lens_ids.map(sigma_map)
            
            # Verify merge didn't drop or add rows
            if len(merged) != original_count:
                raise ValueError(f"Merge changed number of rows from {original_count} to {len(merged)}")
                
            # Calculate matched statistics
            n_matched = merged['kappa_ext'].notna().sum()
            print(f"[INFO] Matched kappa_ext for {n_matched}/{len(merged)} lenses")
            
            if n_matched == 0:
                print("[WARNING] No kappa_ext values were matched. Check lens_id values in both DataFrames.")
                print("  Sample lens_ids in kappa CSV:", kdf['lens_id'].head().tolist())
                print("  Sample lens_ids in lens data:", df_lens['lens_id'].head().tolist())
            
            # Apply scaling if requested
            if scale_to_kappa is not None:
                if 'kappa_ext' not in merged.columns:
                    raise ValueError("Cannot apply scaling: 'kappa_ext' column not found after merge")
                merged["kappa_ext_scaled"] = scale_to_kappa * merged["kappa_ext"]
                print(f"[INFO] Applied kappa_ext scaling: kappa_ext_scaled = {scale_to_kappa} * kappa_ext")
            
            return merged
            
        except Exception as e:
            raise RuntimeError(f"Failed to merge kappa_ext data: {str(e)}")

    # -------------------------------
    # Radiation field from HEALPix map
    # -------------------------------
    def fill_radiation_from_map(self, df_lens: pd.DataFrame,
                              sampler: HealpixSampler,
                              I0_mjysr: Optional[float] = 1.0,
                              gamma: float = 1.0,
                              auto_I0: bool = False) -> pd.DataFrame:
        """
        Sample the map at lens RA/Dec and compute rad_proxy = (I/I0)^gamma.
        
        Args:
            df_lens: DataFrame with ra_deg, dec_deg columns
            sampler: Initialized HealpixSampler instance
            I0_mjysr: Reference intensity [MJy/sr] for normalization.
                     If None and auto_I0=False, uses 1.0.
            gamma: Exponent for intensity scaling
            auto_I0: If True, compute I0 as median intensity across lens positions
                    (overrides any provided I0_mjysr)
            
        Returns:
            Copy of input DataFrame with added columns:
            - rad_proxy: (I/I0)^gamma
            - rad_I_mjysr: Raw intensity [MJy/sr]
            - rad_I0_mjysr: Reference intensity used
            - rad_gamma: Exponent used
        """
        # Sample raw intensities
        I = sampler.sample_mjysr(df_lens["ra_deg"].to_numpy(float),
                               df_lens["dec_deg"].to_numpy(float))
        
        # Compute I0 (either auto or use provided)
        if auto_I0 or (I0_mjysr is None):
            I0 = median_I0_for_lenses(sampler, df_lens["ra_deg"], df_lens["dec_deg"])
        else:
            I0 = float(I0_mjysr)
        
        # Compute dimensionless proxy
        Gmap = radiation_G_from_map(I, I0_mjysr=I0, gamma=gamma)
        
        # Create output with new columns
        out = df_lens.copy()
        out["rad_proxy"] = Gmap
        out["rad_I_mjysr"] = I
        out["rad_I0_mjysr"] = I0
        out["rad_gamma"] = gamma
        
        return out

    # --- Sky map sampling helpers ---
    def _validate_radec(self, df, ra_col: str = "ra_deg", dec_col: str = "dec_deg"):
        if ra_col not in df.columns or dec_col not in df.columns:
            raise KeyError(f"DataFrame must contain '{ra_col}' and '{dec_col}' columns for sky sampling.")
        ra = df[ra_col].to_numpy(float)
        dec = df[dec_col].to_numpy(float)
        finite = np.isfinite(ra) & np.isfinite(dec)
        if not finite.any():
            raise RuntimeError("No finite sky coordinates available for sampling.")
        return ra, dec, finite

    def fill_kappa_from_map_for_lenses(
        self,
        df_lens,
        kappa_map_path,
        kappa_field: int = 0,
        nest: bool = False,
        smooth_fwhm_arcmin: float = None,
        mask_path: str = None,
        mask_field: int = 0,
        kappa_scale: float = 1.0
    ):
        """
        Sample a κ map at lens sightlines. Adds columns:
          - kappa_map : raw map values
          - kappa_ext : scaled map values (using kappa_scale)

        Returns tuple (df_out, good_mask, stats_dict).
        """
        m, nside, is_nest = load_healpix_map(kappa_map_path, field=kappa_field, nest=nest)
        m = smooth_map(m, smooth_fwhm_arcmin)

        ra, dec, finite_coords = self._validate_radec(df_lens)
        vals = sample_kappa_at_radec(m, nside, ra, dec, nest=is_nest, fill_value=np.nan)

        if mask_path:
            mask_map, mask_nside, mask_is_nest = load_healpix_map(mask_path, field=mask_field, nest=nest)
            if mask_nside != nside:
                raise ValueError(f"Mask NSIDE ({mask_nside}) does not match map NSIDE ({nside}).")
            vals, mask_good = apply_mask(
                vals, mask_map, mask_nside, ra, dec, nest=mask_is_nest
            )
        else:
            mask_good = np.ones_like(vals, dtype=bool)

        finite_vals = np.isfinite(vals)
        good = finite_coords & mask_good & finite_vals
        if not good.any():
            raise RuntimeError("No valid κ samples available after masking.")

        scale = 1.0 if kappa_scale is None else float(kappa_scale)
        scaled = np.full_like(vals, np.nan, dtype=float)
        scaled[good] = vals[good] * scale

        out = df_lens.copy()
        out["kappa_map"] = vals
        out["kappa_ext"] = scaled

        stats = {
            "median_kappa": float(np.nanmedian(vals[good])),
            "p16_kappa": float(np.nanpercentile(vals[good], 16)),
            "p84_kappa": float(np.nanpercentile(vals[good], 84)),
            "scale": float(scale),
            "nside": int(nside),
            "n_kept": int(good.sum()),
            "n_total": int(len(vals))
        }
        return out, good, stats

    def fill_kappa_from_map_for_pulsars(
        self,
        df_psr,
        kappa_map_path,
        kappa_field: int = 0,
        nest: bool = False,
        smooth_fwhm_arcmin: float = None,
        mask_path: str = None,
        mask_field: int = 0,
        kappa_scale: float = 1.0
    ):
        """
        Sample κ map at pulsar sightlines. Mirrors lens helper but preserves
        the 'pulsar' column name.
        """
        tmp = df_psr.rename(columns={"pulsar": "lens_id"}) if "pulsar" in df_psr.columns else df_psr
        out_tmp, good, stats = self.fill_kappa_from_map_for_lenses(
            tmp,
            kappa_map_path,
            kappa_field=kappa_field,
            nest=nest,
            smooth_fwhm_arcmin=smooth_fwhm_arcmin,
            mask_path=mask_path,
            mask_field=mask_field,
            kappa_scale=kappa_scale
        )
        if "lens_id" in out_tmp.columns and "pulsar" not in df_psr.columns:
            out = out_tmp
        else:
            out = out_tmp.rename(columns={"lens_id": "pulsar"})
        return out, good, stats

    def fill_radiation_from_map_for_lenses(
        self,
        df_lens,
        rad_map_path,
        rad_field: int = 0,
        nest: bool = False,
        smooth_fwhm_arcmin: float = None,
        mask_path: str = None,
        mask_field: int = 0,
        auto_I0: bool = True,
        I0: float = None,
        gamma: float = 1.0
    ):
        """
        Sample a radiation map at lens sightlines and construct G proxies.

        Returns (df_out, good_mask, I0_used, stats_dict).
        """
        m, nside, is_nest = load_healpix_map(rad_map_path, field=rad_field, nest=nest)
        m = smooth_map(m, smooth_fwhm_arcmin)

        ra, dec, finite_coords = self._validate_radec(df_lens)
        vals = sample_kappa_at_radec(m, nside, ra, dec, nest=is_nest, fill_value=np.nan)

        if mask_path:
            mask_map, mask_nside, mask_is_nest = load_healpix_map(mask_path, field=mask_field, nest=nest)
            if mask_nside != nside:
                raise ValueError(f"Mask NSIDE ({mask_nside}) does not match map NSIDE ({nside}).")
            vals, mask_good = apply_mask(
                vals, mask_map, mask_nside, ra, dec, nest=mask_is_nest
            )
        else:
            mask_good = np.ones_like(vals, dtype=bool)

        finite_vals = np.isfinite(vals)
        good = finite_coords & mask_good & finite_vals
        vals_good = vals[good]
        if vals_good.size == 0:
            raise RuntimeError("No valid radiation samples after masking.")

        if auto_I0 and I0 is None:
            I0_used = float(np.nanmedian(vals_good))
        else:
            if I0 is None or I0 <= 0:
                raise ValueError("I0 must be positive when auto_I0=False.")
            I0_used = float(I0)

        G_raw = np.full_like(vals, 0.0, dtype=float)
        G_raw[good] = np.clip(vals_good / I0_used, 0.0, None)
        G_proxy = np.full_like(vals, 0.0, dtype=float)
        G_proxy[good] = np.power(G_raw[good], float(gamma))

        out = df_lens.copy()
        out["rad_map"] = vals
        out["G_raw"] = G_raw
        out["G_proxy"] = G_proxy

        stats = {
            "I0_used": I0_used,
            "median_rad": float(np.nanmedian(vals_good)),
            "p16_rad": float(np.nanpercentile(vals_good, 16)),
            "p84_rad": float(np.nanpercentile(vals_good, 84)),
            "gamma": float(gamma),
            "nside": int(nside),
            "n_kept": int(good.sum()),
            "n_total": int(len(vals))
        }
        return out, good, I0_used, stats

    def fill_radiation_from_map_for_pulsars(
        self,
        df_psr,
        rad_map_path,
        rad_field: int = 0,
        nest: bool = False,
        smooth_fwhm_arcmin: float = None,
        mask_path: str = None,
        mask_field: int = 0,
        auto_I0: bool = True,
        I0: float = None,
        gamma: float = 1.0
    ):
        """
        Same as lenses, but preserves pulsar naming.
        """
        auto_flag = auto_I0 and (I0 is None)
        tmp = df_psr.rename(columns={"pulsar": "lens_id"}) if "pulsar" in df_psr.columns else df_psr
        out_tmp, good, I0_used, stats = self.fill_radiation_from_map_for_lenses(
            tmp,
            rad_map_path,
            rad_field=rad_field,
            nest=nest,
            smooth_fwhm_arcmin=smooth_fwhm_arcmin,
            mask_path=mask_path,
            mask_field=mask_field,
            auto_I0=auto_flag,
            I0=I0,
            gamma=gamma
        )
        if "lens_id" in out_tmp.columns and "pulsar" not in df_psr.columns:
            out = out_tmp
        else:
            out = out_tmp.rename(columns={"lens_id": "pulsar"})
        return out, good, I0_used, stats
