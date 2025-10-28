# Data Directory

This directory should contain the following files for the IVI time-thickness analysis:

## Required Files

1. **Planck 857 GHz HEALPix Map**
   - File: `planck_857_healpix.fits`
   - Description: HEALPix map of the Planck 857 GHz sky emission
   - How to obtain:
     ```bash
     wget https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/foregrounds/HFI_CompMap_ThermalDustModel_2048_R3.00_0256.fits -O planck_857_healpix.fits
     ```

2. **TDCOSMO Time-Delay Data**
   - File: `tdcosmo_time_delays.csv`
   - Description: CSV file containing time-delay measurements from TDCOSMO/H0LiCOW
   - Format: See example in the file

3. **Kappa Extinction Data**
   - File: `h0licow_kappa_ext.csv`
   - Description: CSV file with kappa_ext values for lens systems
   - Format: `lens_id,kappa_ext,sigma_kappa`

## Directory Structure

```
data/
├── planck_857_healpix.fits        # Planck 857 GHz HEALPix map (optional)
├── tdcosmo_time_delays.csv        # TDCOSMO time-delay data
├── h0licow_kappa_ext.csv          # κ_ext values for lens systems
└── quick/
    ├── tdcosmo_quick.csv
    ├── kappa_ext_quick.csv
    ├── clocks_quick.csv
    └── pulsars_quick.csv
```

The `quick/` subdirectory contains a minimal dataset used by `make quick` for
fast regression smoke tests. Each file is checksummed in `data/SHA256SUMS`.

## Running the Analysis

Once all files are in place, you can run the analysis with:

```bash
python scripts/run_analysis.py \
  --config configs/physical.yaml \
  --tdcosmo-csv data/tdcosmo_time_delays.csv \
  --healpix data/planck_857_healpix.fits \
  --nside 2048 \
  --nest \
  --auto-I0 \
  --kappa-csv data/h0licow_kappa_ext.csv \
  --kappa-scale 1.0 \
  --plots
```
