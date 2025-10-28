PYTHON ?= python3
SEED ?= 0

.PHONY: data verify quick full clean

data:
	@echo "[DATA] Downloading Planck 857 GHz map (if missing)..."
	$(PYTHON) download_planck_map.py
	@echo "[DATA] Verifying tracked data hashes..."
	$(PYTHON) scripts/check_data_integrity.py

verify:
	$(PYTHON) scripts/check_data_integrity.py

quick:
	$(PYTHON) scripts/run_analysis.py \
		--config configs/default.yaml \
		--tdcosmo-csv data/quick/tdcosmo_quick.csv \
		--kappa-csv data/quick/kappa_ext_quick.csv \
		--clock-csv data/quick/clocks_quick.csv \
		--pulsar-csv data/quick/pulsars_quick.csv \
		--output-dir results/quick_run \
		--no-plots \
		--no-jackknife \
		--n-perm 0 \
		--rng-seed $(SEED)
	@echo "[PLOTS] Generating publication figures for quick run"
	$(PYTHON) scripts/plot_publication_figures.py --results-dir results/quick_run

ifndef RAD_MAP
full:
	$(error Please set RAD_MAP to the Planck radiation map path, e.g. make full RAD_MAP=data/planck/HFI_SkyMap_857_2048_R2.02_full.fits KAPPA_MAP=/path/to/kappa.fits)
else ifndef KAPPA_MAP
full:
	$(error Please set KAPPA_MAP to the κ map path, e.g. make full KAPPA_MAP=/path/to/COM_CompMap_Lensing_4096_R3.00_kappa.fits RAD_MAP=data/planck/HFI_SkyMap_857_2048_R2.02_full.fits)
else
full:
	$(PYTHON) scripts/run_analysis.py \
		--config configs/default.yaml \
		--tdcosmo-csv data/tdcosmo_time_delays.csv \
		--kappa-csv data/kappa_ext.csv \
		--kappa-map $(KAPPA_MAP) \
		--kappa-field $(or $(KAPPA_FIELD),0) \
		--rad-map $(RAD_MAP) \
		--rad-field $(or $(RAD_FIELD),0) \
		--auto-I0 \
		--require-sky \
		--rng-seed $(SEED) \
		--output-dir results/full_run \
		--plots
	$(PYTHON) scripts/plot_publication_figures.py --results-dir results/full_run
endif

clean:
	rm -rf results/quick_run results/full_run
