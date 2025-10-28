#!/usr/bin/env python3
"""
Joint MCMC for IVI time–thickness parameters.

Example:
  python scripts/run_mcmc.py \
    --config configs/mcmc.yaml \
    --healpix ./data/planck_857.fits --nside 2048 \
    --kappa-csv ./data/kappa_ext.csv \
    --use-cov --rho-intra-lens 0.25 \
    --walkers 32 --steps 6000 --burn 2000 \
    --outdir results/mcmc_run

Requires: emcee, (optional) healpy, astropy (for maps).
"""
import argparse, yaml, os, json, numpy as np

from ivi_thickness.model import Params
from ivi_thickness.data import DataHub
from ivi_thickness.maps import MapSampler
from ivi_thickness.bayes import (
    make_priors, build_joint_data, log_posterior, run_emcee
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mcmc.yaml")
    ap.add_argument("--healpix", default=None)
    ap.add_argument("--nside", type=int, default=None)
    ap.add_argument("--nest", action="store_true")
    ap.add_argument("--kappa-csv", default=None)
    ap.add_argument("--use-cov", action="store_true")
    ap.add_argument("--rho-intra-lens", type=float, default=0.3)
    ap.add_argument("--walkers", type=int, default=24)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--outdir", default="results/mcmc_out")
    ap.add_argument("--rng-seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Params used only for constructing GT and FK (E0, p, q here are seeds;
    # the MCMC will vary them internally; we rebuild GT,FK inside log-lik
    # using theta values, so we pass a "seed" Params only for initial map scalings)
    seed_params = Params(
        epsilon_grain=cfg["params"]["epsilon_grain"],
        epsilon_flat=cfg["params"]["epsilon_flat"],
        E0_eV=cfg["params"]["E0_eV"],
        kappa0=cfg["params"]["kappa0"],
        p=cfg["params"]["p"],
        q=cfg["params"]["q"]
    )

    hub = DataHub(cfg["io"]["data_dir"])
    df_lens = hub.load_h0licow_like()
    if args.kappa_csv:
        df_lens = hub.load_kappa_ext_csv(df_lens, args.kappa_csv)
    if args.healpix:
        sampler = MapSampler(args.healpix, nside=args.nside, nest=args.nest)
        df_lens = hub.fill_radiation_from_map(df_lens, sampler)

    df_clock = hub.load_clock_like()
    df_psr   = hub.load_nanograv_like()

    # Priors
    P = make_priors(**cfg.get("priors", {}))

    # Build "static" containers for data *indices* and measurement errors.
    # NOTE: GT and FK will be recomputed inside a closure with per-draw theta (p,q,E0).
    # To enable that, we pass a seed now and replace GT,FK on-the-fly in logpost.
    JD_seed = build_joint_data(
        df_lens, df_clock, df_psr, seed_params,
        use_cov=args.use_cov, rho_intra=args.rho_intra_lens,
        puls_frac_unc=cfg.get("pulsar", {}).get("frac_unc", 0.2)
    )

    # Closure for log-posterior that rebuilds GT,FK with current theta
    from ivi_thickness.model import G_temp, F_kappa, Params as Pm
    def make_logpost(JD0):
        def _logpost(theta):
            (lg_eg, lg_ef, p, q, lg_E0,
             a_l, a_c, a_p, ln_s_l, ln_s_c, ln_s_p) = theta

            # rebuild GT,FK with current (p,q,E0)
            params_draw = Pm(
                epsilon_grain=10.0**lg_eg,
                epsilon_flat=(10.0**lg_ef) * P.sign_eps_flat,
                E0_eV=10.0**lg_E0,
                kappa0=seed_params.kappa0,
                p=p, q=q
            )

            # update lens GT,FK in a shallow copy
            GT = G_temp(JD0.lens.GT, params_draw) if JD0.lens.GT.ndim==1 else G_temp(np.asarray(JD0.lens.GT), params_draw)
            FK = F_kappa(JD0.lens.FK, params_draw) if JD0.lens.FK.ndim==1 else F_kappa(np.asarray(JD0.lens.FK), params_draw)

            lens = type(JD0.lens)(
                R=JD0.lens.R, sigR=JD0.lens.sigR,
                GT=GT, FK=FK, lens_ids=JD0.lens.lens_ids,
                use_cov=JD0.lens.use_cov, rho_intra=JD0.lens.rho_intra
            )
            # clocks: need G(T1), G(T2) with current params
            from ivi_thickness.bayes import ClockBlock, PulsarBlock, JointData, log_posterior as _lp
            # reconstruct G1,G2 from original T1,T2: we don't have them stored in JD0;
            # so we stash them in JD0.clock.sig field? Instead, read from df_clock again (OK for small sets):
            T1 = df_clock["T1"].to_numpy(float)
            T2 = df_clock["T2"].to_numpy(float)
            G1 = G_temp(T1, params_draw)
            G2 = G_temp(T2, params_draw)
            clock = ClockBlock(r=JD0.clock.r, sig=JD0.clock.sig, G1=G1, G2=G2)

            # pulsar x depends only weakly on (p) via F(kappa proxy); keep as is (x prebuilt)
            puls  = JD0.puls
            JD    = JointData(lens=lens, clock=clock, puls=puls)
            return _lp(theta, JD, P)
        return _logpost

    logpost = make_logpost(JD_seed)

    # init vector (center of priors)
    def mid(lo, hi): return 0.5*(lo+hi)
    theta0 = np.array([
        mid(*P.log10_eps_grain),
        mid(*P.log10_eps_flat),
        mid(*P.p),
        mid(*P.q),
        mid(*P.log10_E0_eV),
        0.0, 0.0, 0.0,                  # a_l, a_c, a_p
        np.log(1e-3), np.log(1e-3), np.log(1e-3)   # ln s_*
    ], dtype=float)

    # run
    out = run_emcee(logpost, theta0,
                    nwalkers=args.walkers, nsteps=args.steps, burn=args.burn,
                    progress=True, random_state=args.rng_seed)

    # save
    np.savez(os.path.join(args.outdir, "mcmc_chain.npz"),
             chain=out["chain"], log_prob=out["log_prob"], rng_seed=args.rng_seed)
    with open(os.path.join(args.outdir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.outdir, "run_metadata.json"), "w") as f:
        json.dump({
            "rng_seed": args.rng_seed,
            "walkers": args.walkers,
            "steps": args.steps,
            "burn": args.burn,
            "use_cov": args.use_cov,
            "rho_intra_lens": args.rho_intra_lens,
            "kappa_csv": args.kappa_csv,
            "healpix": args.healpix
        }, f, indent=2)

    # quick summary
    chain = out["chain"]
    med = np.median(chain, axis=0)
    lo  = np.percentile(chain, 16, axis=0)
    hi  = np.percentile(chain, 84, axis=0)
    names = ["log10_eps_grain","log10_eps_flat","p","q","log10_E0_eV",
             "a_lens","a_clock","a_puls","ln_s_lens","ln_s_clock","ln_s_puls"]
    print("\nMCMC posterior (median [16%,84%]):")
    for k, m, l, h in zip(names, med, lo, hi):
        print(f"  {k:16s} = {m:+.4f}  [{l:+.4f}, {h:+.4f}]")

    # optional corner plot
    try:
        import matplotlib.pyplot as plt
        import corner
        fig = corner.corner(chain, labels=names, show_titles=True, title_fmt=".3f")
        fig.savefig(os.path.join(args.outdir, "corner.png"), dpi=160)
        print(f"\nSaved: {os.path.join(args.outdir, 'corner.png')}")
    except Exception as e:
        print(f"(corner plot skipped: {e})")

if __name__ == "__main__":
    main()
