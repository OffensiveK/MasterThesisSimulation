
# =========================================================================
# IMPORTS
# =========================================================================

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import Outputs, parser
from runtime.plotting import reference_line, use_style
from simulation.batching import batch_size_for, chunks
from simulation.continuous import HestonParams
from simulation.evt import GapRule, build_procedures, sorted_cache, validate_names
from simulation.scenarios import NULL, simulate_studentized, size_adjusted_calibration

use_style()

# Capped at 500 
BATCH_PATHS = 500

CANDIDATES = {
    "exp":          GapRule("upper",  2,    "single"),
    "renyi_single": GapRule("signed", 2,    "single"),
    "renyi":        GapRule("signed", None, "batch"),
    "renyi_deep":   GapRule("signed", None, "deepest"),
}

SHOWN = validate_names(("renyi", "renyi_single", "exp", "gumbel"), CANDIDATES)
COLOURS = {"renyi": "C0", "renyi_single": "C1", "exp": "C2", "gumbel": "C3"}

# =========================================================================
# SIMULATION
# =========================================================================

def batch_paths_for(n, n_paths):
    return min(BATCH_PATHS, batch_size_for(n, n_paths, arrays_per_path=8))


def fire_rates(z_hat, procedures):
    fired = {name: 0 for name in procedures}
    for i in range(z_hat.shape[1]):
        cache = sorted_cache(z_hat[:, i])
        for name, procedure in procedures.items():
            fired[name] += len(procedure(z_hat[:, i], cache)[0]) > 0
    return {name: count / z_hat.shape[1] for name, count in fired.items()}


def null_paths(n, runs, heston, rng, k_n_exponent=0.5):
    batch_paths = batch_paths_for(n, runs)
    out = []
    for take in chunks(runs, batch_paths):
        z_hat, _ = simulate_studentized(n, take, heston, NULL, rng, k_n_exponent)
        out.append(z_hat)
    return np.concatenate(out, axis=1)


# =========================================================================
# PLOTTING
# =========================================================================

def plot_robustness(scale_rows, bandwidth_rows, slopes, n_show, level):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for name in SHOWN:
        rows = sorted((r for r in scale_rows if r["procedure"] == name and r["n"] == n_show),
                      key=lambda r: r["factor"])
        axes[0].plot([r["factor"] for r in rows], [r["size"] for r in rows],
                     marker="o", color=COLOURS[name], label=name)
        rows = sorted((r for r in bandwidth_rows if r["procedure"] == name and r["n"] == n_show),
                      key=lambda r: r["gamma"])
        axes[1].plot([r["gamma"] for r in rows], [r["size"] for r in rows],
                     marker="o", color=COLOURS[name], label=name)
        rows = sorted((r for r in slopes if r["procedure"] == name), key=lambda r: r["n"])
        axes[2].plot([r["n"] for r in rows], [abs(r["slope"]) for r in rows],
                     marker="o", color=COLOURS[name], label=name)

    axes[0].set_xlabel(r"studentization factor $c$")
    axes[0].set_ylabel("size")
    axes[0].set_title(f"Multiplicative error (n = {n_show})", fontsize=10)
    axes[1].set_xlabel(r"bandwidth exponent $\gamma$ in $k_n = n^\gamma$")
    axes[1].set_ylabel("size")
    axes[1].set_title(f"Bandwidth misspecification (n = {n_show})", fontsize=10)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("n")
    axes[2].set_ylabel(r"$|\Delta$size$/\Delta c|$")
    axes[2].set_title("Sensitivity against n", fontsize=10)
    for ax in axes[:2]:
        reference_line(ax, level)
    reference_line(axes[0], 1.0, axis="x", dotted=True)
    reference_line(axes[1], 0.5, axis="x", dotted=True)
    axes[0].legend()

    fig.suptitle("Sensitivity of each flagging rule to studentization error")
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/supporting/flagging_robustness",
               n_values=[1000, 4000, 16000], n_paths=4000, level=0.05, seed=0)
    p.add_argument("--n-null", type=int, default=8000)
    p.add_argument("--scale-factors", type=float, nargs="+",
                   default=[0.90, 0.94, 0.97, 1.0, 1.03, 1.06, 1.10])
    p.add_argument("--gammas", type=float, nargs="+", default=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    args = p.parse_args()
    out = Outputs(args, "robustness")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams()
    rng = np.random.default_rng(args.seed)

    scale_rows = []
    bandwidth_rows = []
    slopes = []

    if out.replotting:
        print(f"-- replotting from {out.json} --", flush=True)
        saved = json.loads(out.json.read_text())
        scale_rows, bandwidth_rows, slopes = saved["scale"], saved["bandwidth"], saved["slopes"]
    else:
        for n in args.n_values:
            print(f"== n = {n} ==", flush=True)

            t0 = time.time()
            criticals, gumbel_level = size_adjusted_calibration(
                n, heston, dict(CANDIDATES), args.level, args.n_null, rng,
            )
            procedures, _ = build_procedures(
                CANDIDATES, n, args.level, criticals, gumbel_level=gumbel_level,
            )
            print(f"   calibrated on {args.n_null} null paths in {time.time() - t0:.1f}s",
                  flush=True)

            # An explicit multiplicative studentization error, which is exactly the perturbation
            # the 2 log n / q prediction is about.
            base = null_paths(n, args.n_paths, heston, rng)
            for c in args.scale_factors:
                rates = fire_rates(base * c, procedures)
                for name, rate in rates.items():
                    scale_rows.append({"n": n, "factor": c, "procedure": name, "size": rate})
                parts = "  ".join(f"{k}={v:.3f}" for k, v in rates.items())
                print(f"   scale c={c:.2f}  {parts}", flush=True)

            lo, hi = min(args.scale_factors), max(args.scale_factors)
            for name in procedures:
                by_factor = {r["factor"]: r["size"] for r in scale_rows
                             if r["n"] == n and r["procedure"] == name}
                slopes.append({
                    "n": n, "procedure": name,
                    "size_at_low": by_factor[lo], "size_at_high": by_factor[hi],
                    "slope": (by_factor[hi] - by_factor[lo]) / (hi - lo),
                })

            # A real estimator bias rather than an imposed one: the spot-volatility window
            # exponent moved away from the 1/2 used everywhere else.
            for gamma in args.gammas:
                z_hat = null_paths(n, args.n_paths, heston, rng, k_n_exponent=gamma)
                rates = fire_rates(z_hat, procedures)
                for name, rate in rates.items():
                    bandwidth_rows.append({"n": n, "gamma": gamma, "procedure": name,
                                           "size": rate})
                parts = "  ".join(f"{k}={v:.3f}" for k, v in rates.items())
                print(f"   gamma={gamma:.2f}  {parts}", flush=True)

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_robustness(scale_rows, bandwidth_rows, slopes, args.n_values[-1],
                    args.level),
        title="Flagging rules: sensitivity to studentization error",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "n_paths": args.n_paths,
            "n_null": args.n_null,
            "scale_factors": args.scale_factors,
            "gammas": args.gammas,
            "level": args.level,
            "seed": args.seed,
            "batch_paths": BATCH_PATHS,
            "heston": heston,
        },
        data={"scale": scale_rows, "bandwidth": bandwidth_rows, "slopes": slopes},
    )


if __name__ == "__main__":
    main()
