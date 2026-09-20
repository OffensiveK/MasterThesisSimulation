
# =========================================================================
# IMPORTS
# =========================================================================

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import N_GRID, Outputs, parser
from runtime.plotting import reference_line, use_style
from runtime.report import columnar, rows_from_columnar, write_report
from simulation.batching import batch_size_for, chunks
from simulation.continuous import HestonParams, simulate_continuous_increments_and_variance_batch
from simulation.monte_carlo import loglog_slope, percentile_ci, rate_error
from simulation.estimators import (
    VARIANCE_FLOOR,
    spot_volatility_batch,
    studentize_with,
    truncation_threshold,
    window_size,
)

use_style()

LOCATIONS = ["max", "random", "oracle", "hat"]
CURVE_COLOUR = "tab:red"

# =========================================================================
# SIMULATION
# =========================================================================

def errors_for_batch(n, batch_paths, heston, rng, k_n, u_n):
    increments, true_variance = simulate_continuous_increments_and_variance_batch(
        batch_paths, n, heston, rng
    )
    sigma2_hat = spot_volatility_batch(increments, k_n, u_n)
    z_true = studentize_with(increments, true_variance)
    z_hat = studentize_with(increments, sigma2_hat)

    err = np.abs(sigma2_hat - true_variance)
    paths = np.arange(batch_paths)
    j_random = rng.integers(0, n, size=batch_paths)
    j_oracle = np.argmax(np.abs(z_true), axis=0)
    j_hat = np.argmax(np.abs(z_hat), axis=0)

    absolute = {"max": err.max(axis=0)}
    relative = {}
    for label, j in (("random", j_random), ("oracle", j_oracle), ("hat", j_hat)):
        absolute[label] = err[j, paths]
        relative[label] = err[j, paths] / np.maximum(true_variance[j, paths], VARIANCE_FLOOR)
    return absolute, relative, (j_hat != j_oracle)


def error_localization(n, n_paths, heston, rng, batch_size=None):
    k_n = window_size(n)
    u_n = truncation_threshold(n)
    batch_size = batch_size_for(n, n_paths, arrays_per_path=5) if batch_size is None else batch_size

    absolute = {label: [] for label in LOCATIONS}
    relative = {label: [] for label in LOCATIONS[1:]}
    hijacked = []
    for chunk in chunks(n_paths, batch_size):
        abs_b, rel_b, hij_b = errors_for_batch(n, chunk, heston, rng, k_n, u_n)
        for label in LOCATIONS:
            absolute[label].append(abs_b[label])
        for label in LOCATIONS[1:]:
            relative[label].append(rel_b[label])
        hijacked.append(hij_b)

    absolute = {k: np.concatenate(v) for k, v in absolute.items()}
    relative = {k: np.concatenate(v) for k, v in relative.items()}
    return absolute, relative, np.concatenate(hijacked), k_n


def ratio_with_bootstrap(hat_per_path, random_per_path, n_boot, seed):
    rng = np.random.default_rng(seed)
    out = []
    for hat, random in zip(hat_per_path, random_per_path):
        draws = np.empty(n_boot)
        for b in range(n_boot):
            take = rng.integers(0, hat.size, size=hat.size)
            draws[b] = np.median(hat[take]) / np.median(random[take])
        lo, hi = percentile_ci(draws)
        out.append({"ratio": float(np.median(hat) / np.median(random)),
                    "ci_lo": lo, "ci_hi": hi})
    return out


def fit_with_bootstrap(n, per_path, n_boot, seed):
    medians = np.array([np.median(v) for v in per_path])
    beta = loglog_slope(n, medians)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        resampled = [np.median(rng.choice(v, size=v.size, replace=True)) for v in per_path]
        draws[b] = loglog_slope(n, np.array(resampled))
    lo, hi = percentile_ci(draws)
    return {"beta": beta, "ci_lo": lo, "ci_hi": hi, "medians": medians}


# =========================================================================
# PLOTTING
# =========================================================================

def plot_error_localization(n_grid, fits, rows, ratio_ci=None, n_paths=None):

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ratio = fits["hat"]["medians"] / fits["random"]["medians"]
   
    yerr = None
    if ratio_ci is not None:
        yerr = np.vstack([ratio - np.array([c["ci_lo"] for c in ratio_ci]),
                          np.array([c["ci_hi"] for c in ratio_ci]) - ratio])
    ax.errorbar(n_grid, ratio, yerr=yerr, color=CURVE_COLOUR, linestyle="-", linewidth=1.2,
                elinewidth=1.0)
    
    reference_line(ax, 1.0, color="black")
    ax.set_xscale("log")
    ax.set_ylim(0.95, 2.05)
    ax.set_xlabel("$n$")
   
    ax.set_ylabel("selection factor")
    ax.set_title("How much larger the error is where the maximum lands")

    hijack = np.array([r["hijack_fraction"] for r in rows])
    ax2.errorbar(n_grid, hijack, yerr=rate_error(hijack, n_paths), color=CURVE_COLOUR,
                 linestyle="-", linewidth=1.2, elinewidth=1.0)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("$n$")
    ax2.set_ylabel("fraction of paths")
    ax2.set_title(r"Argmax moves, $\arg\max_j|\hat{Z}_j| \neq \arg\max_j|\widetilde{Z}_j|$")
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/error_localization_sweep",
               n_values=N_GRID, n_paths=8000, seed=0, level=None)
    p.add_argument("--n-boot", type=int, default=400)
    args = p.parse_args()
    out = Outputs(args, "localization")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams()
    rng = np.random.default_rng(args.seed)

    if out.replotting:
        print(f"-- replotting from {out.json} --", flush=True)
        saved = json.loads(out.json.read_text())
        rows = rows_from_columnar(saved["localization"])
        n_grid = np.array([row["n"] for row in rows], dtype=float)
        fits = {label: dict(saved["fits"][label],
                            medians=np.array([row[f"median_abs_{label}"] for row in rows]))
                for label in LOCATIONS}
        plot_error_localization(n_grid, fits, rows, saved.get("ratio_ci"),
                                args.n_paths).savefig(out.figure)
        print(f"Saved {out.figure}")
        return

    print("-- spot-volatility error by location along the path (no jumps) --", flush=True)
    rows, abs_paths, rel_paths = [], {k: [] for k in LOCATIONS}, {k: [] for k in LOCATIONS[1:]}
    for n in args.n_values:
        t0 = time.time()
        absolute, relative, hijacked, k_n = error_localization(n, args.n_paths, heston, rng)
        for label in LOCATIONS:
            abs_paths[label].append(absolute[label])
        for label in LOCATIONS[1:]:
            rel_paths[label].append(relative[label])

        row = {"n": n, "k_n": k_n, "independent_windows": n / k_n,
               "hijack_fraction": float(hijacked.mean())}
        row.update({f"median_abs_{k}": float(np.median(v)) for k, v in absolute.items()})
        row.update({f"median_rel_{k}": float(np.median(v)) for k, v in relative.items()})
        rows.append(row)

        print(f"  n={n:>8}  k_n={k_n:>4}  max={row['median_abs_max']:.5f}  "
              f"random={row['median_abs_random']:.5f}  oracle={row['median_abs_oracle']:.5f}  "
              f"hat={row['median_abs_hat']:.5f}  hijack={row['hijack_fraction']:.3f}  "
              f"({time.time() - t0:.1f}s)", flush=True)

    n_grid = np.array(args.n_values, dtype=float)
    fits = {label: fit_with_bootstrap(n_grid, abs_paths[label], args.n_boot, args.seed + 1 + i)
            for i, label in enumerate(LOCATIONS)}
    ratio_ci = ratio_with_bootstrap(abs_paths["hat"], abs_paths["random"], args.n_boot,
                                    args.seed + 21)
    rel_fits = {label: fit_with_bootstrap(n_grid, rel_paths[label], args.n_boot, args.seed + 11 + i)
                for i, label in enumerate(LOCATIONS[1:])}

    print("\nfitted beta (95% bootstrap CI over paths):")
    for label in LOCATIONS:
        f = fits[label]
        print(f"  absolute @ {label:<7} beta={f['beta']:.3f}  [{f['ci_lo']:.3f}, {f['ci_hi']:.3f}]")
    for label in LOCATIONS[1:]:
        f = rel_fits[label]
        print(f"  relative @ {label:<7} beta={f['beta']:.3f}  [{f['ci_lo']:.3f}, {f['ci_hi']:.3f}]")

    # =====================================================================
    # OUTPUT
    # =====================================================================

    plot_error_localization(n_grid, fits, rows, ratio_ci, args.n_paths).savefig(out.figure)
    write_report(
        out.report,
        title="Spot-volatility estimator error by location along the path",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "n_paths": args.n_paths,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "heston": heston,
            "fits_absolute": {k: {kk: vv for kk, vv in v.items() if kk != "medians"}
                              for k, v in fits.items()},
            "fits_relative": {k: {kk: vv for kk, vv in v.items() if kk != "medians"}
                              for k, v in rel_fits.items()},
        },
        data={"localization": columnar(rows),
              "fits": {k: {kk: vv for kk, vv in v.items() if kk != "medians"}
                       for k, v in fits.items()},
              "ratio_ci": ratio_ci},
    )
    print(f"\nSaved {out.figure} and {out.report} / .json")


if __name__ == "__main__":
    main()
