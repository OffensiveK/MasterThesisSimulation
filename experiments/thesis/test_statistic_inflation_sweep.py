
# =========================================================================
# IMPORTS
# =========================================================================

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import N_GRID, Outputs, parser
from runtime.plotting import COLOURS, LABELS, MARKERS, plain_log_ticks, reference_line, use_style
from runtime.report import columnar, rows_from_columnar
from simulation.batching import batch_size_for, chunks
from simulation.continuous import HestonParams, simulate_continuous_increments_and_variance_batch
from simulation.estimators import (
    spot_volatility_batch,
    studentize_with,
    truncation_threshold,
    window_size,
)
from simulation.monte_carlo import rate_error
from simulation.evt import REJECT, STATISTIC_BATCH, TESTS

use_style()

RATE_TICKS = (0.05, 0.1, 0.2, 0.5)

# =========================================================================
# SIMULATION
# =========================================================================

def statistic_inflation_batch(n, batch_paths, heston, rng, k_n, u_n, level):
    increments, true_variance = simulate_continuous_increments_and_variance_batch(
        batch_paths, n, heston, rng
    )
    z_true = studentize_with(increments, true_variance)
    z_hat = studentize_with(increments, spot_volatility_batch(increments, k_n, u_n))

    gaps, reject_true, reject_hat = {}, {}, {}
    for test in TESTS:
        st = STATISTIC_BATCH[test](z_true)
        sh = STATISTIC_BATCH[test](z_hat)
        gaps[test] = sh - st
        reject_true[test] = REJECT[test](st, level)
        reject_hat[test] = REJECT[test](sh, level)
    return gaps, reject_true, reject_hat


def statistic_inflation(n, n_paths, heston, rng, k_n_exponent=0.5, scale=3.0, tau=0.475,
                        level=0.05, batch_size=None):
    k_n = window_size(n, k_n_exponent)
    u_n = truncation_threshold(n, scale, tau)
    batch_size = batch_size_for(n, n_paths, arrays_per_path=6) if batch_size is None else batch_size

    gap_chunks = {test: [] for test in TESTS}
    reject_true = {test: 0 for test in TESTS}
    reject_hat = {test: 0 for test in TESTS}
    for chunk in chunks(n_paths, batch_size):
        g, rt, rh = statistic_inflation_batch(n, chunk, heston, rng, k_n, u_n, level)
        for test in TESTS:
            gap_chunks[test].append(g[test])
            reject_true[test] += rt[test].sum()
            reject_hat[test] += rh[test].sum()
    gaps = {test: np.concatenate(gap_chunks[test]) for test in TESTS}
    return gaps, reject_true, reject_hat


def sweep(n_values, n_paths, heston, seed, level=0.05, verbose=True):
    rng = np.random.default_rng(seed)
    results = []
    for n in n_values:
        t0 = time.time()
        gaps, reject_true, reject_hat = statistic_inflation(n, n_paths, heston, rng, level=level)
        row = {"n": n}
        for test in TESTS:
            row[f"{test}_median_gap"] = float(np.median(gaps[test]))
            row[f"{test}_p_inflated"] = float(np.mean(gaps[test] > 0))
            row[f"{test}_reject_true"] = reject_true[test] / n_paths
            row[f"{test}_reject_hat"] = reject_hat[test] / n_paths
        results.append(row)
        if verbose:
            print(f"  n={n:>8}  ({time.time() - t0:.1f}s)", flush=True)
            for test in TESTS:
                print(f"      {test:8}  median_gap={row[f'{test}_median_gap']:+.4f}  "
                      f"P(inflated)={row[f'{test}_p_inflated']:.3f}  "
                      f"reject_true={row[f'{test}_reject_true']:.3f}  "
                      f"reject_hat={row[f'{test}_reject_hat']:.3f}", flush=True)
    return results


# =========================================================================
# PLOTTING
# =========================================================================

def plot_statistic_inflation(results, level, n_paths):
    n = np.array([r["n"] for r in results], dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    for test in TESTS:
        p_inflated = np.array([r[f"{test}_p_inflated"] for r in results])
        ax1.plot(n, p_inflated, marker=MARKERS[test], color=COLOURS[test], label=LABELS[test],
                 markersize=4, linewidth=1.2)
    reference_line(ax1, 0.5, dotted=True)
    ax1.set_xscale("log")
    ax1.set_xlabel("n")
    ax1.set_ylabel("fraction of paths")
    ax1.set_title(r"Test statistic larger on $\hat{Z}$ than on $\widetilde{Z}$")
    ax1.legend()

    for test in TESTS:
        reject_true = np.array([r[f"{test}_reject_true"] for r in results])
        reject_hat = np.array([r[f"{test}_reject_hat"] for r in results])
        
        ax2.errorbar(n, reject_hat, yerr=rate_error(np.asarray(reject_hat), n_paths),
                     color=COLOURS[test], label=f"{LABELS[test]} (full)",
                     linewidth=1.2, elinewidth=0.9)
        ax2.errorbar(n, reject_true, yerr=rate_error(np.asarray(reject_true), n_paths),
                     color=COLOURS[test], linestyle=":", alpha=0.5,
                     label=f"{LABELS[test]} (oracle)", linewidth=1.2, elinewidth=0.9)
    reference_line(ax2, level)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    plain_log_ticks(ax2, RATE_TICKS)
    ax2.set_xlabel("n")
    ax2.set_ylabel("rejection rate")
    ax2.set_title(r"Size on $\hat{Z}$ against $\widetilde{Z}$, same paths")
    ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/test_statistic_inflation_sweep",
               n_values=N_GRID, n_paths=16000, level=0.05, seed=0)
    args = p.parse_args()
    out = Outputs(args, "inflation")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams()

    if out.replotting:
        print(f"-- replotting from {out.json} --", flush=True)
        results = rows_from_columnar(json.loads(out.json.read_text())["inflation"])
    else:
        print("-- per-test statistic inflation from volatility misestimation (no jumps) --",
              flush=True)
        results = sweep(args.n_values, args.n_paths, heston, args.seed, level=args.level)

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_statistic_inflation(results, args.level, args.n_paths),
        title="Per-test statistic inflation from volatility misestimation",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "n_paths": args.n_paths,
            "level": args.level,
            "seed": args.seed,
            "heston": heston,
        },
        data={"inflation": columnar(results)},
    )


if __name__ == "__main__":
    main()
