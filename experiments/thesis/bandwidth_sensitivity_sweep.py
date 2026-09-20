
# =========================================================================
# IMPORTS
# =========================================================================

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import Outputs, parser
from runtime.plotting import LABELS, plain_log_ticks, reference_line, use_style
from runtime.report import columnar, rows_from_columnar
from simulation.batching import batch_size_for, chunks
from simulation.continuous import HestonParams, simulate_continuous_increments_and_variance_batch
from simulation.estimators import (
    spot_volatility_batch,
    studentize_with,
    truncation_threshold,
    window_size,
)
from simulation.evt import REJECT, STATISTIC_BATCH, TESTS
from simulation.monte_carlo import rate_error

use_style()

# Viridis run to its top end finishes on a glaring yellow, so the ramp stops short of it.
CMAP_SPAN = 0.85
SIZE_TICKS = (0.05, 0.1, 0.2, 0.5, 1.0)

# =========================================================================
# SIMULATION
# =========================================================================

def rejections_for_batch(n, batch_paths, heston, rng, gammas, u_n, level):
    increments, true_variance = simulate_continuous_increments_and_variance_batch(
        batch_paths, n, heston, rng
    )
    z_true = studentize_with(increments, true_variance)
    oracle = {test: REJECT[test](STATISTIC_BATCH[test](z_true), level).sum() for test in TESTS}

    full = {}
    for gamma in gammas:
        k_n = window_size(n, gamma)
        z_hat = studentize_with(increments, spot_volatility_batch(increments, k_n, u_n))
        full[gamma] = {test: REJECT[test](STATISTIC_BATCH[test](z_hat), level).sum()
                       for test in TESTS}
    return oracle, full


def size_by_bandwidth(n, n_paths, heston, rng, gammas, level=0.05):
    u_n = truncation_threshold(n)
    batch_size = batch_size_for(n, n_paths, arrays_per_path=4 + len(gammas))

    oracle = {test: 0 for test in TESTS}
    full = {gamma: {test: 0 for test in TESTS} for gamma in gammas}
    for chunk in chunks(n_paths, batch_size):
        o_b, f_b = rejections_for_batch(n, chunk, heston, rng, gammas, u_n, level)
        for test in TESTS:
            oracle[test] += o_b[test]
            for gamma in gammas:
                full[gamma][test] += f_b[gamma][test]

    oracle = {test: oracle[test] / n_paths for test in TESTS}
    full = {gamma: {test: full[gamma][test] / n_paths for test in TESTS} for gamma in gammas}
    return oracle, full


# =========================================================================
# PLOTTING
# =========================================================================

def plot_size_by_bandwidth(rows, n_values, gammas, n_paths, level):
    x = np.array(gammas, dtype=float)

    fig, axes = plt.subplots(1, len(TESTS), figsize=(4.0 * len(TESTS), 4.0), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, test in zip(axes, TESTS):
        for i, n in enumerate(n_values):
            y = np.array([rows[i][f"{test}_{g}"] for g in gammas])
            ax.errorbar(x, y, yerr=rate_error(y, n_paths), marker="o", markersize=4,
                        color=cmap(CMAP_SPAN * i / max(1, len(n_values) - 1)),
                        label=f"$n$={n}", linewidth=1.2, elinewidth=1.0)
        reference_line(ax, level, dotted=True, color="black")
        ax.set_yscale("log")
        plain_log_ticks(ax, SIZE_TICKS)
        ax.set_xlabel(r"bandwidth exponent $\gamma$")
        ax.set_title(LABELS[test])
    axes[0].set_ylabel("empirical size")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/bandwidth_sensitivity_sweep",
               n_values=[200, 619, 1916, 5932, 18362, 56838], n_paths=6000, level=0.05, seed=0)
    p.add_argument("--gammas", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    args = p.parse_args()
    out = Outputs(args, "bandwidth")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams()
    rng = np.random.default_rng(args.seed)

    if out.replotting:
        print(f"-- replotting from {out.json} --", flush=True)
        rows = rows_from_columnar(json.loads(out.json.read_text())["bandwidth"])
        args.n_values = [row["n"] for row in rows]
    else:
        print("-- empirical size by spot-volatility bandwidth exponent gamma (no jumps) --",
              flush=True)
        rows = []
        for n in args.n_values:
            t0 = time.time()
            oracle, full = size_by_bandwidth(n, args.n_paths, heston, rng, args.gammas, args.level)

            row = {"n": n}
            row.update({f"oracle_{test}": oracle[test] for test in TESTS})
            for gamma in args.gammas:
                row[f"k_n_{gamma}"] = window_size(n, gamma)
                row.update({f"{test}_{gamma}": full[gamma][test] for test in TESTS})
            rows.append(row)

            print(f"  n={n:>8}  ({time.time() - t0:.1f}s)", flush=True)
            for test in TESTS:
                sizes = "  ".join(f"g={g}: {full[g][test]:.4f}" for g in args.gammas)
                print(f"    {test:<8} oracle={oracle[test]:.4f}   {sizes}", flush=True)

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_size_by_bandwidth(
        rows, args.n_values, args.gammas, args.n_paths, args.level
    ),
        title="Empirical size by spot-volatility bandwidth exponent",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "gammas": args.gammas,
            "n_paths": args.n_paths,
            "level": args.level,
            "seed": args.seed,
            "heston": heston,
        },
        data={"bandwidth": columnar(rows)},
    )


if __name__ == "__main__":
    main()
