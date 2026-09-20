
# =========================================================================
# IMPORTS
# =========================================================================

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import Outputs, parser
from runtime.plotting import LABELS, reference_line, use_style
from runtime.report import columnar, rows_from_columnar
from simulation.batching import batch_size_for, chunks
from simulation.continuous import HestonParams
from simulation.estimators import studentize_batch, window_size
from simulation.evt import STATISTIC_BATCH, TESTS, empirical_critical
from simulation.jumps import JumpParams, NormalJumpSizes
from simulation.model import simulate_log_price_increments_batch
from simulation.monte_carlo import rate_error

use_style()

CMAP_SPAN = 0.85

# =========================================================================
# SIMULATION
# =========================================================================

def collect_statistics(n, n_paths, heston, jumps, rng, gammas):
    batch_size = batch_size_for(n, n_paths, arrays_per_path=4 + len(gammas))
    out = {gamma: {test: [] for test in TESTS} for gamma in gammas}
    for chunk in chunks(n_paths, batch_size):
        increments = simulate_log_price_increments_batch(chunk, n, heston, jumps, rng)
        for gamma in gammas:
            z_hat = studentize_batch(increments, gamma=gamma)
            for test in TESTS:
                out[gamma][test].append(STATISTIC_BATCH[test](z_hat))
    return {g: {t: np.concatenate(v) for t, v in d.items()} for g, d in out.items()}


# =========================================================================
# PLOTTING
# =========================================================================

def plot_bandwidth_power(rows, n_values, gammas, n_paths):
    x = np.array(gammas, dtype=float)

    fig, axes = plt.subplots(1, len(TESTS), figsize=(4.0 * len(TESTS), 4.0), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, test in zip(axes, TESTS):
        for i, n in enumerate(n_values):
            y = np.array([rows[i][f"{test}_{g}_power_adj"] for g in gammas])
            ax.errorbar(x, y, yerr=rate_error(y, n_paths), marker="o", markersize=4,
                        color=cmap(CMAP_SPAN * i / max(1, len(n_values) - 1)),
                        label=f"$n$={n}", linewidth=1.2, elinewidth=1.0)
        reference_line(ax, 0.5, axis="x", color="black")
        ax.set_xlabel(r"bandwidth exponent $\gamma$")
        ax.set_title(LABELS[test])
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("size-adjusted power")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Detection power at a common exact size, against the bandwidth")
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/bandwidth_power_sweep",
               n_values=[200, 619, 1916, 5932, 18362], n_paths=6000, level=0.05, seed=1000)
    p.add_argument("--gammas", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    args = p.parse_args()
    out = Outputs(args, "power")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams()
    jump_params = JumpParams(
        intensity=5.0, size_sampler=NormalJumpSizes(0.0, 0.03),
        min_jumps=1, min_peak_magnitude=0.03,
    )
    rng = np.random.default_rng(args.seed)

    if out.replotting:
        print(f"-- replotting from {out.json} --", flush=True)
        rows = rows_from_columnar(json.loads(out.json.read_text())["bandwidth_power"])
        args.n_values = [row["n"] for row in rows]
    else:
        print("-- size-adjusted power by spot-volatility bandwidth exponent gamma --", flush=True)
        rows = []
        for n in args.n_values:
            t0 = time.time()
            null = collect_statistics(n, args.n_paths, heston, None, rng, args.gammas)
            alt = collect_statistics(n, args.n_paths, heston, jump_params, rng, args.gammas)

            row = {"n": n}
            for gamma in args.gammas:
                row[f"k_n_{gamma}"] = window_size(n, gamma)
                for test in TESTS:
                    threshold = empirical_critical(null[gamma][test], args.level)
                    row[f"{test}_{gamma}_threshold"] = threshold
                    row[f"{test}_{gamma}_power_adj"] = float((alt[gamma][test] > threshold).mean())
            rows.append(row)

            print(f"  n={n:>8}  ({time.time() - t0:.1f}s)", flush=True)
            for test in TESTS:
                vals = "  ".join(f"g={g}: {row[f'{test}_{g}_power_adj']:.4f}" for g in args.gammas)
                print(f"    {test:<8} size-adjusted power   {vals}", flush=True)

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_bandwidth_power(rows, args.n_values, args.gammas, args.n_paths),
        title="Size-adjusted power by spot-volatility bandwidth exponent",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "gammas": args.gammas,
            "n_paths": args.n_paths,
            "level": args.level,
            "seed": args.seed,
            "heston": heston,
            "jumps": jump_params,
        },
        data={"bandwidth_power": columnar(rows)},
    )


if __name__ == "__main__":
    main()
