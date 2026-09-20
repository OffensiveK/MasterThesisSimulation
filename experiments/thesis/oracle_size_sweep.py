
# =========================================================================
# IMPORTS
# =========================================================================

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import N_GRID, Outputs, parser
from runtime.plotting import (COLOURS, LABELS, LINESTYLES, plain_line_legend, reference_line,
                              tilt_x_ticks, use_style)
from runtime.report import columnar, rows_from_columnar
from simulation.evt import TESTS
from simulation.monte_carlo import oracle_rejection_rates, rate_error

use_style()

# =========================================================================
# SIMULATION
# =========================================================================

def oracle_sweep(n_values, n_paths, level, seed, verbose=True):
    rng = np.random.default_rng(seed)
    results = []
    for n in n_values:
        t0 = time.time()
        rates = oracle_rejection_rates(n, n_paths, rng, level=level)
        results.append({"n": n, **rates})
        if verbose:
            parts = "  ".join(f"{test}={rates[test]:.4f}" for test in TESTS)
            print(f"  n={n:>8}  {parts}  ({time.time() - t0:.1f}s)", flush=True)
    return results


# =========================================================================
# PLOTTING
# =========================================================================

def plot_rejection_rates(results, title, reference_level=None, n_paths=None):
    fig, axes = plt.subplots(1, len(TESTS), figsize=(4.0 * len(TESTS), 4.0), sharey=True)
    n_values = [r["n"] for r in results]
    for ax, test in zip(axes, TESTS):
        rates = np.array([r[test] for r in results], dtype=float)
        # No series label, since the panel title already names the test and a one-entry legend
        # repeating it would say nothing. Only the reference line needs the key.
        ax.errorbar(n_values, rates, yerr=rate_error(rates, n_paths) if n_paths else None,
                    linestyle=LINESTYLES[test], color=COLOURS[test],
                    linewidth=1.0, elinewidth=1.0)
        if reference_level is not None:
            reference_line(ax, reference_level, label="nominal level")
        ax.set_xscale("log")
        ax.set_xlabel("n")
        ax.set_title(LABELS[test])
        tilt_x_ticks(ax)
    axes[0].set_ylabel("rejection rate")
    plain_line_legend(axes[-1], fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/oracle_size_sweep",
               n_values=N_GRID, n_paths=25000, level=0.05, seed=0)
    args = p.parse_args()
    out = Outputs(args, "size")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    if out.replotting:
        print(f"-- replotting from {out.json} --", flush=True)
        size_results = rows_from_columnar(json.loads(out.json.read_text())["size"])
    else:
        print("-- oracle size (i.i.d. N(0,1), no Heston/jumps/spot-vol estimation) --", flush=True)
        size_results = oracle_sweep(args.n_values, args.n_paths, args.level, args.seed)

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_rejection_rates(
        size_results, "Empirical size, oracle N(0,1) vectors", reference_level=args.level,
        n_paths=args.n_paths,
    ),
        title="Empirical size across n, oracle N(0,1) vectors",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "n_paths": args.n_paths,
            "level": args.level,
            "seed": args.seed,
        },
        data={"size": columnar(size_results)},
    )


if __name__ == "__main__":
    main()
