
# =========================================================================
# IMPORTS
# =========================================================================

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import Outputs, parser
from runtime.plotting import LABELS, plain_log_ticks, reference_line, use_style
from runtime.report import rows_from_columnar
from simulation.evt import TESTS
from simulation.monte_carlo import rate_error

use_style()

LEVEL = 0.05
CMAP_SPAN = 0.85
SIZE_TICKS = (0.05, 0.1, 0.2, 0.5, 1.0)
GAMMA_STAR = 0.5

SIZE_SOURCE = Path("figures/thesis/bandwidth_sensitivity_sweep/bandwidth.json")
POWER_SOURCE = Path("figures/thesis/bandwidth_power_sweep/power.json")

# =========================================================================
# SIMULATION
# =========================================================================

def gammas_in(rows):
    keys = [k for k in rows[0] if k.startswith("k_n_")]
    return sorted(float(k[len("k_n_"):]) for k in keys)


def load_sweeps():
    size = rows_from_columnar(json.loads(SIZE_SOURCE.read_text())["bandwidth"])
    power = rows_from_columnar(json.loads(POWER_SOURCE.read_text())["bandwidth_power"])
    size_gammas, power_gammas = gammas_in(size), gammas_in(power)
    if size_gammas != power_gammas:
        raise ValueError("the two sweeps no longer share a bandwidth grid")
    power_n = [r["n"] for r in power]
    size_n = [r["n"] for r in size]
    if not set(power_n) <= set(size_n):
        raise ValueError("the power sweep runs an n the size sweep does not, so colours cannot pair")
    return size, power, size_gammas


def colours_by_n(size_n):
    # Keyed on the n itself rather than on position, so a given n keeps one colour in both rows.
    # Indexing per row would hand n=619 two different colours, the rows carrying different n sets.
    cmap = plt.get_cmap("viridis")
    span = max(1, len(size_n) - 1)
    return {n: cmap(CMAP_SPAN * i / span) for i, n in enumerate(size_n)}


# =========================================================================
# PLOTTING
# =========================================================================

def plot_bandwidth_comparison(size, power, gammas, n_paths):
    x = np.array(gammas, dtype=float)
    size_n = [r["n"] for r in size]
    colour = colours_by_n(size_n)

    fig, axes = plt.subplots(2, len(TESTS), figsize=(4.0 * len(TESTS), 7.2),
                             sharex=True, sharey="row")
    for column, test in enumerate(TESTS):
        top, bottom = axes[0][column], axes[1][column]

        for row in size:
            y = np.array([row[f"{test}_{g}"] for g in gammas])
            top.errorbar(x, y, yerr=rate_error(y, n_paths), color=colour[row["n"]],
                         label=f"$n$={row['n']}", linewidth=1.2, elinewidth=1.0)
        reference_line(top, LEVEL, dotted=True, color="black")
        reference_line(top, GAMMA_STAR, axis="x", color="black")
        top.set_yscale("log")
        plain_log_ticks(top, SIZE_TICKS)
        top.set_title(LABELS[test])

        for row in power:
            y = np.array([row[f"{test}_{g}_power_adj"] for g in gammas])
            bottom.errorbar(x, y, yerr=rate_error(y, n_paths), color=colour[row["n"]],
                            linewidth=1.2, elinewidth=1.0)
        reference_line(bottom, GAMMA_STAR, axis="x", color="black")
        bottom.set_ylim(0, 1.02)
        bottom.set_xlabel(r"bandwidth exponent $\gamma$")

    axes[0][0].set_ylabel("empirical size")
    axes[1][0].set_ylabel("size-adjusted power")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/bandwidth_comparison", n_paths=6000, from_saved=False,
               level=None, seed=None)
    args = p.parse_args()
    out = Outputs(args, "comparison")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    size, power, gammas = load_sweeps()

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_bandwidth_comparison(size, power, gammas, args.n_paths),
        title="Empirical size and size-adjusted power by spot-volatility bandwidth exponent",
        doc=__doc__,
        script=__file__,
        params={
            "n_paths": args.n_paths,
            "level": LEVEL,
            "gammas": gammas,
            "size_n": [r["n"] for r in size],
            "power_n": [r["n"] for r in power],
            "sources": [str(SIZE_SOURCE), str(POWER_SOURCE)],
        },
    )


if __name__ == "__main__":
    main()
