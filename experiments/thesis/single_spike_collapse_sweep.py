
# =========================================================================
# IMPORTS
# =========================================================================

import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import N_GRID, Outputs, parser
from runtime.plotting import COLOURS, LABELS, LINESTYLES, reference_line, use_style
from runtime.report import grid_key, n_from_grid_key
from simulation.continuous import HestonParams
from simulation.evt import TESTS
from simulation.monte_carlo import rate_error, rates, statistics_sharing_paths
from simulation.spikes import critical_jump_scale, naive_jump_scale, signed_spike_pattern

use_style()

ANCHOR_N = 200
BAND_COLOURS = ("#1f3b73", "#5b9bd5", "#c27ba0", "#c0392b")

# =========================================================================
# SIMULATION
# =========================================================================

def spike_size(heston, n, rho, anchor=ANCHOR_N):
    return critical_jump_scale(heston, anchor) * (n / anchor) ** (-rho)


def studentized_value(heston, n, rho, anchor=ANCHOR_N):
    return spike_size(heston, n, rho, anchor) / naive_jump_scale(heston, n)


def sweep_exponents(n, heston, exponents, n_paths, rng, level, location=0.5, sign=1.0):
    patterns = {"null": None}
    for rho in exponents:
        patterns[f"{rho:g}"] = signed_spike_pattern(n, spike_size(heston, n, rho), location, sign)
    stats = statistics_sharing_paths(n, heston, patterns, n_paths, rng)

    block = {}
    for rho in exponents:
        block[f"{rho:g}"] = {
            "rho": rho,
            "eta": spike_size(heston, n, rho),
            "m": studentized_value(heston, n, rho),
            **rates(stats["null"], stats[f"{rho:g}"], level),
        }
    return block


# =========================================================================
# PLOTTING
# =========================================================================

def plot_single_spike_collapse(data, n_paths=None, level=0.05):
    order = sorted(data, key=n_from_grid_key)
    n_values = [n_from_grid_key(key) for key in order]
    exponents = sorted({r for key in data for r in data[key]}, key=float)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for index, rho in enumerate(exponents):
        colour = BAND_COLOURS[index % len(BAND_COLOURS)]
        for test in TESTS:
            y = np.array([data[key][rho][test]["power_adj"] for key in order])
            left.errorbar(n_values, y, yerr=rate_error(y, n_paths), linestyle=LINESTYLES[test],
                          color=colour, linewidth=1.1, elinewidth=0.9)
            
        left.annotate(rf"$\varrho = {float(rho):g}$", (n_values[-1], y[-1]),
                      textcoords="offset points", xytext=(6, 0), color=colour,
                      fontsize=8, va="center")
    for test in TESTS:
        left.plot([], [], linestyle=LINESTYLES[test], color="gray", linewidth=1.1,
                  label=LABELS[test])
    reference_line(left, level)
    left.set_xscale("log")
    left.set_ylim(0, 1.05)
    left.set_xlim(right=n_values[-1] * 2.6)
    left.set_xlabel("$n$")
    left.set_ylabel("size-adjusted power")
    left.set_title("All three tests, size-adjusted")
    left.legend(loc="center left", fontsize=8)

    total = 0
    for test in TESTS:
        m = np.array([data[key][rho]["m"] for key in order for rho in exponents])
        y = np.array([data[key][rho][test]["power_adj"] for key in order for rho in exponents])
        right.errorbar(m, y, yerr=rate_error(y, n_paths), linestyle="none",
                       color=COLOURS[test], elinewidth=1.0, label=LABELS[test])
        total += y.size
    right.set_ylim(0, 1.05)
    right.set_xlabel("studentized spike value $m$")
    right.set_ylabel("size-adjusted power")
    right.set_title(f"All {total} configurations against $m$ alone")
    right.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/single_spike_collapse_sweep",
               n_values=N_GRID, n_paths=12000, level=0.05, seed=7)
    p.add_argument("--exponents", type=float, nargs="+", default=[0.3, 0.45, 0.5, 0.7])
    p.add_argument("--location", type=float, default=0.5)
    p.add_argument("--negative", action="store_true")
    args = p.parse_args()
    out = Outputs(args, "collapse")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams()
    if out.replotting:
        data = out.saved_data()
    else:
        rng = np.random.default_rng(args.seed)
        sign = -1.0 if args.negative else 1.0
        data = {}
        for n in args.n_values:
            t0 = time.time()
            block = sweep_exponents(n, heston, args.exponents, args.n_paths, rng, args.level,
                                    location=args.location, sign=sign)
            data[grid_key(n)] = block
            print(f"-- n={n}  ({time.time() - t0:.1f}s) --", flush=True)
            for rho in args.exponents:
                row = block[f"{rho:g}"]
                parts = "  ".join(
                    f"{t}: {row[t]['power']:.3f}/{row[t]['power_adj']:.3f}" for t in TESTS
                )
                print(f"  rho={rho:<5g} eta={row['eta']:.6f} m={row['m']:.3f}  (raw/adj)  {parts}",
                      flush=True)

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_single_spike_collapse(data, n_paths=args.n_paths, level=args.level),
        title="Power against a spike shrinking at n^-rho, across a family of exponents",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "exponents": args.exponents,
            "anchor_n": ANCHOR_N,
            "location": args.location,
            "n_paths": args.n_paths,
            "level": args.level,
            "seed": args.seed,
            "negative": args.negative,
            "heston": heston,
        },
        data=data,
    )


if __name__ == "__main__":
    main()
