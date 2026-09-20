
# =========================================================================
# IMPORTS
# =========================================================================

import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import Outputs, parser
from runtime.plotting import COLOURS, LABELS, LINESTYLES, use_style
from runtime.report import grid_key, n_from_grid_key
from simulation.continuous import HestonParams
from simulation.evt import TESTS
from simulation.monte_carlo import rate_error, rates, statistics_sharing_paths
from simulation.spikes import critical_jump_scale, two_spike_pattern

use_style()

SIGN_MODES = ((True, "same"), (False, "opp"))
SIGN_TITLES = {"same": "same sign", "opp": "opposite signs"}

# =========================================================================
# SIMULATION
# =========================================================================

# =========================================================================
# PLOTTING
# =========================================================================

def plot_two_spike(data, n_paths=None):
    # Shorter rows than the panels' natural aspect would give, since the curves occupy a narrow
    # band of the axis and the figure has to share a page with the prose that reads it.
    fig, axes = plt.subplots(len(data), 2, figsize=(5.4 * 2, 3.1 * len(data)), sharey=True,
                             sharex=True, squeeze=False)
    for row, (key, block) in enumerate(data.items()):
        n = n_from_grid_key(key)
        for col, (_, tag) in enumerate(SIGN_MODES):
            ax = axes[row][col]
            keys = [k for k in block if k.startswith(tag + ":")]
            ratios = [block[k]["ratio"] for k in keys]
            for test in TESTS:
                adjusted = np.array([block[k][test]["power_adj"] for k in keys])
                # Line style rather than a marker, as in Figure 2: the markers sat on top
                # of the error bars and hid them, and the styles read in black and white.
                ax.errorbar(ratios, adjusted, yerr=rate_error(adjusted, n_paths),
                            linestyle=LINESTYLES[test], color=COLOURS[test], label=LABELS[test],
                            linewidth=1.2, elinewidth=1.0)
            if row == len(data) - 1:
                ax.set_xlabel("second spike / first spike")
            ax.set_title(f"n = {n}, {SIGN_TITLES[tag]}")
            # The curves live between 0.64 and 0.94, so the old floor at 0.55 was empty axis.
            ax.set_ylim(0.60, 0.95)
            if col == 0:
                ax.set_ylabel("size-adjusted power")
    axes[0][0].legend(loc="lower left")
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/two_spike_sweep",
               n_values=[1916, 10437], n_paths=12000, level=0.05, seed=11)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    p.add_argument("--first-multiplier", type=float, default=1.5)
    p.add_argument("--separation", type=float, default=0.2)
    args = p.parse_args()
    out = Outputs(args, "power")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams()
    rng = np.random.default_rng(args.seed)

    if out.replotting:
        data = out.saved_data()
    else:
        data = {}
        for n in args.n_values:
            scale = critical_jump_scale(heston, n)
            spike_size = args.first_multiplier * scale
            separation = int(args.separation * n)

            patterns = {"null": None}
            for same_sign, label in SIGN_MODES:
                for ratio in args.ratios:
                    patterns[f"{label}:{ratio:g}"] = two_spike_pattern(
                        n, spike_size, ratio, separation, same_sign=same_sign
                    )

            t0 = time.time()
            stats = statistics_sharing_paths(n, heston, patterns, args.n_paths, rng)
            print(f"-- n={n}  first spike {args.first_multiplier:g}x crit = {spike_size:.5f}, "
                  f"separation {separation} ({time.time() - t0:.1f}s) --", flush=True)

            block = {}
            for same_sign, label in SIGN_MODES:
                for ratio in args.ratios:
                    key = f"{label}:{ratio:g}"
                    row = rates(stats["null"], stats[key], args.level)
                    block[key] = {"ratio": ratio, "same_sign": same_sign, **row}
                    parts = "  ".join(f"{t}: {row[t]['power']:.3f}/{row[t]['power_adj']:.3f}"
                                      for t in TESTS)
                    print(f"  {label} ratio={ratio:<5g}  (power/adj)  {parts}", flush=True)
            data[grid_key(n)] = block

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_two_spike(data, n_paths=args.n_paths),
        title="Two spikes of comparable size",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "ratios": args.ratios,
            "first_multiplier": args.first_multiplier,
            "separation": args.separation,
            "n_paths": args.n_paths,
            "level": args.level,
            "seed": args.seed,
            "heston": heston,
        },
        data=data,
    )


if __name__ == "__main__":
    main()
