
# =========================================================================
# IMPORTS
# =========================================================================

import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import Outputs, parser
from runtime.plotting import COLOURS, LABELS, MARKERS, reference_line, use_style
from runtime.report import grid_key, n_from_grid_key
from simulation.continuous import HestonParams
from simulation.estimators import window_size
from simulation.evt import TESTS
from simulation.jumps import grid_index
from simulation.monte_carlo import rates, statistics_sharing_paths
from simulation.spikes import critical_jump_scale, signed_spike_pattern

use_style()

SIGNS = ((1.0, "pos"), (-1.0, "neg"))

# =========================================================================
# SIMULATION
# =========================================================================

# =========================================================================
# PLOTTING
# =========================================================================

def plot_spike_location(data):
    fig, axes = plt.subplots(1, len(data), figsize=(5.0 * len(data), 4.0), sharey=True,
                             squeeze=False)
    for ax, (key, block) in zip(axes.ravel(), data.items()):
        n = n_from_grid_key(key)
        points = block["points"]

        # Binomial intervals are deliberately not drawn. 
        keys = [k for k in points if k.startswith("pos:")]
        # The first location is 0, which a log axis cannot place, nudge it 
        locations = [max(points[k]["location"], 5e-4) for k in keys]
        for test in TESTS:
            ax.plot(locations, [points[k][test]["power_adj"] for k in keys],
                    marker=MARKERS[test], markersize=4, color=COLOURS[test],
                    label=LABELS[test])

        reference_line(ax, block["k_n"] / n, axis="x", color="black", dotted=True)
        ax.set_xscale("log")
        ax.set_xlabel("spike location $t$")
        ax.set_title(f"n = {n}   (forward-window region $t < k_n/n$)")
        ax.set_ylim(0.15, 0.55)

    axes.ravel()[0].set_ylabel("size-adjusted power")
    axes.ravel()[0].legend(loc="lower right")
    fig.suptitle("Single-spike power against the spike's location")
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/spike_location_sweep",
               n_values=[1916, 10437], n_paths=12000, level=0.05, seed=17)
    p.add_argument("--locations", type=float, nargs="+",
                   default=[0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 0.999])
    p.add_argument("--multiplier", type=float, default=1.0)
    p.add_argument("--xi", type=float, default=HestonParams().xi)
    args = p.parse_args()
    out = Outputs(args, "power")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams(xi=args.xi)
    rng = np.random.default_rng(args.seed)

    if out.replotting:
        data = out.saved_data()
    else:
        data = {}
        for n in args.n_values:
            scale = critical_jump_scale(heston, n)
            spike_size = args.multiplier * scale
            k_n = window_size(n)

            patterns = {"null": None}
            for sign, label in SIGNS:
                for loc in args.locations:
                    patterns[f"{label}:{loc:g}"] = signed_spike_pattern(n, spike_size, loc, sign)

            t0 = time.time()
            stats = statistics_sharing_paths(n, heston, patterns, args.n_paths, rng)
            print(f"-- n={n}  spike {args.multiplier:g}x crit = {spike_size:.5f}, k_n={k_n} "
                  f"(forward-window region is t < {k_n/n:.4f}) ({time.time() - t0:.1f}s) --",
                  flush=True)

            block = {}
            for sign, label in SIGNS:
                for loc in args.locations:
                    key = f"{label}:{loc:g}"
                    row = rates(stats["null"], stats[key], args.level)
                    block[key] = {"location": loc, "sign": sign,
                                  "grid_index": grid_index(n, loc), **row}
                    parts = "  ".join(f"{t}: {row[t]['power']:.3f}/{row[t]['power_adj']:.3f}"
                                      for t in TESTS)
                    print(f"  {label} t={loc:<6g} (j={block[key]['grid_index']:>6})  (power/adj)  "
                          f"{parts}", flush=True)
            data[grid_key(n)] = {"k_n": k_n, "points": block}

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_spike_location(data),
        title="Single-spike power against spike location and sign",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "locations": args.locations,
            "multiplier": args.multiplier,
            "n_paths": args.n_paths,
            "level": args.level,
            "seed": args.seed,
            "heston": heston,
        },
        data=data,
    )


if __name__ == "__main__":
    main()
