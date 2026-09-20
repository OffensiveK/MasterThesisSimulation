
# =========================================================================
# IMPORTS
# =========================================================================

import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import Outputs, parser
from runtime.plotting import (
    COLOURS,
    LABELS,
    MARKERS,
    label_ticks,
    reference_line,
    tilt_x_ticks,
    use_style,
)
from runtime.report import grid_key, n_from_grid_key
from simulation.continuous import HestonParams
from simulation.evt import TESTS
from simulation.monte_carlo import rates, statistics_sharing_paths
from simulation.spikes import critical_jump_scale, signed_spike_pattern

use_style()

# =========================================================================
# SIMULATION
# =========================================================================

def sweep_sizes(n, heston, multipliers, n_paths, rng, level, location=0.5, sign=1.0):
    scale = critical_jump_scale(heston, n)
    patterns = {"null": None}
    for m in multipliers:
        patterns[f"{m:g}"] = signed_spike_pattern(n, m * scale, location, sign)
    stats = statistics_sharing_paths(n, heston, patterns, n_paths, rng)

    block = {}
    for m in multipliers:
        row = rates(stats["null"], stats[f"{m:g}"], level)
        block[f"{m:g}"] = {"multiplier": m, "size_eta": m * scale, **row}
    return scale, block


# =========================================================================
# PLOTTING
# =========================================================================

def plot_single_spike_power(data, level=0.05, location=0.5):
    fig, axes = plt.subplots(1, len(data), figsize=(5.0 * len(data), 4.2), sharey=True,
                             squeeze=False)
    for ax, (key, block) in zip(axes.ravel(), data.items()):
        n = n_from_grid_key(key)
        multipliers = [block[k]["multiplier"] for k in block]

        for test in TESTS:
            ax.plot(multipliers, [block[k][test]["power_adj"] for k in block],
                    marker=MARKERS[test], color=COLOURS[test],
                    label=f"{LABELS[test]}, adjusted")
            ax.plot(multipliers, [block[k][test]["power"] for k in block],
                    color=COLOURS[test], linestyle=":", linewidth=1.1, alpha=0.75)

        reference_line(ax, 1.0, axis="x", color="crimson")
        reference_line(ax, level)
        ax.set_xscale("log")
 
        label_ticks(ax, multipliers, {0.25, 0.5, 1.0, 2.0, 3.0}, fmt="{:g}×")
        tilt_x_ticks(ax)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("spike size / critical scale")
        ax.set_title(f"n = {n}")

    first = axes.ravel()[0]
    first.set_ylabel("rejection rate")
    first.plot([], [], color="gray", linestyle=":", label="raw (dotted)")
    first.legend(loc="upper left")
    fig.suptitle(f"Single-spike power at t={location:g}, raw against size-adjusted")
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/single_spike_power_sweep",
               n_values=[10000, 100000], n_paths=12000, seed=7)
    p.add_argument("--location", type=float, default=0.5)
    p.add_argument("--multipliers", type=float, nargs="+",
                   default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
    p.add_argument("--negative", action="store_true")
    args = p.parse_args()
    out = Outputs(args, "power")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams()
    if out.replotting:
        data = out.saved_data()
    else:
        rng = np.random.default_rng(args.seed)
        sign = -1.0 if args.negative else 1.0
        data, scales = {}, {}
        for n in args.n_values:
            t0 = time.time()
            scale, block = sweep_sizes(n, heston, args.multipliers, args.n_paths, rng, args.level,
                                       location=args.location, sign=sign)
            data[grid_key(n)], scales[n] = block, scale
            print(f"-- n={n}  (critical scale={scale:.6f}, {time.time() - t0:.1f}s) --",
                  flush=True)
            for m in args.multipliers:
                row = block[f"{m:g}"]
                parts = "  ".join(
                    f"{t}: {row[t]['size']:.3f}/{row[t]['power']:.3f}/{row[t]['power_adj']:.3f}"
                    for t in TESTS
                )
                print(f"  m={m:<5g} eta={row['size_eta']:.5f}  (size/power/adj)  {parts}",
                      flush=True)

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_single_spike_power(data, level=args.level, location=args.location),
        title="Power against a single spike's size, across n",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "location": args.location,
            "multipliers": args.multipliers,
            "n_paths": args.n_paths,
            "level": args.level,
            "seed": args.seed,
            "negative": args.negative,
            "heston": heston,
            "critical_scales": scales,
        },
        data=data,
    )


if __name__ == "__main__":
    main()
