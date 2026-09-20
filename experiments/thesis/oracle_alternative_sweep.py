
# =========================================================================
# IMPORTS
# =========================================================================

import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import Outputs, parser
from runtime.plotting import COLOURS, LABELS, MARKERS, label_ticks, tilt_x_ticks, use_style
from runtime.report import grid_key, n_from_grid_key
from simulation.batching import batch_size_for, chunks
from simulation.continuous import HestonParams, simulate_continuous_increments_and_variance_batch
from simulation.estimators import (
    spot_volatility_batch,
    studentize_with,
    truncation_threshold,
    window_size,
)
from simulation.evt import STATISTIC_BATCH, TESTS, empirical_critical
from simulation.jumps import grid_index, jump_increments
from simulation.spikes import critical_jump_scale, signed_spike_pattern

use_style()

ARMS = ("oracle", "plugin")

# =========================================================================
# SIMULATION
# =========================================================================

def statistics_sharing_paths(n, heston, patterns, n_paths, rng, spike_index, k_n_exponent=0.5):
    batch_size = batch_size_for(n, n_paths, arrays_per_path=12)
    k_n = window_size(n, k_n_exponent)
    u_n = truncation_threshold(n)

    collected = {key: {arm: {test: [] for test in TESTS} for arm in ARMS} for key in patterns}
    
    spike_z = {key: {arm: [] for arm in ARMS} for key in patterns}
    for take in chunks(n_paths, batch_size):
        continuous, true_variance = simulate_continuous_increments_and_variance_batch(
            take, n, heston, rng
        )
        for key, pattern in patterns.items():
            increments = (continuous if pattern is None
                          else continuous + jump_increments(pattern)[:, None])

            z = {
                "oracle": studentize_with(increments, true_variance),
                "plugin": studentize_with(increments, spot_volatility_batch(increments, k_n, u_n)),
            }
            for arm in ARMS:
                spike_z[key][arm].append(z[arm][spike_index])
                for test in TESTS:
                    collected[key][arm][test].append(STATISTIC_BATCH[test](z[arm]))
    stats = {key: {arm: {test: np.concatenate(v) for test, v in d.items()}
                   for arm, d in arms.items()}
             for key, arms in collected.items()}
    spikes = {key: {arm: np.concatenate(v) for arm, v in arms.items()}
              for key, arms in spike_z.items()}
    return stats, spikes


def rates(null_stats, alt_stats, level=0.05):
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for test in TESTS:
            threshold = empirical_critical(null_stats[arm][test], level)
            out[arm][test] = {
                "power_adj": float((alt_stats[arm][test] > threshold).mean()),
                "threshold_adj": threshold,
            }
    return out


def sweep_sizes(n, heston, multipliers, n_paths, rng, level, location=0.5):
    scale = critical_jump_scale(heston, n)
    spike_index = grid_index(n, location)
    patterns = {"null": None}
    for m in multipliers:
        patterns[f"{m:g}"] = signed_spike_pattern(n, m * scale, location, 1.0)
    stats, spikes = statistics_sharing_paths(n, heston, patterns, n_paths, rng, spike_index)

    block = {}
    for m in multipliers:
        key = f"{m:g}"
        row = rates(stats["null"], stats[key], level)
        for arm in ARMS:
            row[arm]["spike_z_median"] = float(np.median(spikes[key][arm]))
        block[key] = {"multiplier": m, "eta": m * scale, **row}
    return scale, block


# =========================================================================
# PLOTTING
# =========================================================================

def plot_oracle_alternative(data):
    fig, axes = plt.subplots(1, len(data), figsize=(5.0 * len(data), 4.0), sharey=True,
                             squeeze=False)
    for ax, (key, block) in zip(axes.ravel(), data.items()):
        multipliers = [block[k]["multiplier"] for k in block]
        for test in TESTS:
            gap = [100.0 * (block[k]["oracle"][test]["power_adj"]
                            - block[k]["plugin"][test]["power_adj"]) for k in block]
            ax.plot(multipliers, gap, marker=MARKERS[test], markersize=4,
                    color=COLOURS[test], label=LABELS[test])
        ax.axhline(0.0, color="0.6", linewidth=1.0)
        ax.set_xscale("log")
        label_ticks(ax, multipliers, {0.25, 0.5, 1.0, 2.0, 3.0}, fmt="{:g}x")
        tilt_x_ticks(ax)
        ax.set_xlabel("spike size / critical scale")
        ax.set_title(f"n = {n_from_grid_key(key)}")
    axes.ravel()[0].set_ylabel("oracle minus plug-in (pts)")
    axes.ravel()[0].legend(loc="upper right")
    fig.suptitle("Detection cost of the plug-in error, both arms held to an exact level")
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/oracle_alternative_sweep",
               n_values=[10000, 100000], n_paths=12000, level=0.05, seed=23)
    p.add_argument("--location", type=float, default=0.5)
    p.add_argument("--multipliers", type=float, nargs="+",
                   default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
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
        data, scales = {}, {}
        for n in args.n_values:
            t0 = time.time()
            scale, block = sweep_sizes(n, heston, args.multipliers, args.n_paths, rng,
                                       args.level, location=args.location)
            data[grid_key(n)], scales[n] = block, scale
            print(f"-- n={n}  (critical scale={scale:.6f}, {time.time() - t0:.1f}s) --", flush=True)
            for m in args.multipliers:
                row = block[f"{m:g}"]
                parts = "  ".join(
                    f"{t}: {row['oracle'][t]['power_adj']:.3f}/{row['plugin'][t]['power_adj']:.3f}"
                    for t in TESTS
                )
                zo = row["oracle"]["spike_z_median"]
                zp = row["plugin"]["spike_z_median"]
                print(f"  m={m:<5g} (oracle/plugin)  {parts}   spike z {zo:.2f}/{zp:.2f}",
                      flush=True)

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_oracle_alternative(data),
        title="Plug-in cost under the alternative, oracle against feasible",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "location": args.location,
            "multipliers": args.multipliers,
            "n_paths": args.n_paths,
            "level": args.level,
            "seed": args.seed,
            "heston": heston,
            "critical_scales": scales,
        },
        data=data,
    )


if __name__ == "__main__":
    main()
