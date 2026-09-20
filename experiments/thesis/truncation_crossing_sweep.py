
# =========================================================================
# IMPORTS
# =========================================================================

import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import Outputs, parser
from runtime.plotting import label_ticks, reference_line, use_style
from runtime.report import grid_key, n_from_grid_key
from simulation.batching import batch_size_for, chunks
from simulation.continuous import HestonParams, simulate_continuous_increments_batch
from simulation.estimators import (
    spot_volatility_batch,
    studentize_batch,
    truncation_threshold,
    window_size,
)
from simulation.evt import STATISTIC_BATCH, TESTS, asymptotic_critical_values
from simulation.jumps import grid_index, jump_increments
from simulation.monte_carlo import bootstrap_median_ci
from simulation.spikes import signed_spike_pattern

use_style()

# =========================================================================
# SIMULATION
# =========================================================================

def crossing_for_n(n, ratios, n_paths, heston, rng, critical, n_boot=0, seed=0):
    u_n = truncation_threshold(n)
    k_n = window_size(n)
    index = grid_index(n, 0.5)
    # studentize uses a backward window W_j = {j - k_n, ..., j - 1} once j >= k_n, so the
    # windows that swallow the spike are those of the k_n increments that follow it.
    affected = slice(index + 1, min(index + 1 + k_n, n))

    batch_size = batch_size_for(n, n_paths, arrays_per_path=8)
    acc = {r: {"contamination": [], **{t: [] for t in TESTS}} for r in ratios}
    clean_paths = []
    for chunk in chunks(n_paths, batch_size):
        continuous = simulate_continuous_increments_batch(chunk, n, heston, rng)
        clean = spot_volatility_batch(continuous, k_n, u_n)[affected].mean(axis=0)
        clean_paths.append(clean)
        for ratio in ratios:
            pattern = signed_spike_pattern(n, ratio * u_n)
            increments = continuous + jump_increments(pattern)[:, None]
            dirty = spot_volatility_batch(increments, k_n, u_n)[affected].mean(axis=0)
            acc[ratio]["contamination"].append(dirty / clean)
            z_hat = studentize_batch(increments, k_n)
            for test in TESTS:
                acc[ratio][test].append(STATISTIC_BATCH[test](z_hat))

    block = {}
    for index_r, ratio in enumerate(ratios):
        per_path = np.concatenate(acc[ratio]["contamination"])
        contamination = float(np.median(per_path))
        row = {"ratio_to_u_n": ratio, "eta": ratio * u_n, "contamination": contamination}
        if n_boot:
            lo, hi = bootstrap_median_ci(per_path, n_boot,
                                         np.random.default_rng(seed + index_r))
            row["ci_lo"], row["ci_hi"] = lo, hi
        for test in TESTS:
            row[f"{test}_power"] = float(
                (np.concatenate(acc[ratio][test]) > critical[test]).mean()
            )
        block[f"{ratio:g}"] = row

    return u_n, k_n, float(np.median(np.concatenate(clean_paths))), block


# =========================================================================
# PLOTTING
# =========================================================================

def plot_truncation_crossing(data):
    fig, axes = plt.subplots(1, len(data), figsize=(5.0 * len(data), 4.0), sharey=True,
                             squeeze=False)
    for ax, (key, block) in zip(axes.ravel(), data.items()):
        n = n_from_grid_key(key)
        points = block["points"]
        ratios = [points[k]["ratio_to_u_n"] for k in points]
        centre = np.array([points[k]["contamination"] for k in points])
        # The point sitting exactly at u_n is a coin flip between truncating and not, so its
        # median falls in a low-density gap and its interval is the widest on the panel.
        yerr = None
        if all("ci_lo" in points[k] for k in points):
            yerr = np.vstack([centre - np.array([points[k]["ci_lo"] for k in points]),
                              np.array([points[k]["ci_hi"] for k in points]) - centre])
        # No marker: the interval at eta = u_n is the point worth seeing and a dot sits on top of
        # it, while the line's own vertices already mark where the sweep was measured.
        ax.errorbar(ratios, centre, yerr=yerr, color="C3", linewidth=1.2, elinewidth=1.0)
        reference_line(ax, 1.0, axis="x", color="black")
        reference_line(ax, 1.0, dotted=True)
        # Log x, linear y: the sweep spans more than a decade in spike size while the contamination
        # itself is a small multiple, and on a linear x axis the flat post-threshold tail would take
        # three quarters of the panel to say nothing.
        ax.set_xscale("log")
        label_ticks(ax, ratios, {0.1, 0.25, 0.5, 1.0, 2.0, 4.0}, fontsize=9)
        # A contamination factor floors at one, not at zero, so the axis starts just below it.
        ax.set_ylim(bottom=0.8)
        ax.set_xlabel(r"spike size / $u_n$")
        ax.set_title(f"n = {n}   ($u_n$ = {block['u_n']:.4f}, $k_n$ = {block['k_n']})")
    axes.ravel()[0].set_ylabel("contamination factor")
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/truncation_crossing_sweep",
               n_values=[5932, 18362], n_paths=6000, level=0.05, seed=13)
    p.add_argument("--ratios", type=float, nargs="+",
                   default=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.5, 2.0, 4.0])
    p.add_argument("--n-boot", type=int, default=400)
    args = p.parse_args()
    out = Outputs(args, "crossing")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams()
    rng = np.random.default_rng(args.seed)
    critical = asymptotic_critical_values(args.level)

    if out.replotting:
        data = out.saved_data()
    else:
        data = {}
        for n in args.n_values:
            t0 = time.time()
            u_n, k_n, median_clean, block = crossing_for_n(
                n, args.ratios, args.n_paths, heston, rng, critical,
                n_boot=args.n_boot, seed=args.seed + 1)
            data[grid_key(n)] = {"u_n": u_n, "k_n": k_n, "median_clean": median_clean,
                                 "points": block}

            print(f"-- n={n}  u_n={u_n:.5f}  k_n={k_n}  ({time.time() - t0:.1f}s) --", flush=True)
            for ratio in args.ratios:
                row = block[f"{ratio:g}"]
                powers = "  ".join(f"{t}: {row[f'{t}_power']:.3f}" for t in TESTS)
                print(f"  eta/u_n={ratio:<5g} eta={row['eta']:.5f}  "
                      f"contamination={row['contamination']:6.2f}x   power  {powers}", flush=True)

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_truncation_crossing(data),
        title="Spot-variance contamination as a spike crosses the truncation threshold",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "ratios": args.ratios,
            "n_paths": args.n_paths,
            "n_boot": args.n_boot,
            "level": args.level,
            "seed": args.seed,
            "heston": heston,
        },
        data=data,
    )


if __name__ == "__main__":
    main()
