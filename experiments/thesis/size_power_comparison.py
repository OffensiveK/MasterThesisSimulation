
# =========================================================================
# IMPORTS
# =========================================================================

import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import N_GRID, Outputs, parser
from runtime.plotting import (COLOURS, LABELS, LINESTYLES, plain_line_legend, plain_log_ticks,
                              reference_line, tilt_x_ticks, use_style)
from runtime.report import columnar, rows_from_columnar
from simulation.batching import batch_size_for, chunks
from simulation.continuous import HestonParams
from simulation.estimators import studentize_batch, window_size
from simulation.evt import REJECT, STATISTIC_BATCH, TESTS
from simulation.jumps import BimodalJumpSizes, JumpParams, NormalJumpSizes
from simulation.model import simulate_log_price_increments_batch
from simulation.monte_carlo import rate_error

use_style()

SIZE_TICKS = (0.05, 0.1, 0.2, 0.3, 0.5)

#: The two H1 regimes the figure sets side by side, in the order their panels appear.
REGIMES = (
    ("normal", r"jump sizes $\mathcal{N}(0, 0.03^2)$"),
    ("bimodal", r"jump sizes bimodal at $\pm 0.035$"),
)

# =========================================================================
# SIMULATION
# =========================================================================

def jump_law(args, regime):
    if regime == "normal":
        return NormalJumpSizes(args.jump_mean, args.jump_std), args.jump_std
    return BimodalJumpSizes(args.jump_mode, args.jump_spread), 0.0


def jump_params(args, regime):
    size_sampler, peak_floor = jump_law(args, regime)
    return JumpParams(
        intensity=args.intensity,
        size_sampler=size_sampler,
        min_jumps=args.min_jumps,
        min_peak_magnitude=peak_floor,
    )


def rejection_rates(n, heston, jumps, n_paths, rng, k_n_exponent=0.5, level=0.05):
    batch_size = batch_size_for(n, n_paths, arrays_per_path=6)
    k_n = window_size(n, k_n_exponent)

    rejections = {test: 0 for test in TESTS}
    for chunk in chunks(n_paths, batch_size):
        increments = simulate_log_price_increments_batch(chunk, n, heston, jumps, rng)
        z = studentize_batch(increments, k_n)
        for test in TESTS:
            rejections[test] += REJECT[test](STATISTIC_BATCH[test](z), level).sum()
    return {test: count / n_paths for test, count in rejections.items()}


def compare_across_n(n_values, heston, jumps, n_paths, level, seed):
    # Each arm reseeds, so the size arm and either power arm are the runs they always were.
    rng = np.random.default_rng(seed)
    results = []
    for n in n_values:
        t0 = time.time()
        rates = rejection_rates(n, heston, jumps, n_paths, rng, level=level)
        results.append({"n": n, **rates})
        parts = "  ".join(f"{test}={rates[test]:.3f}" for test in TESTS)
        print(f"  n={n:>8}  {parts}  ({time.time() - t0:.1f}s)", flush=True)
    return results


# =========================================================================
# PLOTTING
# =========================================================================

def draw_rejection_rates(ax, results, title, reference_level=None, n_paths=None, legend=True):
    n_values = [r["n"] for r in results]
    for test in TESTS:
        rates = np.array([r[test] for r in results], dtype=float)
        ax.errorbar(n_values, rates, yerr=rate_error(rates, n_paths) if n_paths else None,
                    linestyle=LINESTYLES[test], color=COLOURS[test], label=LABELS[test],
                    linewidth=1.0, elinewidth=1.0)
    if reference_level is not None:
        reference_line(ax, reference_level, label="nominal level")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("rejection rate")
    ax.set_title(title)
    if legend:
        plain_line_legend(ax)
    tilt_x_ticks(ax)


def plot_size_power_comparison(size_results, power_by_regime, n_paths, level):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    draw_rejection_rates(axes[0], size_results, "Size, both regimes",
                         reference_level=level, n_paths=n_paths)
    axes[0].set_yscale("log")
    plain_log_ticks(axes[0], SIZE_TICKS)
    for column, (regime, label) in enumerate(REGIMES, start=1):
        draw_rejection_rates(axes[column], power_by_regime[regime], f"Power, {label}",
                             n_paths=n_paths, legend=False)
    limits = [ax.get_ylim() for ax in axes[1:]]
    for ax in axes[1:]:
        ax.set_ylim(min(low for low, _ in limits), max(high for _, high in limits))
    fig.suptitle("Empirical size and power under asymptotic calibration, two jump-size regimes")
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/size_power_comparison", n_values=N_GRID, n_paths=12000,
               seed=0)
    p.add_argument("--intensity", type=float, default=5.0)
    p.add_argument("--jump-mean", type=float, default=0.0)      # normal regime
    p.add_argument("--jump-std", type=float, default=0.03)      # normal regime
    p.add_argument("--jump-mode", type=float, default=0.035)    # bimodal regime
    p.add_argument("--jump-spread", type=float, default=0.008)  # bimodal regime
    p.add_argument("--min-jumps", type=int, default=1)
    args = p.parse_args()
    out = Outputs(args, "comparison")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams()
    laws = {regime: jump_params(args, regime) for regime, _ in REGIMES}

    if out.replotting:
        saved = out.saved_data()
        size_results = rows_from_columnar(saved["size"])
        power_by_regime = {r: rows_from_columnar(saved[f"power_{r}"]) for r, _ in REGIMES}
    else:
        # Simulated once, not once per regime: the size arm draws no jumps, so it does not
        # depend on the H1 law and the figure gives it a single shared panel.
        print("-- size (H0, no jumps), shared by both regimes --", flush=True)
        size_results = compare_across_n(args.n_values, heston, None, args.n_paths,
                                        args.level, args.seed)
        power_by_regime = {}
        for regime, _ in REGIMES:
            print(f"-- power (H1, {regime} jump sizes) --", flush=True)
            power_by_regime[regime] = compare_across_n(args.n_values, heston, laws[regime],
                                                       args.n_paths, args.level, args.seed)

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_size_power_comparison(size_results, power_by_regime, args.n_paths, args.level),
        title="Empirical size and power under asymptotic calibration, two jump-size regimes",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "n_paths": args.n_paths,
            "level": args.level,
            "seed": args.seed,
            "heston": heston,
            **{f"jumps_h1_{regime}": law for regime, law in laws.items()},
        },
        data={"size": columnar(size_results),
              **{f"power_{r}": columnar(rows) for r, rows in power_by_regime.items()}},
    )


if __name__ == "__main__":
    main()
