
# =========================================================================
# IMPORTS
# =========================================================================


import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import Outputs, parser
from runtime.plotting import use_style
from simulation.continuous import HestonParams, simulate_continuous_increments
from simulation.jumps import JumpParams, NormalJumpSizes, jump_increments, simulate_jump_pattern

use_style()

# =========================================================================
# PLOTTING
# =========================================================================

def plot_path_and_increments(t, increments, jump_indices=None,
                             title="Simulated path and increments"):
    fig, (ax_path, ax_inc) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]}
    )
    ax_path.plot(t, increments.cumsum(), linewidth=0.9, color="tab:blue")
    ax_path.set_ylabel("log price")
    ax_path.set_title(title)

    ax_inc.plot(t, increments, linewidth=0.6, color="tab:blue")
    ax_inc.axhline(0, color="gray", linewidth=0.5)
    if jump_indices is not None and len(jump_indices) > 0:
        ax_inc.scatter(
            t[jump_indices], increments[jump_indices],
            color="tab:orange", s=22, zorder=3, label="jump",
        )
        ax_inc.legend(loc="upper right")
    ax_inc.set_xlabel("t")
    ax_inc.set_ylabel(r"increment $\Delta_j X$")

    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/presentation", seed=8, from_saved=False, level=None)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--jump-seed", type=int, default=None)
    p.add_argument("--kappa", type=float, default=1.5)
    p.add_argument("--xi", type=float, default=0.6)
    p.add_argument("--jump-intensity", type=float, default=8.0)
    p.add_argument("--jump-std", type=float, default=0.03)
    args = p.parse_args()
    out = Outputs(args, "single_path_example")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams(kappa=args.kappa, xi=args.xi)
    jumps = JumpParams(intensity=args.jump_intensity,
                       size_sampler=NormalJumpSizes(std=args.jump_std))

    jump_seed = args.seed if args.jump_seed is None else args.jump_seed
    continuous = simulate_continuous_increments(args.n, heston, np.random.default_rng(args.seed))
    pattern = simulate_jump_pattern(args.n, jumps, np.random.default_rng(jump_seed))
    increments = continuous + jump_increments(pattern)
    t = np.arange(args.n) / args.n

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_path_and_increments(
        t, increments, jump_indices=pattern.grid_indices,
        title="Simulated path and increments: stochastic-volatility clustering",
    ),
        title="Single simulated path highlighting volatility clustering",
        doc=__doc__,
        script=__file__,
        params={
            "n": args.n,
            "seed": args.seed,
            "jump_seed": jump_seed,
            "heston": heston,
            "jumps": jumps,
        },
    )


if __name__ == "__main__":
    main()
