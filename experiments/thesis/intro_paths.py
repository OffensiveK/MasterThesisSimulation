
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

def plot_paths(t, continuous_path, jump_component, jump_path):
    fig, (ax_top, ax_mid, ax_bottom) = plt.subplots(
        3, 1, figsize=(9, 6.4), sharex=True, sharey=True
    )

    ax_top.plot(t, jump_path, linewidth=0.9, color="#333333")
    ax_top.set_title("(a) the path we observe")
    ax_top.set_ylabel(r"$X_t = C_t + J_t$")

    ax_mid.plot(t, continuous_path, linewidth=0.9, color="tab:blue")
    ax_mid.set_title("(b) its continuous part")
    ax_mid.set_ylabel(r"$C_t$")

    ax_bottom.plot(t, jump_component, linewidth=0.9, color="tab:orange")
    ax_bottom.set_title("(c) its jump part")
    ax_bottom.set_ylabel(r"$J_t$")
    ax_bottom.set_xlabel(r"$t$")

    for ax in (ax_top, ax_mid, ax_bottom):
        ax.set_xlim(0.0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/intro_paths", seed=8, from_saved=False, level=None)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--jump-seed", type=int, default=193)
    p.add_argument("--kappa", type=float, default=1.5)
    p.add_argument("--xi", type=float, default=0.6)
    p.add_argument("--theta", type=float, default=0.14)
    p.add_argument("--v0", type=float, default=0.14)
    p.add_argument("--jump-intensity", type=float, default=6.0)
    p.add_argument("--jump-std", type=float, default=0.045)
    p.add_argument("--min-jumps", type=int, default=6)
    args = p.parse_args()
    out = Outputs(args, "intro_paths")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams(kappa=args.kappa, xi=args.xi, theta=args.theta, v0=args.v0)
    jumps = JumpParams(
        intensity=args.jump_intensity,
        size_sampler=NormalJumpSizes(std=args.jump_std),
        min_jumps=args.min_jumps,
    )

    continuous = simulate_continuous_increments(args.n, heston, np.random.default_rng(args.seed))
    pattern = simulate_jump_pattern(args.n, jumps, np.random.default_rng(args.jump_seed))

    t = np.arange(args.n) / args.n
    continuous_path = continuous.cumsum()
    jump_component = jump_increments(pattern).cumsum()
    jump_path = continuous_path + jump_component

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_paths(t, continuous_path, jump_component, jump_path),
        title="Observed path, its continuous part, and its jump part",
        doc=__doc__,
        script=__file__,
        params={
            "n": args.n,
            "seed": args.seed,
            "jump_seed": args.jump_seed,
            "x0": 0.0,
            "heston": heston,
            "jumps": jumps,
        },
        data={
            "jump_times": (pattern.grid_indices / args.n).tolist(),
            "jump_sizes": pattern.sizes.tolist(),
        },
    )


if __name__ == "__main__":
    main()
