"""Generate an example figure: jump pattern, continuous path, and their sum."""

# =========================================================================
# IMPORTS
# =========================================================================

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import Outputs, parser
from runtime.plotting import use_style
from simulation.continuous import HestonParams, simulate_continuous_increments
from simulation.jumps import JumpParams, jump_increments, simulate_jump_pattern

use_style()

# =========================================================================
# PLOTTING
# =========================================================================

def plot_decomposition(pattern, continuous, title="Path decomposition"):
    n = pattern.n
    t = np.arange(n) / n
    jumps = jump_increments(pattern)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
    axes[0].stem(pattern.grid_indices / n, pattern.sizes, basefmt=" ")
    axes[0].set_ylabel("jump size")
    axes[0].set_title("Jump pattern")

    axes[1].plot(t, continuous.cumsum())
    axes[1].set_ylabel("log price")
    axes[1].set_title("Continuous path")

    axes[2].plot(t, (continuous + jumps).cumsum())
    axes[2].set_ylabel("log price")
    axes[2].set_xlabel("t")
    axes[2].set_title("Sum")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():

    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/examples", seed=1, from_saved=False, level=None)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--jump-intensity", type=float, default=8.0)
    args = p.parse_args()
    out = Outputs(args, "jump_example")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams()
    jumps = JumpParams(intensity=args.jump_intensity)

    rng = np.random.default_rng(args.seed)
    pattern = simulate_jump_pattern(args.n, jumps, rng)
    continuous = simulate_continuous_increments(args.n, heston, rng)

    # =====================================================================
    # OUTPUT
    # =====================================================================

    out.finish(
        plot_decomposition(pattern, continuous),
        title="Jump pattern / continuous path / sum decomposition example",
        doc=__doc__,
        script=__file__,
        params={
            "n": args.n,
            "seed": args.seed,
            "heston": heston,
            "jumps": jumps,
        },
    )


if __name__ == "__main__":
    main()
