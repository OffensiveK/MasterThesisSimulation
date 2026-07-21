"""Generate an example figure: jump pattern, continuous path, and their sum."""
from pathlib import Path

import numpy as np

from simulation.jumps import JumpParams, simulate_jump_pattern
from simulation.model import HestonParams, simulate_continuous_increments
from simulation.plotting import plot_decomposition


def main():
    rng = np.random.default_rng(1)
    n = 2000
    pattern = simulate_jump_pattern(n, JumpParams(intensity=8.0), rng)
    continuous = simulate_continuous_increments(n, HestonParams(), rng)

    out = Path("figures")
    out.mkdir(exist_ok=True)
    fig = plot_decomposition(pattern, continuous)
    fig.savefig(out / "jump_decomposition_example.png", dpi=150)


if __name__ == "__main__":
    main()
