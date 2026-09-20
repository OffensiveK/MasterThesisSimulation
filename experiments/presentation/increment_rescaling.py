
# =========================================================================
# IMPORTS
# =========================================================================

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from runtime.plotting import use_style
from runtime.report import write_report
from simulation.continuous import HestonParams
from simulation.jumps import JumpParams, NormalJumpSizes
from simulation.model import increment_rescaling_path

use_style()

# =========================================================================
# SIMULATION
# =========================================================================

# =========================================================================
# PLOTTING
# =========================================================================

def plot_increment_rescaling(results, title="Increment rescaling as n grows"):
    n_cols = len(results)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, r in enumerate(results):
        top, bottom = axes[0, col], axes[1, col]

        top.plot(r["t"], r["increments"], linewidth=0.7, color="tab:blue")
        top.set_title(f"n = {r['n']}")
        top.set_xlabel("t")

        z = r["z"]
        z_max = np.nanmax(np.abs(z))
        bottom.plot(r["t"], z, linewidth=0.7, color="tab:orange")
        bottom.axhline(0, color="gray", linewidth=0.5)
        bottom.set_yscale("symlog", linthresh=1.0)
        bottom.set_ylim(-1.1 * z_max, 1.1 * z_max)
        bottom.set_xlabel("t")
        bottom.text(
            0.02, 0.95, f"max|Z|={z_max:.1f}",
            transform=bottom.transAxes, ha="left", va="top", fontsize=8, color="gray",
        )

        if col == 0:
            top.set_ylabel(r"raw increment $\Delta_j X$")
            bottom.set_ylabel(r"studentized $\hat Z_j$")

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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=int, nargs="+", default=[100, 500, 2000, 10000])
    parser.add_argument("--seed", type=int, default=1452)
    parser.add_argument("--with-jumps", action="store_true")
    parser.add_argument("--jump-intensity", type=float, default=25.0)
    parser.add_argument("--jump-std", type=float, default=0.02)
    parser.add_argument("--out", type=Path, default=Path("figures/presentation"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    stem = "increment_rescaling_jumps" if args.with_jumps else "increment_rescaling"
    figure_path = args.out / f"{stem}.png"

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams()
    jumps = (
        JumpParams(intensity=args.jump_intensity, size_sampler=NormalJumpSizes(std=args.jump_std))
        if args.with_jumps
        else None
    )

    results = increment_rescaling_path(
        args.n_values, heston, jumps=jumps, seed=args.seed, truncate=False
    )

    # =====================================================================
    # OUTPUT
    # =====================================================================

    title = "Increment rescaling as n grows" + (" (with jumps)" if args.with_jumps else "")
    plot_increment_rescaling(results, title).savefig(figure_path)
    write_report(
        figure_path.with_suffix(".md"),
        title=title,
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "seed": args.seed,
            "truncate": False,
            "heston": heston,
            "jumps": jumps,
        },
    )
    print(f"Saved {figure_path} and {figure_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
