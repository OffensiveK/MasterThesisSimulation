
# =========================================================================
# IMPORTS
# =========================================================================

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import Outputs, parser
from runtime.plotting import use_style
from simulation.continuous import HestonParams, simulate_continuous_increments

use_style()

# =========================================================================
# PATH
# =========================================================================

def local_scale(increments, window):
    half = window // 2
    padded = np.pad(increments, (half, half), mode="edge")
    return np.array([padded[i:i + window].std() for i in range(len(increments))])


def build_ornament(n, heston, seed, jump_time, jump_fraction, window):
    increments = simulate_continuous_increments(n, heston, np.random.default_rng(seed))
    continuous = increments.cumsum()

    jump_index = int(jump_time * n)
    jump_size = jump_fraction * (continuous.max() - continuous.min())

    path = continuous.copy()
    path[jump_index:] += jump_size

    x = np.arange(n) / (n - 1)
    y = (path - path.min()) / (path.max() - path.min())
    return x, y, jump_index, local_scale(increments, window), jump_size


# =========================================================================
# TIKZ
# =========================================================================

COORDS_PER_LINE = 5


def _coordinate_block(x, y, indent):
    points = [f"({xi:.4f},{yi:.4f})" for xi, yi in zip(x, y)]
    lines = [
        indent + "".join(points[i:i + COORDS_PER_LINE])
        for i in range(0, len(points), COORDS_PER_LINE)
    ]
    return "\n".join(lines)


def tikz_source(x, y, jump_index, jump_time, n, seed):
    indent = " " * 4
    before = _coordinate_block(x[:jump_index], y[:jump_index], indent)
    after = _coordinate_block(x[jump_index:], y[jump_index:], indent)

    return f"""
\\newcommand{{\\titlepagepath}}[1]{{%
  \\begin{{tikzpicture}}[x={{#1}}, y={{0.1*#1}}, line width=0.45pt,
                      line join=round, line cap=round, draw=tppath]
    \\useasboundingbox (0,0) rectangle (1,1);
    \\draw plot coordinates {{
{before}
    }};
    \\draw plot coordinates {{
{after}
    }};
  \\end{{tikzpicture}}%
}}
"""

# =========================================================================
# PLOTTING
# =========================================================================

def plot_ornament(x, y, jump_index, scale):
    """A proof sheet, not the artwork. The top panel is the ornament at its printed proportions,
    the bottom one the local scale, which says whether the texture is even across the width."""
    fig = plt.figure(figsize=(14, 4))

    ax = fig.add_axes([0.04, 0.55, 0.93, 0.38])
    ax.plot(x[:jump_index], y[:jump_index], linewidth=0.5, color="0.35",
            solid_joinstyle="round", solid_capstyle="round")
    ax.plot(x[jump_index:], y[jump_index:], linewidth=0.5, color="0.35",
            solid_joinstyle="round", solid_capstyle="round")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, 1.04)
    ax.axis("off")
    ax.set_title("Title-page ornament, at the printed 10:1 proportions",
                 fontsize=9, loc="left", color="0.3")

    ax_scale = fig.add_axes([0.04, 0.10, 0.93, 0.33])
    ax_scale.plot(x, scale, linewidth=0.8, color="tab:blue")
    ax_scale.axvline(x[jump_index], color="tab:orange", linewidth=1.0, label="jump")
    ax_scale.set_xlim(0, 1)
    ax_scale.set_ylim(bottom=0.0)
    ax_scale.set_xlabel("t")
    ax_scale.set_ylabel("local scale")
    ax_scale.legend(loc="upper right")

    return fig

# =========================================================================
# MAIN
# =========================================================================

def main():
    
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/titlepage_path", seed=99, from_saved=False, level=None)
    p.add_argument("--n", type=int, default=900)
    p.add_argument("--jump-time", type=float, default=2.0 / 3.0)
    p.add_argument("--jump-fraction", type=float, default=0.425)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--kappa", type=float, default=1.5)
    p.add_argument("--theta", type=float, default=0.14)
    p.add_argument("--xi", type=float, default=0.6)
    p.add_argument("--rho", type=float, default=-0.2)
    p.add_argument("--v0", type=float, default=0.14)
    args = p.parse_args()
    out = Outputs(args, "titlepage_path")
    tex_path = out.directory / "titlepage_path.tex"

    # =====================================================================
    # SIMULATION
    # =====================================================================

    heston = HestonParams(kappa=args.kappa, theta=args.theta, xi=args.xi,
                          rho=args.rho, v0=args.v0)
    x, y, jump_index, scale, jump_size = build_ornament(
        args.n, heston, args.seed, args.jump_time, args.jump_fraction, args.window
    )

    # =====================================================================
    # OUTPUT
    # =====================================================================

    tex_path.write_text(
        tikz_source(x, y, jump_index, args.jump_time, args.n, args.seed),
        encoding="utf-8", newline="\n",
    )
    out.finish(
        plot_ornament(x, y, jump_index, scale),
        title="Title-page ornament: a finely sampled Heston path with one jump",
        doc=__doc__,
        script=__file__,
        params={
            "n": args.n,
            "seed": args.seed,
            "jump_time": round(args.jump_time, 6),
            "jump_index": jump_index,
            "jump_fraction": args.jump_fraction,
            "jump_size": round(float(jump_size), 6),
            "window": args.window,
            "heston": heston,
            "loudest_over_median_scale": round(float(scale.max() / np.median(scale)), 3),
        },
    )
    print(f"Saved {tex_path}, {out.figure} and {out.report}")


if __name__ == "__main__":
    main()
