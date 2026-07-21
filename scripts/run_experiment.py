"""Run the Section 5 practical-part comparison and save the diagrams."""
import argparse
from pathlib import Path

from simulation.experiment import compare_across_n
from simulation.jumps import JumpParams
from simulation.model import HestonParams
from simulation.plotting import plot_rejection_rates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=int, nargs="+", default=[500, 1000, 2000, 5000, 10000])
    parser.add_argument("--n-paths", type=int, default=500)
    parser.add_argument("--level", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    heston = HestonParams()

    size_results = compare_across_n(args.n_values, heston, jumps=None, n_paths=args.n_paths, level=args.level)
    plot_rejection_rates(size_results, "Empirical size (no jumps)", reference_level=args.level).savefig(
        args.out / "size_comparison.png"
    )

    power_results = compare_across_n(
        args.n_values, heston, jumps=JumpParams(), n_paths=args.n_paths, level=args.level
    )
    plot_rejection_rates(power_results, "Empirical power (with jumps)").savefig(
        args.out / "power_comparison.png"
    )


if __name__ == "__main__":
    main()
