"""Diagram generation (the only part of this project allowed to lean on a library)."""
import matplotlib.pyplot as plt

from .jumps import jump_increments


def plot_path(increments, title="Simulated log-price path"):
    fig, ax = plt.subplots()
    ax.plot(increments.cumsum())
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("log price")
    return fig


def plot_decomposition(pattern, continuous, title="Path decomposition"):
    """Three panels: jump pattern, continuous path, and their sum."""
    n = pattern.n
    t = [j / n for j in range(n)]
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


def plot_rejection_rates(results, title, reference_level=None):
    n_values = [r["n"] for r in results]
    fig, ax = plt.subplots()
    ax.plot(n_values, [r["gumbel"] for r in results], marker="o", label="Gumbel")
    ax.plot(n_values, [r["renyi"] for r in results], marker="s", label="Renyi")
    if reference_level is not None:
        ax.axhline(reference_level, linestyle="--", color="gray", label="nominal level")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("rejection rate")
    ax.set_title(title)
    ax.legend()
    return fig
