
import numpy as np

from .continuous import simulate_continuous_increments, simulate_continuous_increments_batch
from .estimators import studentize
from .jumps import (
    JumpPattern,
    jump_increments,
    jump_increments_batch,
    simulate_jump_increments_batch,
    simulate_jump_pattern,
)

def simulate_log_price_increments(n, heston, jumps, rng):
    continuous = simulate_continuous_increments(n, heston, rng)
    if jumps is None:
        return continuous
    pattern = jumps if isinstance(jumps, JumpPattern) else simulate_jump_pattern(n, jumps, rng)
    return continuous + jump_increments(pattern)


def simulate_log_price_increments_batch(runs, n, heston, jumps, rng):
    """`jumps` is None for the continuous part alone, JumpParams to draw a fresh pattern per path,
    one JumpPattern to put the same one on every path, or a sequence of runs patterns to give each
    path its own - which is how an experiment that builds its own jump configurations feeds them
    in without assembling the increment array itself."""
    continuous = simulate_continuous_increments_batch(runs, n, heston, rng)
    if jumps is None:
        return continuous
    if isinstance(jumps, JumpPattern):
        return continuous + jump_increments(jumps)[:, None]
    if isinstance(jumps, (list, tuple)):
        return continuous + jump_increments_batch(jumps)
    return continuous + simulate_jump_increments_batch(n, runs, jumps, rng)


def increment_rescaling_path(
    n_values, heston, jumps, k_n_exponent=0.5, scale=3.0, tau=0.475, seed=0, truncate=True
):
    rng = np.random.default_rng(seed)
    n_max = max(n_values)
    for n in n_values:
        if n_max % n != 0:
            raise ValueError(f"n_max={n_max} must be divisible by every n in n_values (got {n})")

    fine_increments = simulate_log_price_increments(n_max, heston, jumps, rng)

    results = []
    for n in n_values:
        factor = n_max // n
        coarse = fine_increments.reshape(n, factor).sum(axis=1)
        z = studentize(coarse, gamma=k_n_exponent, scale=scale, tau=tau, truncate=truncate)
        t = np.arange(n) / n
        results.append({"n": n, "t": t, "increments": coarse, "z": z})
    return results
