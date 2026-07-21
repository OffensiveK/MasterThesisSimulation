
import numpy as np

from .detection import gumbel_reject, gumbel_statistic, renyi_reject, renyi_statistic
from .estimators import studentize
from .model import simulate_log_price_increments


def rejection_rates(n, heston, jumps, n_paths, k_n_exponent=0.5, level=0.05, rng=None):
    rng = rng or np.random.default_rng()
    k_n = max(int(n**k_n_exponent), 2)
    gumbel_rejections = 0
    renyi_rejections = 0
    for _ in range(n_paths):
        increments = simulate_log_price_increments(n, heston, jumps, rng)
        z = studentize(increments, k_n)
        gumbel_rejections += gumbel_reject(gumbel_statistic(z), level)
        renyi_rejections += renyi_reject(renyi_statistic(z), level)
    return gumbel_rejections / n_paths, renyi_rejections / n_paths


def compare_across_n(n_values, heston, jumps, n_paths=500, level=0.05, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for n in n_values:
        g_rate, r_rate = rejection_rates(n, heston, jumps, n_paths, level=level, rng=rng)
        results.append({"n": n, "gumbel": g_rate, "renyi": r_rate})
    return results
