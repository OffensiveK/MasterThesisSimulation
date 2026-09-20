
import numpy as np

from .batching import batch_size_for, chunks
from .continuous import simulate_continuous_increments_batch
from .estimators import studentize_batch, window_size
from .evt import (REJECT, STATISTIC_BATCH, TESTS, asymptotic_critical_values,
                  empirical_critical)
from .jumps import jump_increments

#: Normal quantile behind a 95% interval, the coverage every reported rate is stated at.
WALD_Z = 1.96

#: The percentile bootstrap interval every figure quoting one reads at the same coverage.
BOOTSTRAP_COVERAGE = 95.0


def rate_error(rate, draws, z=WALD_Z):
    rate = np.asarray(rate, dtype=float)
    return z * np.sqrt(rate * (1.0 - rate) / draws)

def percentile_ci(draws, coverage=BOOTSTRAP_COVERAGE):
    tail = (100.0 - coverage) / 2.0
    lo, hi = np.percentile(np.asarray(draws, dtype=float), [tail, 100.0 - tail])
    return float(lo), float(hi)


def bootstrap_median_ci(values, n_boot, rng, coverage=BOOTSTRAP_COVERAGE):
    values = np.asarray(values, dtype=float)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = np.median(rng.choice(values, size=values.size, replace=True))
    return percentile_ci(draws, coverage)


def loglog_slope(n, y):
    x, ly = np.log(np.asarray(n, dtype=float)), np.log(np.asarray(y, dtype=float))
    design = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(design, ly, rcond=None)
    return -float(coef[1])


def statistics_sharing_paths(n, heston, patterns, n_paths, rng, k_n_exponent=0.5):
    k_n = window_size(n, k_n_exponent)
    collected = {key: {test: [] for test in TESTS} for key in patterns}
    for take in chunks(n_paths, batch_size_for(n, n_paths, arrays_per_path=8)):
        continuous = simulate_continuous_increments_batch(take, n, heston, rng)
        for key, pattern in patterns.items():
            increments = (continuous if pattern is None
                          else continuous + jump_increments(pattern)[:, None])
            z_hat = studentize_batch(increments, k_n)
            for test in TESTS:
                collected[key][test].append(STATISTIC_BATCH[test](z_hat))
    return {key: {test: np.concatenate(v) for test, v in d.items()}
            for key, d in collected.items()}


def rates(null_stats, alt_stats, level=0.05):
    critical = asymptotic_critical_values(level)
    out = {}
    for test in TESTS:
        threshold = empirical_critical(null_stats[test], level)
        out[test] = {
            "size": float((null_stats[test] > critical[test]).mean()),
            "power": float((alt_stats[test] > critical[test]).mean()),
            "power_adj": float((alt_stats[test] > threshold).mean()),
            "threshold_adj": threshold,
        }
    return out


def oracle_rejection_rates(n, n_paths, rng, level=0.05, arrays_per_path=6):
    rejections = {test: 0 for test in TESTS}
    for chunk in chunks(n_paths, batch_size_for(n, n_paths, arrays_per_path)):
        z = rng.standard_normal((n, chunk))
        for test in TESTS:
            rejections[test] += REJECT[test](STATISTIC_BATCH[test](z), level).sum()
    return {test: count / n_paths for test, count in rejections.items()}
