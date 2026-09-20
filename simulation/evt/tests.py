
import numpy as np

from .laws import deheuvels_isf, exponential_isf, gumbel_isf


def normalizing_constants(m):
    log_m = np.log(m)
    b_m = np.sqrt(2 * log_m) - (np.log(np.log(m)) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * log_m))
    a_m = 1.0 / np.sqrt(2 * log_m)
    return a_m, b_m


def folded_normalizing_constants(m):
    a_m, b_m = normalizing_constants(m)
    return a_m, b_m + a_m * np.log(2)


def gumbel_critical(level=0.05):
    return gumbel_isf(level)


def exp_critical(level=0.05):
    return exponential_isf(level)


def renyi_critical(level=0.05):
    return deheuvels_isf(level, tails=2)

def asymptotic_critical_values(level=0.05):
    return {test: CRITICAL[test](level) for test in TESTS}


def empirical_critical(statistics, level=0.05):
    """The Monte-Carlo counterpart of the *_critical values above: the 1 - level quantile of a
    sample of null statistics, which is what a size-adjusted power reading tests against."""
    return float(np.quantile(statistics, 1.0 - level))


def folded_critical_value(n, level=0.05):
    a_n, b_tilde = folded_normalizing_constants(n)
    return b_tilde + a_n * gumbel_critical(level)

def gumbel_reject(statistic, level=0.05):
    return statistic > gumbel_critical(level)


def exp_reject(statistic, level=0.05):
    return statistic > exp_critical(level)


def renyi_reject(statistic, level=0.05):
    return statistic > renyi_critical(level)


def _transpose_contiguous(z):
    return np.ascontiguousarray(z.T)


def gumbel_statistic_batch(z):
    n = z.shape[0]
    a_m, b_tilde_m = folded_normalizing_constants(n)
    zt = np.abs(_transpose_contiguous(z))
    return (zt.max(axis=-1) - b_tilde_m) / a_m


def exp_statistic_batch(z):
    n = z.shape[0]
    a_m, _ = normalizing_constants(n)
    zt = _transpose_contiguous(z)
    top2 = np.sort(np.partition(zt, -2, axis=-1)[:, -2:], axis=-1)
    return (top2[:, -1] - top2[:, -2]) / a_m


def renyi_statistic_batch(z):
    n = z.shape[0]
    a_m, _ = normalizing_constants(n)
    sorted_z = np.sort(_transpose_contiguous(z), axis=-1)
    spacings = np.diff(sorted_z, axis=-1)
    return spacings.max(axis=-1) / a_m

TESTS = ("gumbel", "exp", "renyi")
CRITICAL = {"gumbel": gumbel_critical, "exp": exp_critical, "renyi": renyi_critical}
REJECT = {"gumbel": gumbel_reject, "exp": exp_reject, "renyi": renyi_reject}
# STATISTIC = {"gumbel": gumbel_statistic, "exp": exp_statistic, "renyi": renyi_statistic}
STATISTIC_BATCH = {"gumbel": gumbel_statistic_batch, "exp": exp_statistic_batch,
                   "renyi": renyi_statistic_batch}
