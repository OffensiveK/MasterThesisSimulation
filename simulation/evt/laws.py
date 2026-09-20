
from functools import lru_cache

import numpy as np

def _ppf_from_cdf(cdf, p, tol=1e-10, max_iter=200):
    # Invert function by bisection. 
    lo, hi = 0.0, 1.0
    while cdf(hi) < p:
        lo, hi = hi, hi * 2
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)

def gumbel_ppf(p):
    return -np.log(-np.log(p))


def gumbel_isf(p):
    return gumbel_ppf(1 - p)

def exponential_isf(p):
    return -np.log(p)

def deheuvels_cdf(x, tails=2, terms=200):
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)
    out = np.zeros_like(x)
    positive = x > 0
    if np.any(positive):
        r = np.arange(1, terms + 1)
        log_terms = np.log1p(-np.exp(-np.outer(r, x[positive])))
        out[positive] = np.exp(tails * log_terms.sum(axis=0))
    return out.item() if scalar_input else out


@lru_cache(maxsize=None)
def deheuvels_ppf(p, tails=2, tol=1e-10, max_iter=200, terms=200):
    """`terms` truncates the product at the first `terms` spacings, which is what a
    search restricted to the top K = terms + 1 order statistics converges to; the
    default is large enough to be the full product to machine precision."""
    return _ppf_from_cdf(
        lambda x: deheuvels_cdf(x, tails=tails, terms=terms), p, tol=tol, max_iter=max_iter
    )


def deheuvels_isf(p, tails=2, terms=200):
    return deheuvels_ppf(1 - p, tails=tails, terms=terms)
