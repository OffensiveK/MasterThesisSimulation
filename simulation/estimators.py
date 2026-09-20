import numpy as np

# Guard to avoid division by zero
VARIANCE_FLOOR = 1e-8

def truncation_threshold(n, scale=3.0, tau=0.475):
    return scale * (1.0 / n) ** tau

def window_size(n, gamma=0.5):
    return max(int(n**gamma), 2)

def _window_bounds(n, k_n):
    j = np.arange(n)
    left_edge = j < k_n
    start = np.where(left_edge, j + 1, j - k_n)
    end = np.where(left_edge, np.minimum(j + 1 + k_n, n), j)
    return start, end


def spot_volatility(increments, k_n, u_n):
    """sigma_hat_j^2 = (n / k_n) * sum over the window W_j of
    (Delta_i X)^2 * 1{|Delta_i X| <= u_n}.

    W_j is the backward window [j - k_n, j). Near the left edge (j < k_n)
    there aren't k_n preceding increments, so the window is shifted to
    [0, k_n) rather than shrunk or left undefined. Every index gets a
    full-size estimate, at the cost of a few near-edge windows overlapping
    with points at or after j.
    """
    n = len(increments)
    truncated = np.where(np.abs(increments) <= u_n, increments**2, 0.0)
    cumsum = np.concatenate(([0.0], np.cumsum(truncated)))

    start, end = _window_bounds(n, k_n)
    count = end - start
    return (n / count) * (cumsum[end] - cumsum[start])


def spot_volatility_batch(increments, k_n, u_n):
    """spot_volatility applied to every column of an (n, runs) array at once."""
    n, runs = increments.shape
    truncated = np.where(np.abs(increments) <= u_n, increments**2, 0.0)
    cumsum = np.concatenate([np.zeros((1, runs)), np.cumsum(truncated, axis=0)])

    start, end = _window_bounds(n, k_n)
    count = (end - start)[:, None]
    return (n / count) * (cumsum[end] - cumsum[start])


def studentize_with(increments, sigma2):
    """Z_j = Delta_j X / sqrt(sigma_j^2 * Delta_n), given the variance to divide by. Serves both
    the single-path and the batched shape, and both the feasible and the oracle statistic: the
    studentize* functions below pass the truncated estimate sigma_hat^2, while the experiments
    that isolate the plug-in error pass the simulator's own spot variance."""
    n = increments.shape[0]
    dt = 1.0 / n
    return increments / np.sqrt(np.maximum(sigma2, VARIANCE_FLOOR) * dt)


def studentize(increments, k_n=None, gamma=0.5, scale=3.0, tau=0.475, truncate=True):
    """Z_hat_j = Delta_j X / sqrt(sigma_hat_j^2 * Delta_n). The window defaults to window_size(n,
    gamma), so a caller with no reason to name k_n does not have to; passing k_n overrides it."""
    n = len(increments)
    k_n = window_size(n, gamma) if k_n is None else k_n
    u_n = truncation_threshold(n, scale, tau) if truncate else np.inf
    return studentize_with(increments, spot_volatility(increments, k_n, u_n))


def studentize_batch(increments, k_n=None, gamma=0.5, scale=3.0, tau=0.475, truncate=True):
    """studentize applied to every column of an (n, runs) array at once."""
    n = increments.shape[0]
    k_n = window_size(n, gamma) if k_n is None else k_n
    u_n = truncation_threshold(n, scale, tau) if truncate else np.inf
    return studentize_with(increments, spot_volatility_batch(increments, k_n, u_n))
