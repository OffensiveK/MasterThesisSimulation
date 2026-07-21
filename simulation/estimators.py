import numpy as np

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
    sigma2 = np.empty(n)
    for j in range(n):
        start = min(max(j - k_n, 0), n - k_n)
        sigma2[j] = (n / k_n) * (cumsum[start + k_n] - cumsum[start])
    return sigma2


def studentize(increments, k_n, u_n=None, scale=3.0, tau=0.475):
    """Z_hat_j = Delta_j X / sqrt(sigma_hat_j^2 * Delta_n)."""
    n = len(increments)
    u_n = scale * (1.0 / n) ** tau
    sigma2 = spot_volatility(increments, k_n, u_n)
    dt = 1.0 / n
    return increments / np.sqrt(sigma2 * dt)
