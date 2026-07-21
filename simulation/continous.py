from dataclasses import dataclass

import numpy as np

@dataclass
class HestonParams:
    mu:    float =  0.05   # drift of the log-price
    kappa: float =  2.0    # mean-reversion speed of the variance process
    theta: float =  0.04   # long-run mean of the variance process
    xi:    float =  0.3    # volatility of variance ("vol of vol")
    rho:   float = -0.2    # correlation between the price and variance Brownian motions
    v0:    float =  0.04   # initial variance


def simulate_continuous_increments(n, heston, rng=None):
    # note the following simulates log increments
    rng = rng or np.random.default_rng()
    dt = 1.0 / n
    sqrt_dt = np.sqrt(dt)

    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    w2 = heston.rho * z1 + np.sqrt(1 - heston.rho**2) * z2

    v = heston.v0
    continuous = np.empty(n)
    for j in range(n):
        v_pos = max(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        continuous[j] = (heston.mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * w2[j]
        v += heston.kappa * (heston.theta - v_pos) * dt + heston.xi * sqrt_v * sqrt_dt * z1[j]

    return continuous

def simulate_continuous_increments_batch(runs, n, heston, rng=None):
    # same as previous function for running as a batch
    rng = rng or np.random.default_rng()
    dt = 1.0 / n
    sqrt_dt = np.sqrt(dt)

    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    w2 = heston.rho * z1 + np.sqrt(1 - heston.rho**2) * z2

    v = np.full(runs, heston.v0)    
    continuous = np.empty((n,runs))

    for j in range(n):
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        continuous[j] = (heston.mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * w2[j]
        v += heston.kappa * (heston.theta - v_pos) * dt + heston.xi * sqrt_v * sqrt_dt * z1[j]

    return continuous

