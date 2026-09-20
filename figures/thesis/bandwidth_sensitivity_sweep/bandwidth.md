# Empirical size by spot-volatility bandwidth exponent

Empirical size of the three tests against the spot-volatility bandwidth exponent gamma in
k_n = n^gamma, which is the estimator bias the tests inherit rather than an imposed perturbation.

Generated: 2026-08-10T17:24:23+00:00

## Parameters

script: experiments/thesis/bandwidth_sensitivity_sweep.py
n_values: [200, 619, 1916, 5932, 18362, 56838]
gammas: [0.3, 0.4, 0.5, 0.6, 0.7]
n_paths: 6000
level: 0.05
seed: 0
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04

## Data

Raw output: [bandwidth.json](bandwidth.json)
