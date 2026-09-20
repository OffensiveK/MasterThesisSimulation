# Per-test statistic inflation from volatility misestimation

Per-test inflation of the statistic caused by spot-volatility misestimation.
Isolates how much of each test's size distortion the plug-in error alone accounts for.

Generated: 2026-08-10T20:13:54+00:00

## Parameters

script: experiments/thesis/test_statistic_inflation_sweep.py
n_values: [200, 352, 619, 1089, 1916, 3372, 5932, 10437, 18362, 32306, 56838, 100000]
n_paths: 16000
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

Raw output: [inflation.json](inflation.json)
