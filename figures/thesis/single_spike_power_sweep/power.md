# Power against a single spike's size, across n

Power of the three tests against a single spike's size, at fixed n, in multiples of the critical
jump scale. Every point carries a shared null arm, so the raw and size-adjusted rates are reported
side by side.

Generated: 2026-08-18T00:38:53+00:00

## Parameters

script: experiments/thesis/single_spike_power_sweep.py
n_values: [10000, 100000]
location: 0.5
multipliers: [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
n_paths: 12000
level: 0.05
seed: 7
negative: False
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04
critical_scales: {10000: 0.00858386410515739, 100000: 0.003034854258770293}

## Data

Raw output: [power.json](power.json)
