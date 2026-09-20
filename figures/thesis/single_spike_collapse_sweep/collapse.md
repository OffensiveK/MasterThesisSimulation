# Power against a spike shrinking at n^-rho, across a family of exponents

Power against a single spike shrinking at n^-rho, across n, for a family of exponents.
Size-adjusted power turns out to depend on the spike's studentized value alone, so configurations
with different n and different rho but a shared studentized value collapse onto one curve.

Generated: 2026-09-11T21:33:36+00:00

## Parameters

script: experiments/thesis/single_spike_collapse_sweep.py
n_values: [200, 352, 619, 1089, 1916, 3372, 5932, 10437, 18362, 32306, 56838, 100000]
exponents: [0.3, 0.45, 0.5, 0.7]
anchor_n: 200
location: 0.5
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

## Data

Raw output: [collapse.json](collapse.json)
