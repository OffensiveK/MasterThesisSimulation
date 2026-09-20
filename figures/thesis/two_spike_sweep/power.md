# Two spikes of comparable size

Two spikes of comparable size, which is the configuration the exponential test cannot see.
Sweeps the size ratio and the separation, with the Rényi test as the comparison.

Generated: 2026-08-10T16:58:39+00:00

## Parameters

script: experiments/thesis/two_spike_sweep.py
n_values: [1916, 10437]
ratios: [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
first_multiplier: 1.5
separation: 0.2
n_paths: 12000
level: 0.05
seed: 11
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04

## Data

Raw output: [power.json](power.json)
