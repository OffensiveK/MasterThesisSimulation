# Single-spike power against spike location and sign

Single-spike power against the spike's location along the path and its sign.
The left edge is where the backward spot-volatility window is shifted rather than shrunk.

Generated: 2026-08-18T00:50:39+00:00

## Parameters

script: experiments/thesis/spike_location_sweep.py
n_values: [1916, 10437]
locations: [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 0.999]
multiplier: 1.0
n_paths: 12000
level: 0.05
seed: 17
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04

## Data

Raw output: [power.json](power.json)
