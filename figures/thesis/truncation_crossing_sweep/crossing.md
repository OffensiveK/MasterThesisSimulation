# Spot-variance contamination as a spike crosses the truncation threshold

What the truncation does to the spot estimator as a spike crosses the threshold u_n.
Below u_n the contaminated windows inflate like eta^2; once eta exceeds u_n the indicator drops
the jump increment and the contamination collapses back to one, which is lem:h1_isolate.

Generated: 2026-09-11T21:53:05+00:00

## Parameters

script: experiments/thesis/truncation_crossing_sweep.py
n_values: [5932, 18362]
ratios: [0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.5, 2.0, 4.0]
n_paths: 6000
n_boot: 400
level: 0.05
seed: 13
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04

## Data

Raw output: [crossing.json](crossing.json)
