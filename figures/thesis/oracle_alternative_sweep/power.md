# Plug-in cost under the alternative, oracle against feasible

What the spot-volatility plug-in error costs under the alternative, once size is granted.
The same paths are read twice, once studentized by the simulator's own spot variance and once by
the truncated estimate, each recalibrated to its own null so the gap is detection ability alone.

Generated: 2026-08-18T03:48:26+00:00

## Parameters

script: experiments/thesis/oracle_alternative_sweep.py
n_values: [1916, 10437]
location: 0.5
multipliers: [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
n_paths: 12000
level: 0.05
seed: 23
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04
critical_scales: {1916: 0.017764399855753833, 10437: 0.008421725311931385}

## Data

Raw output: [power.json](power.json)
