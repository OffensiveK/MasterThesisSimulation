# Size-adjusted power by spot-volatility bandwidth exponent

Size-adjusted power of the three tests against the spot-volatility bandwidth exponent gamma.
Each rule is recalibrated at every gamma so the curves read as power alone, not as size drift.

Generated: 2026-08-10T17:31:08+00:00

## Parameters

script: experiments/thesis/bandwidth_power_sweep.py
n_values: [200, 619, 1916, 5932, 18362]
gammas: [0.3, 0.4, 0.5, 0.6, 0.7]
n_paths: 6000
level: 0.05
seed: 1000
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04
jumps: JumpParams
  intensity: 5.0
  size_sampler: NormalJumpSizes
    mean: 0.0
    std: 0.03
  min_jumps: 1
  min_peak_magnitude: 0.03

## Data

Raw output: [power.json](power.json)
