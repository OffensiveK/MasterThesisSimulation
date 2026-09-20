# Increment rescaling as n grows (with jumps)

One path read at several sampling frequencies: raw increments against studentized ones.

Generated: 2026-08-10T02:30:13+00:00

## Parameters

script: experiments/examples/plot_increment_rescaling.py
n_values: [100, 500, 2000, 10000]
seed: 1452
truncate: False
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04
jumps: JumpParams
  intensity: 25.0
  size_sampler: NormalJumpSizes
    mean: 0.0
    std: 0.02
  min_jumps: 0
  min_peak_magnitude: 0.0
