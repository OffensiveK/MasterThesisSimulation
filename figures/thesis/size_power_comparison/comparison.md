# Empirical size and power under asymptotic calibration, two jump-size regimes

The thesis's size/power comparison figure (Figure 2), drawn as one 1x3 figure.
The size arm draws no jumps and so is identical across the two H1 regimes: it is simulated once
and shares a single panel, while the two power arms take one panel each.

Generated: 2026-09-20T20:36:52+00:00

## Parameters

script: experiments/thesis/size_power_comparison.py
n_values: [200, 352, 619, 1089, 1916, 3372, 5932, 10437, 18362, 32306, 56838, 100000]
n_paths: 12000
level: 0.05
seed: 0
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04
jumps_h1_normal: JumpParams
  intensity: 5.0
  size_sampler: NormalJumpSizes
    mean: 0.0
    std: 0.03
  min_jumps: 1
  min_peak_magnitude: 0.03
jumps_h1_bimodal: JumpParams
  intensity: 5.0
  size_sampler: BimodalJumpSizes
    mode: 0.035
    spread: 0.008
  min_jumps: 1
  min_peak_magnitude: 0.0

## Data

Raw output: [comparison.json](comparison.json)
