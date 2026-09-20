# Observed path, its continuous part, and its jump part

Observed path, its continuous part, and its jump part, for the introduction.
The top panel is the sum of the other two, so the jump part is the only
difference between the path we observe and the continuous path below it.

Generated: 2026-09-09T21:09:42+00:00

## Parameters

script: experiments/thesis/intro_paths.py
n: 4000
seed: 8
jump_seed: 193
x0: 0.0
heston: HestonParams
  mu: 0.05
  kappa: 1.5
  theta: 0.14
  xi: 0.6
  rho: -0.2
  v0: 0.14
jumps: JumpParams
  intensity: 6.0
  size_sampler: NormalJumpSizes
    mean: 0.0
    std: 0.045
  min_jumps: 6
  min_peak_magnitude: 0.0

## Data

Raw output: [intro_paths.json](intro_paths.json)
