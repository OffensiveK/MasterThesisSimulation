# Jump pattern / continuous path / sum decomposition example

Generate an example figure: jump pattern, continuous path, and their sum.

Generated: 2026-08-10T02:30:02+00:00

## Parameters

script: experiments/examples/plot_jump_example.py
n: 2000
seed: 1
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04
jumps: JumpParams
  intensity: 8.0
  size_sampler: NormalJumpSizes
    mean: 0.0
    std: 0.03
  min_jumps: 0
  min_peak_magnitude: 0.0
