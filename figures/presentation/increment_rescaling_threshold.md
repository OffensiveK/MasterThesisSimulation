# Increment rescaling as n grows (with truncation threshold)

Increment rescaling as n grows, with the truncation threshold u_n marked on the raw panel.

Generated: 2026-08-10T02:30:17+00:00

## Parameters

script: experiments/examples/plot_increment_rescaling_threshold.py
n_values: [100, 500, 2000, 10000]
seed: 1452
scale: 3.0
tau: 0.475
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04
jumps: None
