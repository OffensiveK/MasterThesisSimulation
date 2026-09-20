# Flagging rules: sensitivity to studentization error

How much each flagging rule's size moves when the studentization is slightly wrong.
A threshold rule's statistic shifts by about (c-1)*2log n against a gap rule's (c-1)*q, so this is
what decides between the two families. Design in notes/flagging_experiments.md.

Generated: 2026-08-10T22:01:21+00:00

## Parameters

script: experiments/supporting/flagging_robustness.py
n_values: [1000, 4000, 16000]
n_paths: 4000
n_null: 8000
scale_factors: [0.9, 0.94, 0.97, 1.0, 1.03, 1.06, 1.1]
gammas: [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
level: 0.05
seed: 0
batch_paths: 500
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04

## Data

Raw output: [robustness.json](robustness.json)
