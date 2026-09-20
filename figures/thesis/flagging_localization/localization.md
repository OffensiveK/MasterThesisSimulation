# Flagging rules: which increments get blamed

Head-to-head of the candidate flagging rules: given a rejection, which increments get blamed.
Every rule runs on the same paths and is size-adjusted on jump-free paths to fire on exactly
`level` of them, so the comparison is not dominated by the spot-volatility plug-in error.

Generated: 2026-08-14T13:33:09+00:00

## Parameters

script: experiments/thesis/flagging_localization.py
n_values: [1000, 4000, 16000]
n_paths: 2000
n_null: 8000
sweep_n: 4000
sweep_multipliers: [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
level: 0.05
seed: 0
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04
candidates: {'exp': 'uppe_K2_sin', 'renyi_single': 'sign_K2_sin', 'renyi_peel': 'sign_full_sin', 'renyi': 'sign_full_bat', 'renyi_deep': 'sign_full_dee'}
scenarios: {'null': 'no jumps', 'single': 'one jump at 2x the critical scale', 'pair_same': 'two equal jumps, same tail', 'pair_opposite': 'two equal jumps, opposite tails', 'cluster5_tight': 'five equal jumps, random signs', 'cluster5_spread': 'five jumps, magnitudes 1.5x-6x', 'many10': 'ten jumps, magnitudes 2x-3x'}

## Data

Raw output: [localization.json](localization.json)
