# Spot-volatility estimator error by location along the path

Where along the path the spot-volatility estimator makes its error, and whether the argmax of
the test statistic sits where that error is largest.

Generated: 2026-09-11T22:02:33+00:00

## Parameters

script: experiments/thesis/error_localization_sweep.py
n_values: [200, 352, 619, 1089, 1916, 3372, 5932, 10437, 18362, 32306, 56838, 100000]
n_paths: 8000
n_boot: 400
seed: 0
heston: HestonParams
  mu: 0.05
  kappa: 2.0
  theta: 0.04
  xi: 0.3
  rho: -0.2
  v0: 0.04
fits_absolute: {'max': {'beta': 0.1634143249104529, 'ci_lo': 0.16151867108468773, 'ci_hi': 0.16522995687495776}, 'random': {'beta': 0.2546100567853824, 'ci_lo': 0.25078823719789234, 'ci_hi': 0.2598843586954731}, 'oracle': {'beta': 0.24540706188824293, 'ci_lo': 0.24110238584158872, 'ci_hi': 0.2496417595694802}, 'hat': {'beta': 0.3146465789970597, 'ci_lo': 0.3112282080064105, 'ci_hi': 0.3182592432463736}}
fits_relative: {'random': {'beta': 0.2521215493003121, 'ci_lo': 0.24856209350178735, 'ci_hi': 0.2558787693898334}, 'oracle': {'beta': 0.246944718935799, 'ci_lo': 0.24331281812836444, 'ci_hi': 0.2503510110401156}, 'hat': {'beta': 0.2825783700382098, 'ci_lo': 0.2798923667801673, 'ci_hi': 0.28520942909056507}}

## Data

Raw output: [localization.json](localization.json)
