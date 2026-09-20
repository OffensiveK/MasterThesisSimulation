# Empirical size and size-adjusted power by spot-volatility bandwidth exponent

The thesis's bandwidth figure (fig:bandwidth), drawn as one 2x3 figure.
Empirical size above size-adjusted power, one column per test, so the size-optimal and the
detection-optimal gamma can be read off the same column. Drawn from the saved sweep JSON.

Generated: 2026-09-11T22:48:19+00:00

## Parameters

script: experiments/thesis/bandwidth_comparison.py
n_paths: 6000
level: 0.05
gammas: [0.3, 0.4, 0.5, 0.6, 0.7]
size_n: [200, 619, 1916, 5932, 18362, 56838]
power_n: [200, 619, 1916, 5932, 18362]
sources: ['figures\\thesis\\bandwidth_sensitivity_sweep\\bandwidth.json', 'figures\\thesis\\bandwidth_power_sweep\\power.json']
