# Empirical size across n, oracle N(0,1) vectors

Empirical size of the Gumbel, Exp and Renyi tests across n, on oracle i.i.d. N(0,1) vectors.
Isolates the extreme-value limit-law approximation error from the spot-volatility plug-in error,
which the full-pipeline sweeps necessarily conflate. See notes/oracle_sweep_task.md.

Generated: 2026-08-10T22:16:46+00:00

## Parameters

script: experiments/thesis/oracle_size_sweep.py
n_values: [200, 352, 619, 1089, 1916, 3372, 5932, 10437, 18362, 32306, 56838, 100000]
n_paths: 25000
level: 0.05
seed: 0

## Data

Raw output: [size.json](size.json)
