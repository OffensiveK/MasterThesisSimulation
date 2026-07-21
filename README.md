# Master Thesis Simulation

Simulation runs and diagram generation for the Masters Thesis

  'Jump Detection in High-Frequency Observations of Itô Semimartingales via Extreme Value Theory'

by Bastian Kraft in 2026.  

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Usage

```bash
python scripts/run_experiment.py --n-values 500 1000 2000 5000 10000 --n-paths 500
```

## Layout

```
simulation/
  model.py        Heston SDE continuous-part simulator
  jumps.py        Compound-Poisson jump patterns 
  estimators.py   Truncated backward-window spot-volatility estimator
  detection.py    Gumbel and Renyi (Deheuvels spacing) test statistics
  experiment.py   Monte Carlo comparison across n
  plotting.py     Diagram generation
scripts/
  run_experiment.py   CLI entry point producing the size/power figures
```

