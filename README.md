# Master Thesis Simulation

Simulation runs and figure generation for the Master's Thesis

  'Jump Detection in High-Frequency Observations of Itô Semimartingales via Extreme Value Theory'

By Bastian Kraft in 2026.

The thesis is submitted, so this code is now an archive. It exists so that every figure, table and
quoted number in the finished documents can be traced back to the run that produced it, and
re-run:

| | file | what it contains |
|---|---|---|
| thesis | `Kraft_Bastian_Masterarbeit_Final.pdf` | 10 figures, 4 tables |
| slides | `Kraft_Bastian_Vortragsfolien_Seminar_Statistik_SS26.pdf` | 37 slides |
| handout | `Kraft_Bastian_Handout_Seminar_Statistik_SS26.pdf` | 6 pages |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

numpy and matplotlib are the only dependencies.

## Layout

```
simulation/    the mathematics: model, estimators, tests. Pure numpy
runtime/       what a run puts out: figures and reports. No mathematics
experiments/   what question is being asked. One script per question
  thesis/         lands in the thesis (16)
  supporting/     backs a thesis claim, no figure of its own (1)
  presentation/   renders a slide or handout figure (3)
  examples/       illustrative, rendered nowhere (1)
figures/       mirrors experiments/: a script writes into figures/<group>/<script name>/
```

What decides where code goes is what it is *about*, never how many callers it has. `simulation/`
and `runtime/` are independent siblings: neither imports the other, and `experiments/` imports
both.

Reuse alone never promotes anything into `simulation/`: a sweep driver or a scenario sampler is
experiment logic even when several scripts want it, and stays duplicated on purpose. No plot
lives in `simulation/` or `runtime/`, because a figure belongs to the experiment that draws it.

## Running an experiment

Every script is self-contained: it simulates, plots, and writes its own `.png`, `.json` and `.md`
report side by side.

```bash
python experiments/thesis/oracle_size_sweep.py --n-values 500 1000 2000 --n-paths 500
python experiments/presentation/increment_rescaling.py --with-jumps
```

`--n-values`, `--n-paths`, `--level`, `--seed`, `--out` and `--from-saved` come from
`runtime.experiment.parser()` and mean the same everywhere. Anything else is the script's own.
`--from-saved` redraws from the last run's `.json` without simulating again.

`bandwidth_comparison.py` is the one script that simulates nothing. It plots what the two
bandwidth sweeps measured, so those have to have been run first.

## What produced what

Keyed by the number **printed in the PDF**, not the LaTeX label behind it. Thesis pages are as
printed. The front matter is unnumbered, so a viewer's page counter reads 8 higher.

### Thesis figures

| fig. | page | produced by |
|---|---|---|
| 1 | 1 | `thesis/intro_paths.py` |
| 2 | 54 | `thesis/size_power_comparison.py` |
| 3 | 56 | `thesis/oracle_size_sweep.py` |
| 4 | 58 | `thesis/test_statistic_inflation_sweep.py` |
| 5 | 58 | `thesis/error_localization_sweep.py` |
| 6 | 59 | `thesis/bandwidth_comparison.py`, drawn from `bandwidth_sensitivity_sweep.py` (size panel) and `bandwidth_power_sweep.py` (power panel) |
| 7 | 61 | `thesis/single_spike_power_sweep.py` |
| 8 | 63 | `thesis/truncation_crossing_sweep.py` |
| 9 | 64 | `thesis/single_spike_collapse_sweep.py` |
| 10 | 65 | `thesis/two_spike_sweep.py` |

### Thesis tables

| tab. | page | produced by |
|---|---|---|
| 1 | 52 | *(none, the simulation parameters are written by hand)* |
| 2 | 56 | `thesis/test_statistic_inflation_sweep.py` + `thesis/oracle_size_sweep.py` |
| 3 | 57 | `thesis/error_localization_sweep.py` |
| 4 | 66 | `thesis/flagging_localization.py` |

The titlepage path is `thesis/titlepage_path.py`, which writes TikZ rather than an image.

### Quoted in the prose only

No figure or table points at these, which makes them the easiest to delete by accident.

| section | page | what is quoted | produced by |
|---|---|---|---|
| 6.2 | 54 | size rates 48.70 % and 29.37 % at n = 200 | `thesis/size_power_comparison.py` |
| 6.4 | 61 | power along the path: 23.3 %, 41.7 %, 43.3 % | `thesis/spike_location_sweep.py` |
| 6.4 | 62 | 56.6 % oracle against 42.6 % feasible at n = 1 916 | `thesis/oracle_alternative_sweep.py` |
| 6.5 | 67 | Gumbel threshold at 14.5 / 18.5 / 25.3 % against 5.9 to 7.8 % | `supporting/flagging_robustness.py` |

### Slides and handout

Beamer does not number figures, so these go by slide number and frame title.

| where | produced by |
|---|---|
| slide 9, A Simulated Example | `presentation/single_path_example.py` |
| slide 12, Normalizing Without Truncation | `presentation/increment_rescaling.py --with-jumps` |
| slide 13, Why Truncate? | `presentation/increment_rescaling_threshold.py --with-jumps` |
| handout Figure 2, p. 2 | `presentation/increment_rescaling_threshold.py --with-jumps` |

The `--with-jumps` flag matters: each of those scripts draws two pictures and the jump-carrying
one is used. Handout Figure 1 is TikZ inside the handout and comes from nowhere here.
`experiments/examples/jump_example.py` renders in neither document. It is the worked reference
for the file shape every script follows.


