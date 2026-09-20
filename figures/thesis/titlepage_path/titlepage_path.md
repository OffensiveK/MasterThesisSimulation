# Title-page ornament: a finely sampled Heston path with one jump

The title-page ornament: one finely sampled Heston path, cut by a single jump.

The thesis opens every section with a rule beneath its heading. On the title page that rule is
replaced by the object the thesis is about, so the page states its subject before a word of it is
read. This script is what draws it.

The Heston parameters are deliberately the ones intro_paths.py uses for the introduction's figure.
The title page and the first figure of the thesis then show the same kind of object, and the
ornament is a real sample path of the model the thesis works with rather than a decorative
squiggle that happens to resemble one. What makes it read as a path and not as a line is the
sampling density: the texture has to be fine and even across the whole width, which is what a
diffusion looks like, and that comes from the number of observations rather than from cranking up
the volatility. A large vol-of-vol relative to the variance level buys regime swings instead, and
those read as a mountain range.

Two knobs decide how the picture reads, and they pull against each other. The path is rescaled
into the unit box, so the volatility level itself changes nothing: only the shape of the particular
sample path matters. Fine jitter is worth about 1/sqrt(n) of the box height, so fewer observations
make a visibly noisier line at the cost of a coarser, more angular one. The slow meander competes
with that jitter for the same vertical budget, since a path with a large low-frequency excursion
has a larger range to divide the jitter by. The seed is picked by maximising the jitter subject to
the meander staying above a floor, which is the only way to get both.

Exactly one jump, placed two thirds along, so the single discontinuity is unambiguous and there is
nothing else in the picture that could be mistaken for one.

The output is a drop-in replacement for sections/titlepage_path.tex in the thesis repository. The
coordinates are frozen into that file rather than computed at compile time, so that the title page
is reproducible without a Python toolchain, and this script is the record of where they came from.

Generated: 2026-09-12T01:13:49+00:00

## Parameters

script: experiments/thesis/titlepage_path.py
n: 900
seed: 99
jump_time: 0.666667
jump_index: 600
jump_fraction: 0.425
jump_size: 0.136286
window: 60
heston: HestonParams
  mu: 0.05
  kappa: 1.5
  theta: 0.14
  xi: 0.6
  rho: -0.2
  v0: 0.14
loudest_over_median_scale: 1.313
