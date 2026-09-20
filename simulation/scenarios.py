
from dataclasses import dataclass

import numpy as np

from .batching import chunks
from .continuous import simulate_continuous_increments_batch
from .estimators import studentize_batch
from .evt import empirical_critical, gap_statistics_batch, sqrt_2_log
from .jumps import JumpPattern, jump_increments_batch
from .spikes import critical_jump_scale

#: Batches are capped rather than left to the memory budget alone, because the draw order
#: depends on the batch size and the numbers the thesis quotes were measured at this value.
BATCH_PATHS = 500


@dataclass
class Scenario:
    name: str
    n_jumps: int = 1
    mag_low: float = 2.0
    mag_high: float = 2.0
    sign: str = "random"
    description: str = ""

    @property
    def is_null(self):
        return self.n_jumps == 0


#: No jumps at all, which is what every calibration run draws.
NULL = Scenario("null", n_jumps=0)


def sample_patterns(n, runs, scenario, heston, rng):
    if scenario.is_null:
        return [JumpPattern(n=n, grid_indices=np.array([], dtype=int), sizes=np.array([]))] * runs

    crit = critical_jump_scale(heston, n)
    patterns = []
    for _ in range(runs):
        count = scenario.n_jumps
        grid_indices = rng.choice(n, size=count, replace=False)
        magnitude = rng.uniform(scenario.mag_low, scenario.mag_high, count) * crit
        if scenario.sign == "random":
            magnitude = magnitude * rng.choice(np.array([-1.0, 1.0]), size=count)
        elif scenario.sign == "alternating":
            magnitude = magnitude * np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
        patterns.append(JumpPattern(n=n, grid_indices=np.sort(grid_indices), sizes=magnitude))
    return patterns


def simulate_studentized(n, runs, heston, scenario, rng, k_n_exponent=0.5):
    increments = simulate_continuous_increments_batch(runs, n, heston, rng)
    patterns = sample_patterns(n, runs, scenario, heston, rng)
    if not scenario.is_null:
        increments = increments + jump_increments_batch(patterns)
    return studentize_batch(increments, gamma=k_n_exponent), patterns


def size_adjusted_calibration(n, heston, rules, level, n_paths, rng, chunk=BATCH_PATHS,
                              k_n_exponent=0.5):
    scale = sqrt_2_log(n)
    stats = {name: [] for name in rules}
    max_abs = []
    for take in chunks(n_paths, chunk):
        z_hat, _ = simulate_studentized(n, take, heston, NULL, rng, k_n_exponent)
        for name, values in gap_statistics_batch(z_hat, rules, scale).items():
            stats[name].append(values)
        max_abs.append(np.abs(z_hat).max(axis=0))

    criticals = {
        name: empirical_critical(np.concatenate(parts), level) for name, parts in stats.items()
    }
    return criticals, empirical_critical(np.concatenate(max_abs), level)
