
from dataclasses import dataclass, field
import numpy as np

@dataclass
class NormalJumpSizes:
    mean: float = 0.0
    std: float = 0.03

    def __call__(self, rng, size):
        return rng.normal(self.mean, self.std, size)

@dataclass
class BimodalJumpSizes:
    mode: float = 0.035
    spread: float = 0.008

    def __call__(self, rng, size):
        magnitude = np.abs(rng.normal(self.mode, self.spread, size))
        sign = rng.choice(np.array([-1.0, 1.0]), size=size)
        return sign * magnitude


@dataclass
class JumpParams:
    intensity: float = 5.0
    size_sampler: callable = field(default_factory=NormalJumpSizes)
    min_jumps: int = 0
    min_peak_magnitude: float = 0.0

@dataclass
class JumpPattern:
    n: int
    grid_indices: np.ndarray
    sizes: np.ndarray


def grid_index(n, t):
    index = np.minimum((np.asarray(t) * n).astype(int), n - 1)
    return int(index) if index.ndim == 0 else index



def _sample_jump_counts_and_sizes(intensity, runs, size_sampler, min_jumps, min_peak_magnitude, rng):
    counts = np.empty(runs, dtype=int)
    sizes_by_path = np.empty(runs, dtype=object)
    pending = np.arange(runs)
    for _ in range(10_000):
        if pending.size == 0:
            break
        k = pending.size
        candidate_counts = rng.poisson(intensity, size=k)
        total = int(candidate_counts.sum())
        candidate_sizes = size_sampler(rng, total) if total > 0 else np.array([])
        chunks = np.split(candidate_sizes, np.cumsum(candidate_counts)[:-1])
        peaks = np.array([np.abs(chunk).max() if chunk.size else 0.0 for chunk in chunks])
        ok = (candidate_counts >= min_jumps) & (peaks >= min_peak_magnitude)
        for idx, count, sizes in zip(pending[ok], candidate_counts[ok], (c for c, o in zip(chunks, ok) if o)):
            counts[idx] = count
            sizes_by_path[idx] = sizes
        pending = pending[~ok]
    else:
        raise ValueError(
            f"could not draw {pending.size} of {runs} jump pattern(s) satisfying min_jumps={min_jumps} "
        )
    return counts, sizes_by_path


def simulate_jump_pattern(n, params, rng):
    counts, sizes_by_path = _sample_jump_counts_and_sizes(
        params.intensity, 1, params.size_sampler, params.min_jumps, params.min_peak_magnitude, rng
    )
    count, sizes = counts[0], sizes_by_path[0]
    times = np.sort(rng.uniform(0.0, 1.0, count))
    return JumpPattern(n=n, grid_indices=grid_index(n, times), sizes=sizes)


def jump_increments(pattern):
    increments = np.zeros(pattern.n)
    np.add.at(increments, pattern.grid_indices, pattern.sizes)
    return increments


def jump_increments_batch(patterns):
    runs = len(patterns)
    n = patterns[0].n
    increments = np.zeros((n, runs))
    counts = [len(p.grid_indices) for p in patterns]
    if sum(counts) == 0:
        return increments

    rows = np.concatenate([p.grid_indices for p in patterns]).astype(int)
    cols = np.repeat(np.arange(runs), counts)
    sizes = np.concatenate([p.sizes for p in patterns])
    np.add.at(increments, (rows, cols), sizes)
    return increments


def simulate_jump_increments_batch(n, runs, params, rng):
    counts, sizes_by_path = _sample_jump_counts_and_sizes(
        params.intensity, runs, params.size_sampler, params.min_jumps, params.min_peak_magnitude, rng
    )
    total = int(counts.sum())
    increments = np.zeros((n, runs))
    if total == 0:
        return increments

    run_ids = np.repeat(np.arange(runs), counts)
    times = rng.uniform(0.0, 1.0, total)
    sizes = np.concatenate([sizes_by_path[i] for i in range(runs) if counts[i] > 0])
    np.add.at(increments, (grid_index(n, times), run_ids), sizes)
    return increments

