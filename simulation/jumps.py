"""
Compound-Poisson jump generation, decoupled from the continuous path simulator.

A homogeneous Poisson process on [0, 1] can be sampled by drawing only the
total count N ~ Poisson(intensity), then placing N jump times as i.i.d.
Uniform(0, 1) draws (conditional on N, Poisson arrival times have the same
law as sorted uniforms)
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# (distribution, count) -> samples
Sampler = Callable[[np.random.Generator, int], np.ndarray]

def normal_jump_sizes(mean=0.0, std=0.03) -> Sampler:
    return lambda rng, size: rng.normal(mean, std, size)

def uniform_jump_sizes(low=-0.05, high=0.05) -> Sampler:
    return lambda rng, size: rng.uniform(low, high, size)

@dataclass
class JumpParams:
    intensity: float = 5.0
    size_sampler: Sampler = field(default_factory=normal_jump_sizes)

@dataclass
class JumpPattern:
    n: int
    grid_indices: np.ndarray
    sizes: np.ndarray


def simulate_jump_pattern(n, params, rng=None):
    rng = rng or np.random.default_rng()
    count = rng.poisson(params.intensity)
    times = np.sort(rng.uniform(0.0, 1.0, count))
    grid_indices = np.minimum((times * n).astype(int), n - 1)
    sizes = params.size_sampler(rng, count)
    return JumpPattern(n=n, grid_indices=grid_indices, sizes=sizes)


def jump_increments(pattern: JumpPattern) -> np.ndarray:
    increments = np.zeros(pattern.n)
    np.add.at(increments, pattern.grid_indices, pattern.sizes)
    return increments
