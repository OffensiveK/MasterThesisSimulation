
import numpy as np

from .continous import HestonParams, simulate_continuous_increments, simulate_continuous_increments_batch 
from .jumps import JumpPattern, jump_increments, simulate_jump_pattern

def simulate_log_price_increments(n, heston, jumps=None, rng=None):
    rng = rng or np.random.default_rng()
    continuous = simulate_continuous_increments(n, heston, rng)
    if jumps is None:
        return continuous
    pattern = jumps if isinstance(jumps, JumpPattern) else simulate_jump_pattern(n, jumps, rng)
    return continuous + jump_increments(pattern)
