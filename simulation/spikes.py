
import numpy as np

from .jumps import JumpPattern, grid_index

def critical_jump_scale(heston, n):
    return np.sqrt(heston.theta) * np.sqrt(2 * np.log(n) / n)


def naive_jump_scale(heston, n):
    return np.sqrt(heston.theta / n)


def signed_spike_pattern(n, size, location=0.5, sign=1.0):
    index = grid_index(n, location)
    return JumpPattern(n=n, grid_indices=np.array([index]), sizes=np.array([sign * size]))


def two_spike_pattern(n, size, ratio, separation, location=0.5, same_sign=True):
    first = grid_index(n, location)
    second = min(max(first + int(separation), 0), n - 1)
    second_sign = 1.0 if same_sign else -1.0
    return JumpPattern(
        n=n,
        grid_indices=np.array([first, second]),
        sizes=np.array([size, second_sign * ratio * size]),
    )
