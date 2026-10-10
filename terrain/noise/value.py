"""Value noise: random values on an integer lattice, smoothly interpolated."""

from __future__ import annotations

import numpy as np

from .common import coordinate_grid, fade, hash2, lerp, normalize, permutation, random_table


def value2(x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Sample 2D value noise at the given coordinates. Output is roughly in [-1, 1]."""
    perm = permutation(seed)
    values = random_table(seed, 1) * 2.0 - 1.0
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    u = fade(x - x0)
    v = fade(y - y0)
    v00 = values[hash2(perm, x0, y0)]
    v10 = values[hash2(perm, x0 + 1, y0)]
    v01 = values[hash2(perm, x0, y0 + 1)]
    v11 = values[hash2(perm, x0 + 1, y0 + 1)]
    return lerp(lerp(v00, v10, u), lerp(v01, v11, u), v)


def value_noise(width: int, height: int, seed: int = 0, frequency: float = 4.0) -> np.ndarray:
    """Heightmap (``height x width``, values in [0, 1]) of single-octave value noise."""
    x, y = coordinate_grid(width, height, frequency)
    return normalize(value2(x, y, seed))
