"""Perlin ("improved", 2002) gradient noise."""

from __future__ import annotations

import numpy as np

from .common import GRAD8, coordinate_grid, fade, hash2, lerp, normalize, permutation


def perlin2(x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Sample 2D Perlin noise. Output is roughly in [-1, 1]."""
    perm = permutation(seed)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0

    def corner(dx: int, dy: int) -> np.ndarray:
        g = GRAD8[hash2(perm, x0 + dx, y0 + dy) & 7]
        return g[..., 0] * (fx - dx) + g[..., 1] * (fy - dy)

    u = fade(fx)
    v = fade(fy)
    n = lerp(lerp(corner(0, 0), corner(1, 0), u), lerp(corner(0, 1), corner(1, 1), u), v)
    return n * np.sqrt(2.0)


def perlin_noise(width: int, height: int, seed: int = 0, frequency: float = 4.0) -> np.ndarray:
    """Heightmap (values in [0, 1]) of single-octave Perlin noise."""
    x, y = coordinate_grid(width, height, frequency)
    return normalize(perlin2(x, y, seed))
