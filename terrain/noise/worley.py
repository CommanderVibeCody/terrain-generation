"""Worley (cellular / Voronoi) noise: distance to the nearest random feature point."""

from __future__ import annotations

import numpy as np

from .common import coordinate_grid, hash2, normalize, permutation, random_table


def worley_distances(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (F1, F2): distances to the nearest and second-nearest feature points.

    Every lattice cell holds one feature point at a seeded random offset; checking
    the 3x3 neighbourhood of cells is enough to find the two closest points.
    """
    perm = permutation(seed)
    off_x = random_table(seed, 2)
    off_y = random_table(seed, 3)
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    f1 = np.full(x.shape, np.inf)
    f2 = np.full(x.shape, np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx = ix + dx
            cy = iy + dy
            h = hash2(perm, cx, cy)
            d = np.hypot(cx + off_x[h] - x, cy + off_y[h] - y)
            f2 = np.where(d < f1, f1, np.minimum(f2, d))
            f1 = np.minimum(f1, d)
    return f1, f2


def worley2(x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """F1 Worley noise mapped to roughly [-1, 1] (low at feature points)."""
    f1, _ = worley_distances(x, y, seed)
    return np.clip(f1, 0.0, 1.0) * 2.0 - 1.0


def worley_noise(width: int, height: int, seed: int = 0, frequency: float = 4.0) -> np.ndarray:
    """Heightmap (values in [0, 1]) of F1 Worley noise: cone-shaped cells."""
    x, y = coordinate_grid(width, height, frequency)
    f1, _ = worley_distances(x, y, seed)
    return normalize(f1)
