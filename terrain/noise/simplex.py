"""2D simplex noise (triangular lattice, Perlin 2001 / Gustavson's formulation)."""

from __future__ import annotations

import numpy as np

from .common import GRAD3, coordinate_grid, normalize, permutation

F2 = 0.5 * (np.sqrt(3.0) - 1.0)
G2 = (3.0 - np.sqrt(3.0)) / 6.0


def simplex2(x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Sample 2D simplex noise. Output is roughly in [-1, 1]."""
    perm = permutation(seed)
    # Skew the input space to find the simplex (triangle) cell.
    s = (x + y) * F2
    i = np.floor(x + s).astype(np.int64)
    j = np.floor(y + s).astype(np.int64)
    t = (i + j) * G2
    x0 = x - (i - t)
    y0 = y - (j - t)
    # Which of the two triangles of the skewed square are we in?
    i1 = (x0 > y0).astype(np.int64)
    j1 = 1 - i1
    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2

    ii = i & 255
    jj = j & 255
    g0 = GRAD3[perm[ii + perm[jj]] % 12]
    g1 = GRAD3[perm[ii + i1 + perm[jj + j1]] % 12]
    g2 = GRAD3[perm[ii + 1 + perm[jj + 1]] % 12]

    def contribution(g: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        falloff = np.maximum(0.5 - dx * dx - dy * dy, 0.0)
        return falloff ** 4 * (g[..., 0] * dx + g[..., 1] * dy)

    return 70.0 * (contribution(g0, x0, y0) + contribution(g1, x1, y1) + contribution(g2, x2, y2))


def simplex_noise(width: int, height: int, seed: int = 0, frequency: float = 4.0) -> np.ndarray:
    """Heightmap (values in [0, 1]) of single-octave simplex noise."""
    x, y = coordinate_grid(width, height, frequency)
    return normalize(simplex2(x, y, seed))
