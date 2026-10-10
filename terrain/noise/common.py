"""Shared helpers for the noise functions: seeded hash tables, grids, normalisation."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Classic 2D gradient set (the x/y parts of Perlin's 12 "grad3" edge vectors).
GRAD3 = np.array(
    [[1, 1], [-1, 1], [1, -1], [-1, -1],
     [1, 0], [-1, 0], [1, 0], [-1, 0],
     [0, 1], [0, -1], [0, 1], [0, -1]],
    dtype=np.float64,
)

# Eight evenly spread gradients used by Perlin noise.
GRAD8 = np.array(
    [[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]],
    dtype=np.float64,
)
GRAD8[:4] /= np.sqrt(2.0)


@lru_cache(maxsize=256)
def permutation(seed: int) -> np.ndarray:
    """Return a seeded permutation of 0..255, doubled to 512 entries (no wrap-around needed)."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    perm = rng.permutation(256).astype(np.int64)
    out = np.concatenate([perm, perm])
    out.setflags(write=False)
    return out


@lru_cache(maxsize=256)
def random_table(seed: int, salt: int, size: int = 256) -> np.ndarray:
    """Return a seeded table of uniform values in [0, 1)."""
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, salt])
    out = rng.random(size)
    out.setflags(write=False)
    return out


def hash2(perm: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Hash integer lattice coordinates to 0..255 using a permutation table."""
    return perm[perm[ix & 255] + (iy & 255)]


def fade(t: np.ndarray) -> np.ndarray:
    """Perlin's quintic smoothstep 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def coordinate_grid(width: int, height: int, frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample coordinates for a ``height x width`` map.

    The map always covers the same domain (``frequency`` lattice cells across the
    longer side), so a small preview tile and a large single view of the same
    algorithm show the same terrain at different resolutions.
    """
    side = max(width, height)
    xs = (np.arange(width) + 0.5) / side * frequency
    ys = (np.arange(height) + 0.5) / side * frequency
    return np.meshgrid(xs, ys)


def normalize(values: np.ndarray) -> np.ndarray:
    """Linearly rescale an array to the range [0, 1] (constant arrays become 0.5)."""
    values = np.asarray(values, dtype=np.float64)
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-12:
        return np.full(values.shape, 0.5)
    return (values - lo) / (hi - lo)


def resample(values: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinearly resample a 2D array to ``height x width``."""
    src_h, src_w = values.shape
    ys = np.linspace(0, src_h - 1, height)
    xs = np.linspace(0, src_w - 1, width)
    y0 = np.clip(np.floor(ys).astype(int), 0, src_h - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, src_w - 2)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    a = values[y0][:, x0]
    b = values[y0][:, x0 + 1]
    c = values[y0 + 1][:, x0]
    d = values[y0 + 1][:, x0 + 1]
    return lerp(lerp(a, b, fx), lerp(c, d, fx), fy)
