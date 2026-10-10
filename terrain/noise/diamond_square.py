"""Diamond-square (midpoint displacement) fractal heightmaps."""

from __future__ import annotations

import numpy as np

from .common import normalize, resample

#: Internal grid size (2^n + 1). The result is resampled to the requested size, so
#: previews and full-size views of the same seed look identical.
GRID_POWER = 9


def _square_step(h: np.ndarray, rows: range, cols: range, half: int,
                 amp: float, rng: np.random.Generator) -> None:
    """Set each point to the mean of its (up to four) axis neighbours plus noise."""
    size = h.shape[0]
    rr, cc = np.meshgrid(np.array(rows), np.array(cols), indexing="ij")
    total = np.zeros(rr.shape)
    count = np.zeros(rr.shape)
    for dr, dc in ((-half, 0), (half, 0), (0, -half), (0, half)):
        r = rr + dr
        c = cc + dc
        ok = (r >= 0) & (r < size) & (c >= 0) & (c < size)
        total += np.where(ok, h[np.clip(r, 0, size - 1), np.clip(c, 0, size - 1)], 0.0)
        count += ok
    h[rr, cc] = total / count + rng.uniform(-amp, amp, rr.shape)


def diamond_square(width: int, height: int, seed: int = 0, frequency: float = 4.0,
                   roughness: float = 0.5) -> np.ndarray:
    """Heightmap (values in [0, 1]) generated with the diamond-square algorithm.

    ``frequency`` sets how many random seed points the coarsest lattice has across
    the map (rounded to a power of two); ``roughness`` is the factor by which the
    random displacement shrinks at every subdivision level (higher = rougher).
    """
    n = 2 ** GRID_POWER
    size = n + 1
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    cells = int(np.clip(2 ** round(np.log2(max(frequency, 1.0))), 1, n // 2))
    step = n // cells
    h = np.zeros((size, size))
    h[::step, ::step] = rng.uniform(-1.0, 1.0, (cells + 1, cells + 1))
    amp = roughness
    while step > 1:
        half = step // 2
        # Diamond step: centre of every square = mean of its 4 corners + noise.
        corners = (h[0:n:step, 0:n:step] + h[0:n:step, step::step]
                   + h[step::step, 0:n:step] + h[step::step, step::step])
        h[half::step, half::step] = corners / 4.0 + rng.uniform(-amp, amp, corners.shape)
        # Square step: edge midpoints = mean of their diamond neighbours + noise.
        _square_step(h, range(half, size, step), range(0, size, step), half, amp, rng)
        _square_step(h, range(0, size, step), range(half, size, step), half, amp, rng)
        amp *= roughness
        step = half
    side = max(width, height)
    full = resample(h, side, side)
    return normalize(full[:height, :width])
