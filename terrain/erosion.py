"""Erosion post-processing for heightmaps (pure NumPy).

Both algorithms come in two flavours: a generator (``*_steps``) that yields
between chunks of work so the UI can stay responsive, and a plain function that
runs to completion. Inputs are never modified; outputs are normalised to [0, 1].
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from .noise.common import normalize


def hydraulic_erosion_steps(
    heightmap: np.ndarray,
    seed: int = 0,
    droplets: Optional[int] = None,
    max_lifetime: int = 30,
    inertia: float = 0.05,
    capacity: float = 4.0,
    min_capacity: float = 0.01,
    deposit_rate: float = 0.3,
    erode_rate: float = 0.3,
    evaporate_rate: float = 0.02,
    gravity: float = 4.0,
    batch: int = 2048,
) -> Iterator[np.ndarray]:
    """Droplet-based hydraulic erosion, simulated for a batch of droplets at a time.

    Each droplet starts at a random position, rolls downhill (with some inertia),
    picks up sediment while it is fast and its carrying capacity is not reached,
    and deposits sediment when it slows down, climbs or evaporates. All droplets
    of a batch are advanced together with NumPy. Yields the work-in-progress map
    after every batch; the last yielded array is the normalised result.
    """
    h = np.array(heightmap, dtype=np.float64, copy=True)
    rows, cols = h.shape
    if rows < 3 or cols < 3:
        yield normalize(h)
        return
    # Heights are 0..1 over the whole map; express slopes per cell relative to a
    # 256-cell map so the parameters behave similarly at every resolution.
    vertical = 256.0 / max(rows, cols)
    h *= 1.0 / vertical
    flat = h.ravel()
    total = int(rows * cols * 0.6) if droplets is None else int(droplets)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    done = 0
    while done < total:
        n = min(batch, total - done)
        px = rng.uniform(0, cols - 1, n)
        py = rng.uniform(0, rows - 1, n)
        dx = np.zeros(n)
        dy = np.zeros(n)
        speed = np.ones(n)
        water = np.ones(n)
        sediment = np.zeros(n)
        alive = np.arange(n)
        for _ in range(max_lifetime):
            if alive.size == 0:
                break
            x, y = px[alive], py[alive]
            ix = np.clip(x.astype(np.int64), 0, cols - 2)
            iy = np.clip(y.astype(np.int64), 0, rows - 2)
            fx, fy = x - ix, y - iy
            i00 = iy * cols + ix
            h00, h10, h01, h11 = flat[i00], flat[i00 + 1], flat[i00 + cols], flat[i00 + cols + 1]
            grad_x = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
            grad_y = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
            height = (h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy)
                      + h01 * (1 - fx) * fy + h11 * fx * fy)

            ndx = dx[alive] * inertia - grad_x * (1 - inertia)
            ndy = dy[alive] * inertia - grad_y * (1 - inertia)
            length = np.hypot(ndx, ndy)
            stuck = length < 1e-12
            length[stuck] = 1.0
            ndx /= length
            ndy /= length
            nx, ny = x + ndx, y + ndy
            out = stuck | (nx < 0) | (nx >= cols - 1) | (ny < 0) | (ny >= rows - 1)

            jx = np.clip(nx.astype(np.int64), 0, cols - 2)
            jy = np.clip(ny.astype(np.int64), 0, rows - 2)
            gx, gy = nx - jx, ny - jy
            j00 = jy * cols + jx
            new_height = (flat[j00] * (1 - gx) * (1 - gy) + flat[j00 + 1] * gx * (1 - gy)
                          + flat[j00 + cols] * (1 - gx) * gy + flat[j00 + cols + 1] * gx * gy)
            dh = new_height - height

            sp, wa, se = speed[alive], water[alive], sediment[alive]
            cap = np.maximum(-dh * sp * wa * capacity, min_capacity)
            depositing = (se > cap) | (dh > 0)
            dep = np.where(dh > 0, np.minimum(dh, se), (se - cap) * deposit_rate)
            ero = np.minimum((cap - se) * erode_rate, np.maximum(-dh, 0.0))
            change = np.where(depositing, dep, -ero)  # > 0 adds material to the terrain
            change[out] = 0.0

            weights = (change * (1 - fx) * (1 - fy), change * fx * (1 - fy),
                       change * (1 - fx) * fy, change * fx * fy)
            index = np.concatenate([i00, i00 + 1, i00 + cols, i00 + cols + 1])
            flat += np.bincount(index, weights=np.concatenate(weights), minlength=flat.size)

            sediment[alive] = se - change
            speed[alive] = np.sqrt(np.maximum(sp * sp - dh * gravity, 0.0))
            water[alive] = wa * (1 - evaporate_rate)
            dx[alive], dy[alive] = ndx, ndy
            px[alive], py[alive] = nx, ny
            alive = alive[~out]
        done += n
        yield normalize(h)
    if total <= 0:
        yield normalize(h)


def hydraulic_erosion(heightmap: np.ndarray, seed: int = 0, **kwargs) -> np.ndarray:
    """Run :func:`hydraulic_erosion_steps` to completion and return the eroded map."""
    result = None
    for result in hydraulic_erosion_steps(heightmap, seed=seed, **kwargs):
        pass
    return result


def thermal_erosion_steps(
    heightmap: np.ndarray,
    iterations: int = 50,
    talus: float = 2.5,
    rate: float = 0.5,
    chunk: int = 10,
    normalized: bool = True,
) -> Iterator[np.ndarray]:
    """Thermal erosion: material slides down slopes steeper than the talus angle.

    ``talus`` is the critical slope in "map heights per map width" (resolution
    independent). Each iteration moves ``rate`` of half the excess height of a
    cell to its lower 4-neighbours, proportionally to how far each exceeds the
    talus slope. Mass is conserved; borders are closed. Yields every ``chunk``
    iterations (normalised to [0, 1] unless ``normalized`` is False).
    """
    h = np.array(heightmap, dtype=np.float64, copy=True)
    rows, cols = h.shape
    limit = talus / max(rows, cols)
    shifts = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for it in range(max(0, int(iterations))):
        padded = np.pad(h, 1, mode="edge")
        diffs = [h - padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols] for dr, dc in shifts]
        excess = [np.maximum(d - limit, 0.0) for d in diffs]
        total = excess[0] + excess[1] + excess[2] + excess[3]
        move = rate * 0.5 * np.maximum(np.maximum.reduce(diffs) - limit, 0.0)
        safe = np.where(total > 0, total, 1.0)
        out = np.zeros_like(h)
        incoming = np.zeros((rows + 2, cols + 2))
        for (dr, dc), ex in zip(shifts, excess):
            flow = move * ex / safe
            out += flow
            incoming[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols] += flow
        h += incoming[1:-1, 1:-1] - out
        if (it + 1) % chunk == 0:
            yield normalize(h) if normalized else h.copy()
    yield normalize(h) if normalized else h.copy()


def thermal_erosion(heightmap: np.ndarray, **kwargs) -> np.ndarray:
    """Run :func:`thermal_erosion_steps` to completion and return the eroded map."""
    result = None
    for result in thermal_erosion_steps(heightmap, **kwargs):
        pass
    return result
