"""Turn heightmaps into RGB images (pure NumPy, no Pygame).

All functions take a ``height x width`` array in [0, 1] and return a
``height x width x 3`` ``uint8`` array.
"""

from __future__ import annotations

import numpy as np

#: Biome bands by height: (upper limit, name, colour at the bottom, colour at the top).
BIOMES = (
    (0.30, "Deep water", (16, 38, 92), (28, 66, 140)),
    (0.42, "Water", (34, 84, 168), (64, 128, 205)),
    (0.46, "Beach", (214, 196, 140), (228, 214, 162)),
    (0.60, "Grass", (96, 160, 64), (72, 136, 50)),
    (0.72, "Forest", (44, 104, 44), (30, 78, 36)),
    (0.86, "Rock", (118, 106, 96), (150, 142, 134)),
    (1.01, "Snow", (226, 230, 236), (252, 252, 255)),
)
SEA_LEVEL = BIOMES[1][0]

RENDER_MODES = ("Grayscale", "Biomes", "Biomes + hillshade", "Grayscale + hillshade")


def grayscale(heightmap: np.ndarray) -> np.ndarray:
    """Plain heightmap: black = low, white = high."""
    g = (np.clip(heightmap, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def biome_colors(heightmap: np.ndarray) -> np.ndarray:
    """Colour each cell by height band (deep water ... snow), with a gradient inside each band."""
    h = np.clip(heightmap, 0.0, 1.0)
    out = np.zeros(h.shape + (3,), dtype=np.float64)
    lower = 0.0
    for upper, _name, low_col, high_col in BIOMES:
        mask = (h >= lower) & (h < upper)
        t = ((h[mask] - lower) / (upper - lower))[:, None]
        out[mask] = np.array(low_col) * (1 - t) + np.array(high_col) * t
        lower = upper
    return out.astype(np.uint8)


def hillshade(heightmap: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0,
              z_factor: float = 6.0, flatten_below: float = 0.0) -> np.ndarray:
    """Lambertian hillshading in [0, 1] (light from ``azimuth`` degrees, NW by default).

    ``z_factor`` exaggerates relief relative to the map width; heights below
    ``flatten_below`` are treated as flat (e.g. a water surface).
    """
    h = np.maximum(np.asarray(heightmap, dtype=np.float64), flatten_below)
    side = max(h.shape)
    dzdy, dzdx = np.gradient(h * side * z_factor / 64.0)
    az = np.radians(azimuth)
    alt = np.radians(altitude)
    lx, ly, lz = np.cos(alt) * np.sin(az), -np.cos(alt) * np.cos(az), np.sin(alt)
    # Surface normal of z = f(x, y) is (-dz/dx, -dz/dy, 1); y grows downwards on screen.
    shade = (-dzdx * lx - dzdy * ly + lz) / np.sqrt(dzdx ** 2 + dzdy ** 2 + 1.0)
    return np.clip(shade, 0.0, 1.0)


def apply_shade(rgb: np.ndarray, shade: np.ndarray, ambient: float = 0.35) -> np.ndarray:
    factor = ambient + (1.0 - ambient) * shade / max(np.sin(np.radians(45.0)), 1e-6)
    return np.clip(rgb.astype(np.float64) * factor[..., None], 0, 255).astype(np.uint8)


def render(heightmap: np.ndarray, mode: str = "Biomes") -> np.ndarray:
    """Render a heightmap in one of :data:`RENDER_MODES`."""
    if mode == "Grayscale":
        return grayscale(heightmap)
    if mode == "Biomes":
        return biome_colors(heightmap)
    if mode == "Biomes + hillshade":
        return apply_shade(biome_colors(heightmap), hillshade(heightmap, flatten_below=SEA_LEVEL))
    if mode == "Grayscale + hillshade":
        return apply_shade(grayscale(heightmap), hillshade(heightmap), ambient=0.5)
    raise ValueError(f"unknown render mode {mode!r}, choose from {RENDER_MODES}")
