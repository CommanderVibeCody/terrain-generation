"""Fractal combinations of a base noise: fBm, billow, ridged multifractal, domain warping."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .common import coordinate_grid, normalize
from .perlin import perlin2
from .simplex import simplex2
from .value import value2
from .worley import worley2

Sampler = Callable[[np.ndarray, np.ndarray, int], np.ndarray]

#: Base noises that the fractal functions can be built on. Each returns ~[-1, 1].
BASE_NOISES: dict[str, Sampler] = {
    "perlin": perlin2,
    "simplex": simplex2,
    "value": value2,
    "worley": worley2,
}


def _sampler(base: str) -> Sampler:
    try:
        return BASE_NOISES[base]
    except KeyError:
        raise ValueError(f"unknown base noise {base!r}, choose from {sorted(BASE_NOISES)}") from None


def _octave_seed(seed: int, octave: int) -> int:
    # A different hash table per octave avoids visible alignment between octaves.
    return (seed * 1_000_003 + octave * 7919) & 0x7FFFFFFF


def fbm2(x: np.ndarray, y: np.ndarray, seed: int = 0, octaves: int = 6, persistence: float = 0.5,
         lacunarity: float = 2.0, base: str = "perlin") -> np.ndarray:
    """Fractal Brownian motion: a sum of octaves of ``base`` noise, normalised to ~[-1, 1]."""
    sample = _sampler(base)
    total = np.zeros_like(x, dtype=np.float64)
    amp, freq, norm = 1.0, 1.0, 0.0
    for octave in range(max(1, int(octaves))):
        total += amp * sample(x * freq, y * freq, _octave_seed(seed, octave))
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return total / norm


def fbm(width: int, height: int, seed: int = 0, frequency: float = 4.0, octaves: int = 6,
        persistence: float = 0.5, lacunarity: float = 2.0, base: str = "perlin") -> np.ndarray:
    """Heightmap (values in [0, 1]) of fBm: big shapes from low octaves, detail from high ones."""
    x, y = coordinate_grid(width, height, frequency)
    return normalize(fbm2(x, y, seed, octaves, persistence, lacunarity, base))


def billow(width: int, height: int, seed: int = 0, frequency: float = 4.0, octaves: int = 6,
           persistence: float = 0.5, lacunarity: float = 2.0, base: str = "perlin") -> np.ndarray:
    """Heightmap (values in [0, 1]) of billow noise: fBm of ``|noise|`` (puffy, rounded hills)."""
    sample = _sampler(base)
    x, y = coordinate_grid(width, height, frequency)
    total = np.zeros_like(x)
    amp, freq = 1.0, 1.0
    for octave in range(max(1, int(octaves))):
        total += amp * np.abs(sample(x * freq, y * freq, _octave_seed(seed, octave)))
        amp *= persistence
        freq *= lacunarity
    return normalize(total)


def ridged_multifractal(width: int, height: int, seed: int = 0, frequency: float = 4.0,
                        octaves: int = 6, persistence: float = 0.5, lacunarity: float = 2.0,
                        base: str = "perlin", offset: float = 1.0, gain: float = 2.0) -> np.ndarray:
    """Heightmap (values in [0, 1]) of Musgrave's ridged multifractal.

    Each octave uses ``(offset - |noise|)^2`` which turns the zero crossings of the
    noise into sharp ridges. Higher octaves are weighted by the previous signal,
    so detail accumulates on the ridges while valleys stay smooth.
    """
    sample = _sampler(base)
    x, y = coordinate_grid(width, height, frequency)
    total = np.zeros_like(x)
    weight = np.ones_like(x)
    amp, freq = 1.0, 1.0
    for octave in range(max(1, int(octaves))):
        signal = offset - np.abs(sample(x * freq, y * freq, _octave_seed(seed, octave)))
        signal = signal * signal * weight
        weight = np.clip(signal * gain, 0.0, 1.0)
        total += signal * amp
        amp *= persistence
        freq *= lacunarity
    return normalize(total)


def domain_warp(width: int, height: int, seed: int = 0, frequency: float = 4.0, octaves: int = 6,
                persistence: float = 0.5, lacunarity: float = 2.0, base: str = "perlin",
                warp_strength: float = 1.0) -> np.ndarray:
    """Heightmap (values in [0, 1]) of domain-warped fBm (Inigo Quilez style).

    Two extra fBm fields displace the sample coordinates before the final fBm is
    evaluated: ``h(p) = fbm(p + strength * (fbm(p + a), fbm(p + b)))``.
    """
    x, y = coordinate_grid(width, height, frequency)
    # The warp fields use few octaves so the swirls stay large and readable.
    warp_octaves = min(max(1, int(octaves)), 2)
    qx = fbm2(x + 0.0, y + 0.0, seed + 101, warp_octaves, persistence, lacunarity, base)
    qy = fbm2(x + 5.2, y + 1.3, seed + 202, warp_octaves, persistence, lacunarity, base)
    return normalize(fbm2(x + warp_strength * 2.0 * qx, y + warp_strength * 2.0 * qy,
                          seed, octaves, persistence, lacunarity, base))
