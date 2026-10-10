"""Algorithm registry and the (incremental) generation pipeline: noise -> erosion."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

from . import noise
from .erosion import hydraulic_erosion_steps, thermal_erosion_steps
from .params import Params


@dataclass(frozen=True)
class Algorithm:
    key: str
    name: str
    build: Callable[[int, Params], np.ndarray]  # (resolution, params) -> heightmap
    uses: tuple[str, ...]                        # parameter names that affect the result
    description: str


def _fractal_kwargs(p: Params) -> dict:
    return dict(seed=p.seed, frequency=p.frequency, octaves=p.octaves,
                persistence=p.persistence, lacunarity=p.lacunarity, base=p.base)


_BASIC = ("seed", "frequency")
_FRACTAL = ("seed", "frequency", "octaves", "persistence", "lacunarity", "base")

ALGORITHMS: tuple[Algorithm, ...] = (
    Algorithm("value", "Value Noise",
              lambda r, p: noise.value_noise(r, r, p.seed, p.frequency), _BASIC,
              "Random values on a grid, smoothly interpolated. Blocky."),
    Algorithm("perlin", "Perlin Noise",
              lambda r, p: noise.perlin_noise(r, r, p.seed, p.frequency), _BASIC,
              "Random gradients on a grid. The classic."),
    Algorithm("simplex", "Simplex Noise",
              lambda r, p: noise.simplex_noise(r, r, p.seed, p.frequency), _BASIC,
              "Gradients on a triangular grid. Fewer axis artefacts."),
    Algorithm("worley", "Worley Noise",
              lambda r, p: noise.worley_noise(r, r, p.seed, p.frequency), _BASIC,
              "Distance to the nearest random point. Cells."),
    Algorithm("diamond_square", "Diamond-Square",
              lambda r, p: noise.diamond_square(r, r, p.seed, p.frequency, p.persistence),
              ("seed", "frequency", "persistence"),
              "Recursive midpoint displacement. Roughness = persistence."),
    Algorithm("fbm", "fBm",
              lambda r, p: noise.fbm(r, r, **_fractal_kwargs(p)), _FRACTAL,
              "Octaves of the base noise summed: hills plus detail."),
    Algorithm("ridged", "Ridged Multifractal",
              lambda r, p: noise.ridged_multifractal(r, r, **_fractal_kwargs(p)), _FRACTAL,
              "(1 - |noise|)^2 octaves: sharp mountain ridges."),
    Algorithm("domain_warp", "Domain Warping",
              lambda r, p: noise.domain_warp(r, r, warp_strength=p.warp_strength,
                                             **_fractal_kwargs(p)),
              _FRACTAL + ("warp_strength",),
              "fBm sampled at coordinates displaced by fBm. Swirly."),
    Algorithm("billow", "Billow",
              lambda r, p: noise.billow(r, r, **_fractal_kwargs(p)), _FRACTAL,
              "fBm of |noise|: puffy, rounded hills."),
)

ALGORITHM_KEYS = tuple(a.key for a in ALGORITHMS)


def get_algorithm(key: str) -> Algorithm:
    for algorithm in ALGORITHMS:
        if algorithm.key == key:
            return algorithm
    raise KeyError(key)


@dataclass
class Result:
    heightmap: np.ndarray
    elapsed_ms: float   # pure compute time (noise + erosion)
    noise_ms: float
    erosion_ms: float


def generation_steps(algorithm: Algorithm, resolution: int, params: Params) -> Iterator[Optional[Result]]:
    """Generate a heightmap step by step.

    Yields ``None`` between chunks of work (so a UI can draw a frame) and finally
    yields a :class:`Result`. Time spent outside the generator is not counted.
    """
    start = time.perf_counter()
    heightmap = algorithm.build(resolution, params)
    noise_ms = (time.perf_counter() - start) * 1000.0
    erosion_ms = 0.0
    stages = []
    if params.hydraulic:
        stages.append(lambda h: hydraulic_erosion_steps(h, seed=params.seed))
    if params.thermal:
        stages.append(lambda h: thermal_erosion_steps(h))
    for make_stage in stages:
        yield None
        steps = make_stage(heightmap)
        while True:
            t0 = time.perf_counter()
            try:
                heightmap = next(steps)
            except StopIteration:
                erosion_ms += (time.perf_counter() - t0) * 1000.0
                break
            erosion_ms += (time.perf_counter() - t0) * 1000.0
            yield None
    yield Result(heightmap, noise_ms + erosion_ms, noise_ms, erosion_ms)


def generate(algorithm: Algorithm | str, resolution: int, params: Params) -> Result:
    """Generate a heightmap in one go (noise plus optional erosion)."""
    if isinstance(algorithm, str):
        algorithm = get_algorithm(algorithm)
    result = None
    for result in generation_steps(algorithm, resolution, params):
        pass
    assert result is not None
    return result
