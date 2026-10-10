"""Pure NumPy noise and terrain functions.

Every ``*_noise`` / heightmap function takes ``(width, height, seed, ...)`` and
returns a ``height x width`` float array normalised to [0, 1]. The lower-level
``*2(x, y, seed)`` samplers evaluate a noise at arbitrary coordinates.
"""

from .common import coordinate_grid, normalize, resample
from .diamond_square import diamond_square
from .fractal import BASE_NOISES, billow, domain_warp, fbm, fbm2, ridged_multifractal
from .perlin import perlin2, perlin_noise
from .simplex import simplex2, simplex_noise
from .value import value2, value_noise
from .worley import worley2, worley_distances, worley_noise

__all__ = [
    "BASE_NOISES", "billow", "coordinate_grid", "diamond_square", "domain_warp", "fbm", "fbm2",
    "normalize", "perlin2", "perlin_noise", "resample", "ridged_multifractal", "simplex2",
    "simplex_noise", "value2", "value_noise", "worley2", "worley_distances", "worley_noise",
]
