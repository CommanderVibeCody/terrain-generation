"""Tests for the pure heightmap functions, the generator pipeline and rendering."""

import numpy as np
import pytest

from terrain import noise
from terrain.generator import ALGORITHMS, generate, generation_steps, get_algorithm
from terrain.params import BASE_NAMES, Params
from terrain.rendering import RENDER_MODES, hillshade, render

HEIGHTMAPS = {
    "value": noise.value_noise,
    "perlin": noise.perlin_noise,
    "simplex": noise.simplex_noise,
    "worley": noise.worley_noise,
    "diamond_square": noise.diamond_square,
    "fbm": noise.fbm,
    "billow": noise.billow,
    "ridged": noise.ridged_multifractal,
    "domain_warp": noise.domain_warp,
}
SAMPLERS = [noise.value2, noise.perlin2, noise.simplex2, noise.worley2]


@pytest.mark.parametrize("name", HEIGHTMAPS)
@pytest.mark.parametrize("shape", [(64, 64), (48, 80), (33, 17)])
def test_shape_and_range(name, shape):
    width, height = shape
    h = HEIGHTMAPS[name](width, height, seed=5)
    assert h.shape == (height, width)
    assert h.dtype == np.float64
    assert np.isfinite(h).all()
    assert h.min() >= 0.0 and h.max() <= 1.0
    assert h.min() == pytest.approx(0.0) and h.max() == pytest.approx(1.0)


@pytest.mark.parametrize("name", HEIGHTMAPS)
def test_deterministic_by_seed(name):
    a = HEIGHTMAPS[name](64, 64, seed=42)
    b = HEIGHTMAPS[name](64, 64, seed=42)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", HEIGHTMAPS)
def test_different_seeds_differ(name):
    a = HEIGHTMAPS[name](64, 64, seed=1)
    b = HEIGHTMAPS[name](64, 64, seed=2)
    assert np.abs(a - b).mean() > 0.02


@pytest.mark.parametrize("name", ["value", "perlin", "simplex", "worley", "fbm", "ridged"])
def test_frequency_changes_output(name):
    a = HEIGHTMAPS[name](64, 64, seed=3, frequency=2.0)
    b = HEIGHTMAPS[name](64, 64, seed=3, frequency=8.0)
    assert not np.allclose(a, b)


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_samplers_are_bounded_and_continuous(sampler):
    rng = np.random.default_rng(0)
    x = rng.uniform(-50, 50, 20000)
    y = rng.uniform(-50, 50, 20000)
    v = sampler(x, y, 9)
    assert np.isfinite(v).all()
    assert v.min() >= -1.05 and v.max() <= 1.05
    # Tiny steps give tiny changes (no discontinuities at cell borders).
    dv = np.abs(sampler(x + 1e-4, y, 9) - v)
    assert dv.max() < 0.01


def test_perlin_is_zero_on_lattice():
    sampler = noise.perlin2
    i = np.arange(-5, 5, dtype=np.float64)
    x, y = np.meshgrid(i, i)
    np.testing.assert_allclose(sampler(x, y, 3), 0.0, atol=1e-12)


def test_negative_coordinates_work():
    x, y = np.meshgrid(np.linspace(-300, -290, 16), np.linspace(-7, 3, 16))
    for sampler in SAMPLERS:
        assert np.isfinite(sampler(x, y, 1)).all()


def test_same_domain_at_different_resolutions():
    small = noise.perlin_noise(32, 32, seed=7)
    large = noise.perlin_noise(128, 128, seed=7)
    down = large.reshape(32, 4, 32, 4).mean(axis=(1, 3))
    assert np.corrcoef(small.ravel(), down.ravel())[0, 1] > 0.98


@pytest.mark.parametrize("base", BASE_NAMES)
def test_fractals_accept_every_base(base):
    for fn in (noise.fbm, noise.billow, noise.ridged_multifractal, noise.domain_warp):
        h = fn(40, 40, seed=1, octaves=3, base=base)
        assert h.shape == (40, 40) and 0.0 <= h.min() and h.max() <= 1.0


def test_unknown_base_raises():
    with pytest.raises(ValueError):
        noise.fbm(16, 16, base="nope")


def test_octaves_add_detail():
    def roughness(h):
        return np.abs(np.diff(h, axis=1)).mean()

    smooth = noise.fbm(128, 128, seed=4, octaves=1)
    rough = noise.fbm(128, 128, seed=4, octaves=7)
    assert roughness(rough) > roughness(smooth) * 1.3


def test_domain_warp_strength_zero_is_plain_fbm():
    a = noise.domain_warp(64, 64, seed=8, warp_strength=0.0)
    b = noise.fbm(64, 64, seed=8)
    np.testing.assert_allclose(a, b)


def test_diamond_square_roughness():
    smooth = noise.diamond_square(128, 128, seed=2, roughness=0.3)
    rough = noise.diamond_square(128, 128, seed=2, roughness=0.8)
    assert np.abs(np.diff(rough)).mean() > np.abs(np.diff(smooth)).mean()


def test_worley_f2_not_smaller_than_f1():
    x, y = noise.coordinate_grid(50, 50, 5.0)
    f1, f2 = noise.worley_distances(x, y, 3)
    assert (f2 >= f1).all()


def test_params_clamping():
    p = Params().with_changes(octaves=99, persistence=-1, frequency=1000, base="nope", seed=-1)
    assert p.octaves == 10 and p.persistence == 0.05 and p.frequency == 32.0
    assert p.base == BASE_NAMES[0] and 0 <= p.seed < 1_000_000


@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=lambda a: a.key)
def test_registry_generates_valid_maps(algorithm):
    result = generate(algorithm, 48, Params(seed=11, octaves=4))
    assert result.heightmap.shape == (48, 48)
    assert 0.0 <= result.heightmap.min() and result.heightmap.max() <= 1.0
    assert result.elapsed_ms >= 0 and result.erosion_ms == 0


def test_generation_steps_yield_then_result():
    params = Params(seed=3, hydraulic=True, thermal=True)
    items = list(generation_steps(get_algorithm("fbm"), 64, params))
    assert all(item is None for item in items[:-1]) and len(items) > 2
    final = items[-1]
    assert final.heightmap.shape == (64, 64) and final.erosion_ms > 0
    np.testing.assert_allclose(final.heightmap, generate("fbm", 64, params).heightmap)


@pytest.mark.parametrize("mode", RENDER_MODES)
def test_render_modes(mode):
    h = noise.fbm(40, 30, seed=1)
    rgb = render(h, mode)
    assert rgb.shape == (30, 40, 3) and rgb.dtype == np.uint8


def test_render_unknown_mode():
    with pytest.raises(ValueError):
        render(np.zeros((4, 4)), "nope")


def test_hillshade_flat_and_slope():
    flat = hillshade(np.full((16, 16), 0.5))
    np.testing.assert_allclose(flat, np.sin(np.radians(45.0)))
    # Light comes from the north-west: a slope facing west is lit, one facing east is darker.
    ramp_east = np.tile(np.linspace(0, 1, 16), (16, 1))  # rises towards the east, faces west
    assert hillshade(ramp_east).mean() > flat.mean()
    assert hillshade(ramp_east[:, ::-1]).mean() < flat.mean()
