"""Tests for hydraulic and thermal erosion."""

import numpy as np
import pytest

from terrain import noise
from terrain.erosion import (hydraulic_erosion, hydraulic_erosion_steps, thermal_erosion,
                             thermal_erosion_steps)


@pytest.fixture
def terrain():
    return noise.fbm(64, 48, seed=21)


@pytest.mark.parametrize("erode", [hydraulic_erosion, thermal_erosion])
def test_shape_range_and_input_untouched(terrain, erode):
    before = terrain.copy()
    out = erode(terrain)
    assert out.shape == terrain.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    np.testing.assert_array_equal(terrain, before)
    assert np.abs(out - terrain).mean() > 1e-4  # it actually changed something
    # ... but it is still recognisably the same terrain
    assert np.corrcoef(out.ravel(), terrain.ravel())[0, 1] > 0.8


def test_hydraulic_is_deterministic_by_seed(terrain):
    np.testing.assert_array_equal(hydraulic_erosion(terrain, seed=4), hydraulic_erosion(terrain, seed=4))
    assert not np.array_equal(hydraulic_erosion(terrain, seed=4), hydraulic_erosion(terrain, seed=5))


def test_hydraulic_yields_per_batch(terrain):
    steps = list(hydraulic_erosion_steps(terrain, droplets=1000, batch=250))
    assert len(steps) == 4
    assert all(s.shape == terrain.shape for s in steps)


def test_hydraulic_zero_droplets_and_tiny_maps():
    flat = np.linspace(0, 1, 16).reshape(4, 4)
    np.testing.assert_allclose(hydraulic_erosion(flat, droplets=0), flat)
    assert hydraulic_erosion(np.zeros((2, 2))).shape == (2, 2)


def test_hydraulic_carves_into_slopes():
    # On a smooth cone, hydraulic erosion must roughen the slopes (gullies).
    y, x = np.mgrid[0:64, 0:64]
    cone = 1.0 - np.hypot(x - 32, y - 32) / 45.0
    eroded = hydraulic_erosion(cone, seed=1, droplets=4000)
    lap = lambda h: np.abs(np.diff(h, 2, axis=0)).mean() + np.abs(np.diff(h, 2, axis=1)).mean()  # noqa: E731
    assert lap(eroded) > lap(cone) * 1.5


def test_thermal_reduces_steep_slopes():
    rng = np.random.default_rng(0)
    spiky = rng.random((40, 40))
    # Nothing moves while every slope is below the talus limit.
    np.testing.assert_allclose(thermal_erosion(spiky, talus=1e9, normalized=False), spiky)
    raw = list(thermal_erosion_steps(spiky, iterations=100, talus=1.0, normalized=False))[-1]
    assert np.abs(np.diff(raw)).max() < np.abs(np.diff(spiky)).max()


def test_thermal_conserves_mass_and_yields_in_chunks():
    rng = np.random.default_rng(1)
    h = rng.random((20, 30))
    steps = list(thermal_erosion_steps(h, iterations=25, talus=0.5, chunk=10, normalized=False))
    assert len(steps) == 3  # after 10, after 20, final
    assert steps[-1].sum() == pytest.approx(h.sum())
