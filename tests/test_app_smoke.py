"""Headless smoke tests for the Pygame front-end (uses SDL's dummy video driver)."""

import os

import pygame
import pytest

from terrain import app as app_module
from terrain.app import GRID, SINGLE, SINGLE_RES, TILE_RES, App
from terrain.generator import ALGORITHMS
from terrain.params import Params
from terrain.rendering import RENDER_MODES


@pytest.fixture
def app(tmp_path):
    assert os.environ.get("SDL_VIDEODRIVER") == "dummy"
    application = App(screenshot_dir=str(tmp_path / "shots"))
    yield application
    pygame.quit()


def press(app, key, mod=0):
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod, unicode=""))


def click(app, pos, button=1):
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=button))


def run_frames(app, n):
    for _ in range(n):
        app.step_frame()


def test_main_loop_generates_all_tiles(app):
    run_frames(app, 400)
    assert app.running and app.idle
    assert all((a.key, TILE_RES) in app.tiles for a in ALGORITHMS)
    # Every tile used the same parameters (and therefore the same seed).
    assert {app.tiles[(a.key, TILE_RES)].key[-1] for a in ALGORITHMS} == {app.params.seed}


def test_run_loop_exits_on_quit_event(app, monkeypatch):
    monkeypatch.setattr(app_module, "FPS", 1000)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app.run()  # returns because QUIT stops the loop
    assert not app.running


def test_frames_stay_responsive(app):
    # A single update never blocks for long, even with erosion enabled.
    app.set_params(hydraulic=True, thermal=True)
    import time
    worst = 0.0
    while not app.idle:
        start = time.perf_counter()
        app.update()
        worst = max(worst, time.perf_counter() - start)
    assert worst < 1.0


@pytest.mark.parametrize("index", range(len(ALGORITHMS)))
def test_single_view_for_each_algorithm(app, index):
    press(app, pygame.K_1 + index)
    assert app.view == SINGLE and app.algo_index == index
    app.finish_generation()
    tile = app.tiles[(ALGORITHMS[index].key, SINGLE_RES)]
    assert tile.result.heightmap.shape == (SINGLE_RES, SINGLE_RES)
    run_frames(app, 2)
    press(app, pygame.K_TAB)
    assert app.view == GRID


def test_every_key(app):
    keys = [
        pygame.K_TAB, pygame.K_TAB, pygame.K_PAGEDOWN, pygame.K_PAGEUP, pygame.K_PERIOD,
        pygame.K_COMMA, pygame.K_UP, pygame.K_DOWN, pygame.K_RIGHT, pygame.K_LEFT,
        pygame.K_PLUS, pygame.K_MINUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_KP_MINUS,
        pygame.K_SPACE, pygame.K_r, pygame.K_b, pygame.K_v, pygame.K_h, pygame.K_t,
        pygame.K_p, pygame.K_BACKSPACE, pygame.K_KP1, pygame.K_a, pygame.K_F1,
    ] + [pygame.K_1 + i for i in range(len(ALGORITHMS))]
    for key in keys:
        for mod in (0, pygame.KMOD_SHIFT):
            press(app, key, mod)
            run_frames(app, 1)
    assert app.running
    assert app.last_saved and os.path.exists(app.last_saved)
    press(app, pygame.K_ESCAPE)  # single view -> grid
    assert app.view == GRID and app.running
    press(app, pygame.K_ESCAPE)  # grid -> quit
    assert not app.running


def test_parameter_rows_with_keys(app):
    defaults = Params()
    for row in range(len(app.rows)):
        app.selected_row = row
        press(app, pygame.K_RIGHT)
        press(app, pygame.K_RIGHT, pygame.KMOD_SHIFT)
    p = app.params
    assert p.seed == defaults.seed + 101
    assert p.base != defaults.base
    assert p.frequency > defaults.frequency
    assert p.octaves == defaults.octaves + 2
    assert p.persistence > defaults.persistence
    assert p.lacunarity > defaults.lacunarity
    assert p.warp_strength > defaults.warp_strength
    assert p.hydraulic is False and p.thermal is False  # toggled twice
    assert app.render_mode == RENDER_MODES[(RENDER_MODES.index("Biomes + hillshade") + 2) % len(RENDER_MODES)]
    press(app, pygame.K_BACKSPACE)
    assert app.params == Params(seed=p.seed)


def test_toggles_and_regeneration(app):
    app.finish_generation()
    old = app.tiles[("fbm", TILE_RES)].result.heightmap.copy()
    press(app, pygame.K_h)
    assert app.params.hydraulic and not app.idle
    app.finish_generation()
    eroded = app.tiles[("fbm", TILE_RES)].result
    assert eroded.erosion_ms > 0 and not (eroded.heightmap == old).all()
    press(app, pygame.K_t)
    assert app.params.thermal
    app.finish_generation()


def test_warp_change_only_regenerates_warp_tile(app):
    app.finish_generation()
    app.set_params(warp_strength=2.0)
    pending = [a.key for a, _ in app.pending_targets()]
    assert pending == ["domain_warp"]


def test_render_mode_change_needs_no_regeneration(app):
    app.finish_generation()
    for _ in range(len(RENDER_MODES)):
        press(app, pygame.K_v)
        assert app.idle
        assert all(t.render_mode == app.render_mode for t in app.tiles.values())
        run_frames(app, 1)


def test_param_change_during_generation_restarts(app):
    app.update(budget_ms=0.0)
    press(app, pygame.K_SPACE)
    app.finish_generation()
    seeds = {app.tiles[(a.key, TILE_RES)].key[-1] for a in ALGORITHMS}
    assert seeds == {app.params.seed}


def test_mouse_buttons_tiles_and_wheel(app):
    app.finish_generation()
    run_frames(app, 1)
    # Clicking a tile opens it in single view; clicking the map again goes back.
    rect = app.tile_rects()[4]
    click(app, rect.center)
    assert app.view == SINGLE and app.algo_index == 4
    click(app, (100, 100))
    assert app.view == GRID
    # Every panel button can be clicked.
    for button in list(app.buttons):
        click(app, button.rect.center)
        run_frames(app, 1)
    assert app.running
    # Clicking a parameter label selects the row, the wheel changes it.
    x = app_module.VIEW_W + 40
    y = app.row_top + 3 * app.row_h + 5
    click(app, (x, y))
    assert app.selected_row == 3
    octaves = app.params.octaves
    pygame.mouse.set_pos((x, y))
    app.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1, flipped=False))
    if app.row_at(pygame.mouse.get_pos()) == 3:  # dummy driver may ignore set_pos
        assert app.params.octaves == octaves - 1


def test_save_screenshot(app):
    app.finish_generation()
    path = app.save_screenshot()
    assert os.path.exists(path) and os.path.getsize(path) > 1000
