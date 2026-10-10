"""Pygame front-end: grid comparison view, single view, parameter panel."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import pygame

from . import ui
from .generator import ALGORITHMS, Algorithm, Result, generation_steps
from .params import BASE_NAMES, Params
from .rendering import RENDER_MODES, render

# ----------------------------------------------------------------------------- layout
VIEW_W, HEIGHT = 800, 840
PANEL_W = 340
WIDTH = VIEW_W + PANEL_W
FPS = 60

GRID_COLS = 3
TILE_RES = 120            # heightmap resolution of a grid tile
TILE_PX = 240             # on-screen size of a grid tile
LABEL_H = 24
SINGLE_RES = 384          # heightmap resolution in single view
SINGLE_PX = 768

FRAME_BUDGET_MS = 25.0    # generation work per frame, keeps the window responsive

GRID, SINGLE = "grid", "single"


@dataclass
class TileState:
    key: tuple                      # cache key the heightmap was generated for
    result: Result
    surface: pygame.Surface
    render_mode: str


@dataclass
class ParamRow:
    name: str        # attribute used by the algorithms (for dimming unused rows)
    label: str
    value: Callable[[], str]
    adjust: Callable[[int, bool], None]   # (direction, coarse)


def cache_key(algorithm: Algorithm, params: Params) -> tuple:
    """Only the parameters an algorithm actually uses (plus erosion) affect its output."""
    return (algorithm.key,) + tuple(getattr(params, name) for name in algorithm.uses) \
        + (params.hydraulic, params.thermal, params.seed)


class App:
    def __init__(self, screenshot_dir: str = "screenshots") -> None:
        pygame.init()
        pygame.display.set_caption("Terrain Generation - algorithm comparison")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.fonts = ui.Fonts()
        self.running = True
        self.screenshot_dir = screenshot_dir

        self.params = Params()
        self.view = GRID
        self.algo_index = 0
        self.render_index = RENDER_MODES.index("Biomes + hillshade")
        self.selected_row = 0

        self.tiles: dict[tuple[str, int], TileState] = {}
        self.job: Optional[Iterator[Optional[Result]]] = None
        self.job_target: Optional[tuple[str, int]] = None
        self.job_key: Optional[tuple] = None
        self.last_saved: Optional[str] = None

        self.rows = self._build_rows()
        self._build_panel()

    # ------------------------------------------------------------------ state helpers
    @property
    def render_mode(self) -> str:
        return RENDER_MODES[self.render_index]

    @property
    def algorithm(self) -> Algorithm:
        return ALGORITHMS[self.algo_index]

    def set_params(self, **changes) -> None:
        self.params = self.params.with_changes(**changes)

    def _cycle(self, options: tuple, current, direction: int):
        return options[(options.index(current) + direction) % len(options)]

    def _build_rows(self) -> list[ParamRow]:
        p = lambda: self.params  # noqa: E731
        return [
            ParamRow("seed", "Seed", lambda: str(p().seed),
                     lambda d, c: self.set_params(seed=p().seed + d * (100 if c else 1))),
            ParamRow("base", "Base noise", lambda: p().base.capitalize(),
                     lambda d, c: self.set_params(base=self._cycle(BASE_NAMES, p().base, d))),
            ParamRow("frequency", "Frequency", lambda: f"{p().frequency:.2f}",
                     lambda d, c: self.set_params(frequency=p().frequency * (2.0 if c else 1.25) ** d)),
            ParamRow("octaves", "Octaves", lambda: str(p().octaves),
                     lambda d, c: self.set_params(octaves=p().octaves + d)),
            ParamRow("persistence", "Persistence", lambda: f"{p().persistence:.2f}",
                     lambda d, c: self.set_params(persistence=p().persistence + d * (0.1 if c else 0.05))),
            ParamRow("lacunarity", "Lacunarity", lambda: f"{p().lacunarity:.2f}",
                     lambda d, c: self.set_params(lacunarity=p().lacunarity + d * (0.5 if c else 0.1))),
            ParamRow("warp_strength", "Warp strength", lambda: f"{p().warp_strength:.2f}",
                     lambda d, c: self.set_params(warp_strength=p().warp_strength + d * (0.5 if c else 0.1))),
            ParamRow("hydraulic", "Hydraulic erosion", lambda: "on" if p().hydraulic else "off",
                     lambda d, c: self.toggle_hydraulic()),
            ParamRow("thermal", "Thermal erosion", lambda: "on" if p().thermal else "off",
                     lambda d, c: self.toggle_thermal()),
            ParamRow("render", "Render mode", lambda: self.render_mode,
                     lambda d, c: self.cycle_render_mode(d)),
        ]

    def _build_panel(self) -> None:
        x = VIEW_W + 15
        self.row_top = 92
        self.row_h = 27
        self.buttons: list[ui.Button] = []
        for i, row in enumerate(self.rows):
            y = self.row_top + i * self.row_h
            self.buttons.append(ui.Button(pygame.Rect(x + 252, y + 2, 30, self.row_h - 4), "-",
                                          lambda r=row: r.adjust(-1, False), centered=True))
            self.buttons.append(ui.Button(pygame.Rect(x + 286, y + 2, 30, self.row_h - 4), "+",
                                          lambda r=row: r.adjust(1, False), centered=True))
        y = self.row_top + len(self.rows) * self.row_h + 12
        actions = [
            ("New seed (Space)", self.new_seed, lambda: False),
            ("Grid / single (Tab)", self.toggle_view, lambda: self.view == SINGLE),
            ("Reset params (Bksp)", self.reset_params, lambda: False),
            ("Save PNG (P)", self.save_screenshot, lambda: False),
        ]
        for i, (label, action, active) in enumerate(actions):
            rect = pygame.Rect(x + (i % 2) * 158, y + (i // 2) * 34, 152, 28)
            self.buttons.append(ui.Button(rect, label, action, active))
        self.panel_info_y = y + 2 * 34 + 10

    # ------------------------------------------------------------------ actions
    def select_algorithm(self, index: int, single: bool = True) -> None:
        self.algo_index = index % len(ALGORITHMS)
        if single:
            self.view = SINGLE

    def toggle_view(self) -> None:
        self.view = SINGLE if self.view == GRID else GRID

    def new_seed(self) -> None:
        self.set_params(seed=random.randrange(1_000_000))

    def reset_params(self) -> None:
        self.params = Params(seed=self.params.seed)

    def toggle_hydraulic(self) -> None:
        self.set_params(hydraulic=not self.params.hydraulic)

    def toggle_thermal(self) -> None:
        self.set_params(thermal=not self.params.thermal)

    def cycle_base(self, direction: int = 1) -> None:
        self.set_params(base=self._cycle(BASE_NAMES, self.params.base, direction))

    def cycle_render_mode(self, direction: int = 1) -> None:
        self.render_index = (self.render_index + direction) % len(RENDER_MODES)
        for target, tile in self.tiles.items():
            self._render_tile(target, tile.key, tile.result)

    def adjust_selected(self, direction: int, coarse: bool = False) -> None:
        self.rows[self.selected_row].adjust(direction, coarse)

    def save_screenshot(self) -> str:
        os.makedirs(self.screenshot_dir, exist_ok=True)
        name = f"terrain-{self.view}-{self.algorithm.key}-seed{self.params.seed}-{int(time.time())}.png"
        path = os.path.join(self.screenshot_dir, name)
        self.draw(flip=False)
        pygame.image.save(self.screen, path)
        self.last_saved = path
        return path

    # ------------------------------------------------------------------ generation
    def visible_targets(self) -> list[tuple[Algorithm, int]]:
        if self.view == SINGLE:
            return [(self.algorithm, SINGLE_RES)]
        return [(a, TILE_RES) for a in ALGORITHMS]

    def pending_targets(self) -> list[tuple[Algorithm, int]]:
        out = []
        for algorithm, res in self.visible_targets():
            tile = self.tiles.get((algorithm.key, res))
            if tile is None or tile.key != cache_key(algorithm, self.params):
                out.append((algorithm, res))
        return out

    @property
    def idle(self) -> bool:
        return self.job is None and not self.pending_targets()

    def _render_tile(self, target: tuple[str, int], key: tuple, result: Result) -> None:
        surface = ui.array_to_surface(render(result.heightmap, self.render_mode))
        px = SINGLE_PX if target[1] == SINGLE_RES else TILE_PX
        surface = pygame.transform.smoothscale(surface, (px, px))
        self.tiles[target] = TileState(key, result, surface, self.render_mode)

    def update(self, budget_ms: float = FRAME_BUDGET_MS) -> None:
        """Advance background generation for at most ``budget_ms`` (at least one step)."""
        deadline = time.perf_counter() + budget_ms / 1000.0
        while True:
            if self.job is not None:
                # Drop the job if its target is no longer visible or its parameters changed.
                algorithm = next((a for a, r in self.visible_targets()
                                  if (a.key, r) == self.job_target), None)
                if algorithm is None or cache_key(algorithm, self.params) != self.job_key:
                    self.job = None
            if self.job is None:
                pending = self.pending_targets()
                if not pending:
                    return
                algorithm, res = pending[0]
                self.job = generation_steps(algorithm, res, self.params)
                self.job_target = (algorithm.key, res)
                self.job_key = cache_key(algorithm, self.params)
            try:
                result = next(self.job)
            except StopIteration:  # pragma: no cover - generation_steps always yields a Result
                result, self.job = None, None
            if result is not None:
                self._render_tile(self.job_target, self.job_key, result)
                self.job = None
            if time.perf_counter() >= deadline:
                return

    def finish_generation(self, max_steps: int = 100_000) -> None:
        """Run generation until every visible tile is up to date (used by tests/tools)."""
        for _ in range(max_steps):
            if self.idle:
                return
            self.update(budget_ms=1000.0)
        raise RuntimeError("generation did not finish")

    # ------------------------------------------------------------------ input
    def tile_rects(self) -> list[pygame.Rect]:
        gap_x = (VIEW_W - GRID_COLS * TILE_PX) // (GRID_COLS + 1)
        rows = (len(ALGORITHMS) + GRID_COLS - 1) // GRID_COLS
        gap_y = (HEIGHT - rows * (TILE_PX + LABEL_H)) // (rows + 1)
        rects = []
        for i in range(len(ALGORITHMS)):
            c, r = i % GRID_COLS, i // GRID_COLS
            rects.append(pygame.Rect(gap_x + c * (TILE_PX + gap_x),
                                     gap_y + r * (TILE_PX + LABEL_H + gap_y), TILE_PX, TILE_PX))
        return rects

    def row_at(self, pos: tuple[int, int]) -> Optional[int]:
        x, y = pos
        if x < VIEW_W + 10 or x > VIEW_W + 262:
            return None
        index = (y - self.row_top) // self.row_h
        return index if 0 <= index < len(self.rows) and y >= self.row_top else None

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            self._handle_key(event)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in self.buttons:
                if button.handle_click(event.pos):
                    return
            row = self.row_at(event.pos)
            if row is not None:
                self.selected_row = row
                return
            if self.view == GRID:
                for i, rect in enumerate(self.tile_rects()):
                    if rect.inflate(0, LABEL_H).move(0, LABEL_H // 2).collidepoint(event.pos):
                        self.select_algorithm(i)
                        return
            elif event.pos[0] < VIEW_W:
                self.view = GRID
        elif event.type == pygame.MOUSEWHEEL:
            row = self.row_at(pygame.mouse.get_pos())
            if row is not None and event.y:
                self.selected_row = row
                self.adjust_selected(1 if event.y > 0 else -1)

    def _handle_key(self, event: pygame.event.Event) -> None:
        key = event.key
        coarse = bool(getattr(event, "mod", 0) & pygame.KMOD_SHIFT)
        if pygame.K_1 <= key < pygame.K_1 + len(ALGORITHMS):
            self.select_algorithm(key - pygame.K_1)
        elif pygame.K_KP1 <= key < pygame.K_KP1 + len(ALGORITHMS):
            self.select_algorithm(key - pygame.K_KP1)
        elif key == pygame.K_TAB:
            self.toggle_view()
        elif key in (pygame.K_PAGEDOWN, pygame.K_PERIOD):
            self.select_algorithm(self.algo_index + 1, single=self.view == SINGLE)
        elif key in (pygame.K_PAGEUP, pygame.K_COMMA):
            self.select_algorithm(self.algo_index - 1, single=self.view == SINGLE)
        elif key == pygame.K_UP:
            self.selected_row = (self.selected_row - 1) % len(self.rows)
        elif key == pygame.K_DOWN:
            self.selected_row = (self.selected_row + 1) % len(self.rows)
        elif key in (pygame.K_RIGHT, pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.adjust_selected(1, coarse)
        elif key in (pygame.K_LEFT, pygame.K_MINUS, pygame.K_KP_MINUS):
            self.adjust_selected(-1, coarse)
        elif key in (pygame.K_SPACE, pygame.K_r):
            self.new_seed()
        elif key == pygame.K_b:
            self.cycle_base(-1 if coarse else 1)
        elif key == pygame.K_v:
            self.cycle_render_mode(-1 if coarse else 1)
        elif key == pygame.K_h:
            self.toggle_hydraulic()
        elif key == pygame.K_t:
            self.toggle_thermal()
        elif key == pygame.K_p:
            self.save_screenshot()
        elif key == pygame.K_BACKSPACE:
            self.reset_params()
        elif key == pygame.K_ESCAPE:
            if self.view == SINGLE:
                self.view = GRID
            else:
                self.running = False

    # ------------------------------------------------------------------ drawing
    def draw(self, flip: bool = True) -> None:
        self.screen.fill(ui.BG)
        if self.view == GRID:
            self._draw_grid_view()
        else:
            self._draw_single_view()
        self._draw_panel()
        if flip:
            pygame.display.flip()

    def _draw_tile(self, algorithm: Algorithm, res: int, rect: pygame.Rect, font: pygame.font.Font,
                   highlight: bool) -> None:
        tile = self.tiles.get((algorithm.key, res))
        up_to_date = tile is not None and tile.key == cache_key(algorithm, self.params)
        if tile is None:
            pygame.draw.rect(self.screen, ui.TILE_BG, rect)
            img = font.render("generating...", True, ui.TEXT_DIM)
            self.screen.blit(img, img.get_rect(center=rect.center))
        else:
            self.screen.blit(tile.surface, rect)
            if not up_to_date:
                badge = font.render(" updating... ", True, (30, 30, 30), ui.BADGE)
                self.screen.blit(badge, (rect.x + 6, rect.y + 6))
        if highlight:
            pygame.draw.rect(self.screen, ui.ACCENT, rect.inflate(6, 6), 3, border_radius=3)
        y = rect.bottom + 4
        index = ALGORITHMS.index(algorithm) + 1
        ui.draw_text(self.screen, font, f"{index}  {algorithm.name}", (rect.x, y))
        if tile is not None:
            timing = f"{tile.result.elapsed_ms:.0f} ms"
            if tile.result.erosion_ms > 0:
                timing = f"{tile.result.noise_ms:.0f} + {tile.result.erosion_ms:.0f} ms"
            ui.draw_text_right(self.screen, font, timing, rect.right, y,
                               ui.TEXT_DIM if up_to_date else ui.TEXT_FAINT)

    def _draw_grid_view(self) -> None:
        for i, (algorithm, rect) in enumerate(zip(ALGORITHMS, self.tile_rects())):
            self._draw_tile(algorithm, TILE_RES, rect, self.fonts.label, i == self.algo_index)

    def _draw_single_view(self) -> None:
        rect = pygame.Rect((VIEW_W - SINGLE_PX) // 2, 16, SINGLE_PX, SINGLE_PX)
        self._draw_tile(self.algorithm, SINGLE_RES, rect, self.fonts.normal, False)
        ui.draw_text(self.screen, self.fonts.small, self.algorithm.description,
                     (rect.x, rect.bottom + 28), ui.TEXT_DIM)

    def _draw_panel(self) -> None:
        x0 = VIEW_W
        pygame.draw.rect(self.screen, ui.PANEL_BG, (x0, 0, PANEL_W, HEIGHT))
        x = x0 + 15
        ui.draw_text(self.screen, self.fonts.title, "Terrain Generation", (x, 16))
        view = "Grid comparison (same seed)" if self.view == GRID else f"Single: {self.algorithm.name}"
        ui.draw_text(self.screen, self.fonts.small, view, (x, 44), ui.TEXT_DIM)
        ui.draw_text(self.screen, self.fonts.small, "PARAMETERS   (arrow keys / buttons / wheel)",
                     (x, self.row_top - 18), ui.TEXT_DIM)

        used = set(self.algorithm.uses) | {"hydraulic", "thermal", "render"}
        for i, row in enumerate(self.rows):
            y = self.row_top + i * self.row_h
            if i == self.selected_row:
                pygame.draw.rect(self.screen, ui.SELECTED_ROW,
                                 (x - 6, y, PANEL_W - 18, self.row_h), border_radius=5)
            dim = self.view == SINGLE and row.name not in used
            colour = ui.TEXT_FAINT if dim else ui.TEXT
            ui.draw_text(self.screen, self.fonts.normal, row.label, (x, y + 6), colour)
            ui.draw_text_right(self.screen, self.fonts.normal, row.value(), x + 244, y + 6,
                               ui.TEXT_FAINT if dim else ui.ACCENT)

        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            button.draw(self.screen, self.fonts.small, mouse)

        y = self.panel_info_y
        y = ui.draw_text(self.screen, self.fonts.small, "STATUS", (x, y), ui.TEXT_DIM)
        pending = len(self.pending_targets())
        status = "All maps up to date" if pending == 0 else f"Generating... {pending} map(s) left"
        y = ui.draw_text(self.screen, self.fonts.normal, status, (x, y))
        res = SINGLE_RES if self.view == SINGLE else TILE_RES
        tile = self.tiles.get((self.algorithm.key, res))
        y = ui.draw_text(self.screen, self.fonts.normal, f"Selected: {self.algorithm.name}", (x, y))
        if tile is not None:
            line = f"{res}x{res}: noise {tile.result.noise_ms:.1f} ms"
            if tile.result.erosion_ms > 0:
                line += f", erosion {tile.result.erosion_ms:.0f} ms"
            y = ui.draw_text(self.screen, self.fonts.small, line, (x, y), ui.TEXT_DIM)
        uses = [u.replace("_strength", "") for u in self.algorithm.uses if u != "seed"]
        for i in range(0, len(uses), 3):
            prefix = "Uses: " if i == 0 else "      "
            y = ui.draw_text(self.screen, self.fonts.small, prefix + ", ".join(uses[i:i + 3]),
                             (x, y), ui.TEXT_DIM)
        if self.last_saved:
            y = ui.draw_text(self.screen, self.fonts.small, f"Saved {os.path.basename(self.last_saved)}"[:52],
                             (x, y), ui.TEXT_DIM)
        y += 6
        y = ui.draw_text(self.screen, self.fonts.small, "BIOMES (by height)", (x, y), ui.TEXT_DIM)
        y = ui.draw_biome_legend(self.screen, self.fonts.small, x, y)
        y += 4
        hints = [
            "1-9: open algorithm   Tab: grid / single",
            "Click a tile to enlarge, PgUp/PgDn: prev/next",
            "Space: new seed   B: base noise   V: render",
            "H: hydraulic   T: thermal   P: save PNG",
            "Shift = bigger steps   Esc: back / quit",
        ]
        for hint in hints:
            y = ui.draw_text(self.screen, self.fonts.small, hint, (x, y), ui.TEXT_DIM)

    # ------------------------------------------------------------------ main loop
    def step_frame(self) -> None:
        for event in pygame.event.get():
            self.handle_event(event)
        self.update()
        self.draw()

    def run(self) -> None:
        while self.running:
            self.step_frame()
            self.clock.tick(FPS)
        pygame.quit()
