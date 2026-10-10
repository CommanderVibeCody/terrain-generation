"""Render the README screenshots headlessly: ``python tools/make_screenshot.py``."""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pygame  # noqa: E402

from terrain.app import App  # noqa: E402
from terrain.rendering import RENDER_MODES  # noqa: E402


def shoot(app: App, out: str) -> None:
    app.finish_generation()
    app.draw(flip=False)
    pygame.image.save(app.screen, out)
    print(f"saved {out}")


def main(out_dir: str = "docs", seed: int = 1337) -> None:
    os.makedirs(out_dir, exist_ok=True)
    app = App()
    app.set_params(seed=seed)
    shoot(app, os.path.join(out_dir, "screenshot-grid.png"))

    app.render_index = RENDER_MODES.index("Grayscale")
    app.cycle_render_mode(0)
    shoot(app, os.path.join(out_dir, "screenshot-grayscale.png"))

    app.render_index = RENDER_MODES.index("Biomes + hillshade")
    app.select_algorithm(7)  # domain warping
    app.set_params(hydraulic=True)
    app.selected_row = 7
    shoot(app, os.path.join(out_dir, "screenshot-single.png"))
    pygame.quit()


if __name__ == "__main__":
    main()
