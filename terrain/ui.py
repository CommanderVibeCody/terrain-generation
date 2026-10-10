"""Pygame UI helpers: colours, fonts, buttons, text and the biome legend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pygame

from .rendering import BIOMES

# ----------------------------------------------------------------------------- colours
BG = (22, 24, 30)
PANEL_BG = (32, 35, 45)
TILE_BG = (40, 44, 56)
TEXT = (230, 232, 240)
TEXT_DIM = (150, 155, 170)
TEXT_FAINT = (95, 100, 115)
ACCENT = (98, 132, 255)
SELECTED_ROW = (48, 54, 74)
BUTTON = (52, 57, 72)
BUTTON_HOVER = (68, 74, 94)
BADGE = (255, 196, 64)


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    action: Callable[[], None]
    is_active: Callable[[], bool] = field(default=lambda: False)
    centered: bool = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, mouse: tuple[int, int]) -> None:
        if self.is_active():
            colour = ACCENT
        elif self.rect.collidepoint(mouse):
            colour = BUTTON_HOVER
        else:
            colour = BUTTON
        pygame.draw.rect(surface, colour, self.rect, border_radius=6)
        text = font.render(self.label, True, TEXT)
        if self.centered:
            surface.blit(text, text.get_rect(center=self.rect.center))
        else:
            surface.blit(text, text.get_rect(midleft=(self.rect.x + 10, self.rect.centery)))

    def handle_click(self, pos: tuple[int, int]) -> bool:
        if self.rect.collidepoint(pos):
            self.action()
            return True
        return False


class Fonts:
    def __init__(self) -> None:
        self.title = pygame.font.Font(None, 30)
        self.normal = pygame.font.Font(None, 22)
        self.small = pygame.font.Font(None, 19)
        self.label = pygame.font.Font(None, 20)


def draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str, pos: tuple[int, int],
              colour: tuple[int, int, int] = TEXT) -> int:
    """Blit ``text`` at ``pos`` and return the y coordinate below it."""
    img = font.render(text, True, colour)
    surface.blit(img, pos)
    return pos[1] + img.get_height() + 4


def draw_text_right(surface: pygame.Surface, font: pygame.font.Font, text: str,
                    right: int, y: int, colour: tuple[int, int, int] = TEXT) -> None:
    img = font.render(text, True, colour)
    surface.blit(img, img.get_rect(topright=(right, y)))


def array_to_surface(rgb: np.ndarray) -> pygame.Surface:
    """Convert a ``height x width x 3`` uint8 array into a Pygame surface."""
    return pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))


def draw_biome_legend(surface: pygame.Surface, font: pygame.font.Font, x: int, y: int) -> int:
    for i, (_upper, name, low, high) in enumerate(BIOMES):
        cx = x + (i % 2) * 150
        cy = y + (i // 2) * 20
        colour = tuple((a + b) // 2 for a, b in zip(low, high))
        pygame.draw.rect(surface, colour, (cx, cy + 2, 13, 13), border_radius=3)
        surface.blit(font.render(name, True, TEXT_DIM), (cx + 19, cy + 1))
    return y + ((len(BIOMES) + 1) // 2) * 20 + 4
