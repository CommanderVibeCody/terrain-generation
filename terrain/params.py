"""Generation parameters shared by every tile (same seed for a fair comparison)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .noise import BASE_NOISES

BASE_NAMES = tuple(BASE_NOISES)  # ("perlin", "simplex", "value", "worley")


@dataclass(frozen=True)
class Params:
    seed: int = 1337
    frequency: float = 4.0
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0
    warp_strength: float = 1.0
    base: str = "perlin"
    hydraulic: bool = False
    thermal: bool = False

    def with_changes(self, **changes) -> "Params":
        return replace(self, **changes).clamped()

    def clamped(self) -> "Params":
        return replace(
            self,
            seed=int(self.seed) % 1_000_000,
            frequency=round(min(max(self.frequency, 0.5), 32.0), 3),
            octaves=int(min(max(self.octaves, 1), 10)),
            persistence=round(min(max(self.persistence, 0.05), 0.95), 2),
            lacunarity=round(min(max(self.lacunarity, 1.2), 4.0), 2),
            warp_strength=round(min(max(self.warp_strength, 0.0), 4.0), 2),
            base=self.base if self.base in BASE_NAMES else BASE_NAMES[0],
        )
