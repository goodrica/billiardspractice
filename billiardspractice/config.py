from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class Ball:
    x: float
    y: float
    radius: float = 28.575  # 2.25 inch ball diameter in mm
    color: str = "yellow"


@dataclass(frozen=True)
class Pocket:
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class Shot:
    cue_ball: Ball
    object_ball: Ball
    pocket: Pocket
    ghost_ball: Ball
    contact_point: Point
    cue_path: tuple[Point, Point]
    object_path: tuple[Point, Point]
    cut_angle_degrees: float
    pocket_distance: float
    cue_distance: float


@dataclass(frozen=True)
class TableConfig:
    length: float = 2336.8  # 92 inch playing surface
    width: float = 1168.4   # 46 inch playing surface
    ball_radius: float = 28.575
    pockets: tuple[Pocket, ...] = (
        Pocket("top-left", 0.0, 1168.4),
        Pocket("top-center", 1168.4, 1168.4),
        Pocket("top-right", 2336.8, 1168.4),
        Pocket("bottom-left", 0.0, 0.0),
        Pocket("bottom-center", 1168.4, 0.0),
        Pocket("bottom-right", 2336.8, 0.0),
    )

    @property
    def corners(self) -> tuple[Point, Point, Point, Point]:
        return ((0.0, 0.0), (self.length, 0.0), (self.length, self.width), (0.0, self.width))
