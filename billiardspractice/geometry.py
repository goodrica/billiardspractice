from __future__ import annotations

import math
from typing import Iterable

from .config import Ball, Pocket, Shot, TableConfig, Point


def subtract(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def scale(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def normalize(a: Point) -> Point:
    n = norm(a)
    if n == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return (a[0] / n, a[1] / n)


def angle_between_degrees(a: Point, b: Point) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot(a, b) / (na * nb)))
    return math.degrees(math.acos(cosine))


def point_in_table(point: Point | Ball, table: TableConfig, margin: float = 0.0) -> bool:
    if isinstance(point, Ball):
        x, y = point.x, point.y
    else:
        x, y = point
    return (
        -margin <= x <= table.length + margin
        and -margin <= y <= table.width + margin
    )


def compute_shot(
    cue_ball: Ball,
    object_ball: Ball,
    pocket: Pocket,
    table: TableConfig,
) -> Shot:
    """Compute the geometric shot line for a single object ball and pocket.

    The cue ball is aimed at the ghost-ball center. The ghost ball touches the
    object ball along the object-ball-to-pocket line.
    """
    if cue_ball.x == object_ball.x and cue_ball.y == object_ball.y:
        raise ValueError("Cue ball and object ball cannot occupy the same point")

    object_to_pocket = subtract((pocket.x, pocket.y), (object_ball.x, object_ball.y))
    object_to_pocket_unit = normalize(object_to_pocket)

    ghost_center = subtract((object_ball.x, object_ball.y), scale(object_to_pocket_unit, 2 * object_ball.radius))
    contact_point = subtract((object_ball.x, object_ball.y), scale(object_to_pocket_unit, object_ball.radius))

    cue_to_ghost = subtract((ghost_center[0], ghost_center[1]), (cue_ball.x, cue_ball.y))
    cue_to_ghost_unit = normalize(cue_to_ghost)

    cue_path = ((cue_ball.x, cue_ball.y), (ghost_center[0], ghost_center[1]))
    object_path = ((object_ball.x, object_ball.y), (pocket.x, pocket.y))

    cut_angle = angle_between_degrees(cue_to_ghost_unit, object_to_pocket_unit)
    pocket_distance = norm(object_to_pocket)
    cue_distance = norm(cue_to_ghost)

    return Shot(
        cue_ball=cue_ball,
        object_ball=object_ball,
        pocket=pocket,
        ghost_ball=Ball(ghost_center[0], ghost_center[1], object_ball.radius, "ghost"),
        contact_point=contact_point,
        cue_path=cue_path,
        object_path=object_path,
        cut_angle_degrees=cut_angle,
        pocket_distance=pocket_distance,
        cue_distance=cue_distance,
    )


def choose_best_pocket(
    cue_ball: Ball,
    object_ball: Ball,
    pockets: Iterable[Pocket],
    table: TableConfig,
) -> Shot:
    pocket_list = list(pockets)
    if not pocket_list:
        raise ValueError("At least one pocket is required")

    candidates = [compute_shot(cue_ball, object_ball, pocket, table) for pocket in pocket_list]
    valid = [shot for shot in candidates if point_in_table(shot.ghost_ball, table, margin=table.ball_radius)]
    if not valid:
        raise ValueError("No geometrically reachable pocket from the current ball positions")

    return min(valid, key=lambda shot: (shot.cut_angle_degrees, shot.pocket_distance, shot.cue_distance))
