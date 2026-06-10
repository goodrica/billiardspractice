from __future__ import annotations

import cv2
import numpy as np

from .config import Ball, Shot, TableConfig, Point


def _px(point: Point, homography: np.ndarray) -> tuple[int, int]:
    px = cv2.perspectiveTransform(np.array([[[point[0], point[1]]]], dtype=np.float32), homography)[0][0]
    return tuple(int(round(v)) for v in px)


def _draw_text(canvas: np.ndarray, point: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    cv2.putText(canvas, text, (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def render_projector_overlay(
    table: TableConfig,
    shot: Shot,
    image_size: tuple[int, int],
    table_to_projector: np.ndarray,
    detected_balls: list[Ball] | None = None,
) -> np.ndarray:
    """Render a dark projector overlay for the recommended shot."""
    width, height = image_size
    overlay = np.zeros((height, width, 4), dtype=np.uint8)

    # Draw all detected balls as small context dots.
    for ball in detected_balls or []:
        x, y = _px((ball.x, ball.y), table_to_projector)
        cv2.circle(overlay, (x, y), 6, (180, 180, 180, 150), -1)

    # Draw object-to-pocket path.
    p1 = _px(shot.object_path[0], table_to_projector)
    p2 = _px(shot.object_path[1], table_to_projector)
    cv2.line(overlay, p1, p2, (0, 220, 255, 170), 4)

    # Draw cue-to-ghost-ball aiming line. This is the main practice aid.
    c1 = _px(shot.cue_path[0], table_to_projector)
    c2 = _px(shot.cue_path[1], table_to_projector)
    cv2.line(overlay, c1, c2, (0, 255, 0, 230), 6)
    cv2.line(overlay, c1, c2, (255, 255, 255, 255), 2)

    # Draw ghost ball, contact point, and pocket.
    ghost = _px((shot.ghost_ball.x, shot.ghost_ball.y), table_to_projector)
    contact = _px(shot.contact_point, table_to_projector)
    pocket = _px((shot.pocket.x, shot.pocket.y), table_to_projector)
    cv2.circle(overlay, ghost, 12, (0, 255, 0, 230), -1)
    cv2.circle(overlay, contact, 7, (255, 255, 0, 230), -1)
    cv2.circle(overlay, pocket, 16, (0, 180, 255, 230), -1)

    _draw_text(overlay, c2, f"{shot.pocket.name}  cut {shot.cut_angle_degrees:.1f}°", (0, 255, 0))
    return overlay


def blend_overlay(frame_bgr: np.ndarray, overlay_bgra: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """Alpha-blend a BGRA projector overlay onto a BGR camera frame."""
    rgb = overlay_bgra[:, :, :3]
    alpha_map = (overlay_bgra[:, :, 3] / 255.0 * alpha).astype(np.float32)
    alpha_map = alpha_map[:, :, None]
    blended = frame_bgr.astype(np.float32) * (1.0 - alpha_map) + rgb.astype(np.float32) * alpha_map
    return np.clip(blended, 0, 255).astype(np.uint8)


def render_table_grid(table: TableConfig, image_size: tuple[int, int], table_to_projector: np.ndarray) -> np.ndarray:
    """Render a calibration/debug grid over the table."""
    width, height = image_size
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    for x in np.linspace(0, table.length, 12):
        a = _px((x, 0), table_to_projector)
        b = _px((x, table.width), table_to_projector)
        cv2.line(overlay, a, b, (120, 120, 120, 80), 1)
    for y in np.linspace(0, table.width, 7):
        a = _px((0, y), table_to_projector)
        b = _px((table.length, y), table_to_projector)
        cv2.line(overlay, a, b, (120, 120, 120, 80), 1)
    return overlay
