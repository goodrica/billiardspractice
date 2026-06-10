from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .config import Ball, Pocket, TableConfig, Point


GREEN_LOWER = np.array([35, 35, 35], dtype=np.uint8)
GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four rectangle corners as bottom-left, bottom-right, top-right, top-left."""
    pts = points.reshape(4, 2).astype(np.float32)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top = y_sorted[:2]
    bottom = y_sorted[2:]
    top_left, top_right = top[np.argsort(top[:, 0])]
    bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]
    return np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float32)


def table_corners_for_image(table: TableConfig, image_size: tuple[int, int], margin: float = 60.0) -> np.ndarray:
    width, height = image_size
    usable_w = max(1.0, width - 2 * margin)
    usable_h = max(1.0, height - 2 * margin)
    left = margin
    top = margin
    right = margin + usable_w
    bottom = margin + usable_h
    return np.array(
        [
            [left, bottom],
            [right, bottom],
            [right, top],
            [left, top],
        ],
        dtype=np.float32,
    )


def homography_from_table(table: TableConfig, image_size: tuple[int, int], margin: float = 60.0) -> np.ndarray:
    """Return the perspective transform from table coordinates to image pixels."""
    table_corners = np.array(table.corners, dtype=np.float32)
    image_corners = table_corners_for_image(table, image_size, margin=margin)
    return cv2.getPerspectiveTransform(table_corners, image_corners)


def make_synthetic_frame(
    table: TableConfig,
    image_size: tuple[int, int] = (1280, 720),
    cue_ball: Point = (520.0, 300.0),
    object_ball: Point = (760.0, 300.0),
    margin: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a repeatable synthetic top-down table image for tests and demos."""
    width, height = image_size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (35, 35, 35)

    image_corners = table_corners_for_image(table, image_size, margin=margin)
    H = cv2.getPerspectiveTransform(np.array(table.corners, dtype=np.float32), image_corners)

    rail_mask = np.zeros((height, width), dtype=np.uint8)
    rail_corners = np.array(
        [
            [max(0, margin - 35), max(0, margin + (height - 2 * margin) / 2 + 35)],
            [min(width, width - margin + 35), max(0, margin + (height - 2 * margin) / 2 + 35)],
            [min(width, width - margin + 35), min(height, margin - 35)],
            [max(0, margin - 35), min(height, margin - 35)],
        ],
        dtype=np.float32,
    )
    cv2.fillConvexPoly(rail_mask, rail_corners.astype(np.int32), 255)
    frame[rail_mask > 0] = (95, 82, 65)

    felt_mask = np.zeros((height, width), dtype=np.uint8)
    felt_corners = image_corners.astype(np.int32)
    cv2.fillConvexPoly(felt_mask, felt_corners, 255)
    frame[felt_mask > 0] = (22, 112, 55)

    # Draw a subtle center line and head string.
    cv2.line(frame, tuple(image_corners[0].astype(int)), tuple(image_corners[2].astype(int)), (35, 135, 70), 1)
    cv2.line(frame, tuple(image_corners[3].astype(int)), tuple(image_corners[1].astype(int)), (35, 135, 70), 1)

    # Draw pockets after felt so they are visible.
    for pocket in table.pockets:
        center_px = cv2.perspectiveTransform(np.array([[[pocket.x, pocket.y]]], dtype=np.float32), H)[0][0]
        radius_px = max(10, int(round(table.ball_radius * 0.75 * (image_corners[1, 0] - image_corners[0, 0]) / table.length)))
        cv2.circle(frame, tuple(center_px.astype(int)), radius_px, (0, 0, 0), -1)

    for label, point, color in [
        ("cue", cue_ball, (245, 245, 245)),
        ("object", object_ball, (40, 190, 235)),
    ]:
        center_px = cv2.perspectiveTransform(np.array([[[point[0], point[1]]]], dtype=np.float32), H)[0][0]
        radius_px = int(round(table.ball_radius * (image_corners[1, 0] - image_corners[0, 0]) / table.length))
        cv2.circle(frame, tuple(center_px.astype(int)), radius_px + 3, (0, 0, 0), -1)
        cv2.circle(frame, tuple(center_px.astype(int)), radius_px, color, -1)

    return frame, H


def detect_table_homography(image_bgr: np.ndarray, table: TableConfig) -> np.ndarray:
    """Detect the felt region and estimate the table-to-image homography.

    This is intentionally simple for the MVP: it thresholds green felt, finds
    the largest contour, and orders its corners to match the table coordinate
    corners. The real system should replace or refine this with calibration
    markers and lighting-aware segmentation.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not detect green felt. Try calibration markers or improved lighting.")
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4:
        raise ValueError("Detected felt contour is too small to estimate a homography.")

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    ordered = order_points(box)
    return cv2.getPerspectiveTransform(np.array(table.corners, dtype=np.float32), ordered)


def _sample_color(image_bgr: np.ndarray, center: Point) -> tuple[int, int, int]:
    x, y = int(round(center[0])), int(round(center[1]))
    h, w = image_bgr.shape[:2]
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    r, g, b = image_bgr[y, x]
    return int(b), int(g), int(r)


def _color_name(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    if r > 180 and g > 180 and b > 180:
        return "white"
    if r > 180 and g > 150 and b < 120:
        return "yellow"
    if r > 150 and g < 120 and b < 120:
        return "red"
    if r < 120 and g > 150 and b > 150:
        return "cyan"
    if r < 120 and g > 150 and b < 120:
        return "green"
    if r < 120 and g < 120 and b > 180:
        return "blue"
    if r > 160 and g < 120 and b > 160:
        return "purple"
    return "unknown"


def detect_balls_hough(
    image_bgr: np.ndarray,
    table: TableConfig,
    homography: np.ndarray,
    min_radius_px: int = 8,
    max_radius_px: int = 45,
) -> list[Ball]:
    """Detect non-black circular balls with OpenCV Hough circles."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=18,
        param1=60,
        param2=28,
        minRadius=min_radius_px,
        maxRadius=max_radius_px,
    )
    if circles is None:
        return []

    balls: list[Ball] = []
    for circle in circles[0, :]:
        x_px, y_px, radius_px = [float(v) for v in circle]
        rgb = _sample_color(image_bgr, (x_px, y_px))
        if rgb[0] < 45 and rgb[1] < 45 and rgb[2] < 45:
            continue  # Skip black pockets.
        table_point = cv2.perspectiveTransform(np.array([[[x_px, y_px]]], dtype=np.float32), homography)[0][0]
        x, y = float(table_point[0]), float(table_point[1])
        if not (0 <= x <= table.length and 0 <= y <= table.width):
            continue
        balls.append(Ball(x=x, y=y, radius=table.ball_radius, color=_color_name(rgb)))

    return balls


def detected_balls_to_table_points(balls: Iterable[Ball]) -> list[Point]:
    return [(ball.x, ball.y) for ball in balls]
