from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .config import Ball, TableConfig
from .geometry import choose_best_pocket
from .overlay import blend_overlay, render_projector_overlay
from .vision import detect_balls_hough, detect_table_homography, make_synthetic_frame


def _write_json(path: Path | None, data: dict) -> None:
    if path is None:
        return
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def synthetic_demo(output: Path, summary: Path | None = None) -> dict:
    table = TableConfig()
    cue = Ball(520.0, 300.0, table.ball_radius, "white")
    obj = Ball(760.0, 300.0, table.ball_radius, "yellow")
    frame, camera_homography = make_synthetic_frame(table, image_size=(1280, 720), cue_ball=(cue.x, cue.y), object_ball=(obj.x, obj.y))

    shot = choose_best_pocket(cue, obj, table.pockets, table)
    detected = detect_balls_hough(frame, table, camera_homography)

    # For the demo, assume the projector is roughly aligned like the camera.
    projector_homography = camera_homography
    overlay = render_projector_overlay(table, shot, (frame.shape[1], frame.shape[0]), projector_homography, detected)
    blended = blend_overlay(frame, overlay)

    cv2.imwrite(str(output), blended)
    data = {
        "output": str(output),
        "selected_pocket": shot.pocket.name,
        "cut_angle_degrees": round(shot.cut_angle_degrees, 3),
        "ghost_ball": [round(shot.ghost_ball.x, 3), round(shot.ghost_ball.y, 3)],
        "detected_balls": [[round(ball.x, 3), round(ball.y, 3), ball.color] for ball in detected],
    }
    _write_json(summary, data)
    return data


def process_image(image_path: Path, output: Path, summary: Path | None = None, cue: tuple[float, float] | None = None, obj: tuple[float, float] | None = None) -> dict:
    table = TableConfig()
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    homography = detect_table_homography(frame, table)
    detected = detect_balls_hough(frame, table, homography)

    if cue is None or obj is None:
        if len(detected) < 2:
            raise ValueError("At least two balls must be detected, or pass --cue-ball and --object-ball in table coordinates.")
        cue_ball = detected[0]
        object_ball = detected[1]
    else:
        cue_ball = Ball(cue[0], cue[1], table.ball_radius, "white")
        object_ball = Ball(obj[0], obj[1], table.ball_radius, "yellow")

    shot = choose_best_pocket(cue_ball, object_ball, table.pockets, table)
    overlay = render_projector_overlay(table, shot, (frame.shape[1], frame.shape[0]), homography, detected)
    blended = blend_overlay(frame, overlay)
    cv2.imwrite(str(output), blended)

    data = {
        "output": str(output),
        "selected_pocket": shot.pocket.name,
        "cut_angle_degrees": round(shot.cut_angle_degrees, 3),
        "detected_balls": [[round(ball.x, 3), round(ball.y, 3), ball.color] for ball in detected],
    }
    _write_json(summary, data)
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Billiards practice camera/projector MVP")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("synthetic-demo", help="Run the synthetic camera/projector demo")
    demo.add_argument("--output", default="demo_output.png", type=Path)
    demo.add_argument("--summary", type=Path)

    image = sub.add_parser("process-image", help="Process a still image and render a shot overlay")
    image.add_argument("--image", required=True, type=Path)
    image.add_argument("--output", default="processed_output.png", type=Path)
    image.add_argument("--summary", type=Path)
    image.add_argument("--cue-ball", type=float, nargs=2, metavar=("X", "Y"), help="Cue ball position in table coordinates (mm)")
    image.add_argument("--object-ball", type=float, nargs=2, metavar=("X", "Y"), help="Object ball position in table coordinates (mm)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "synthetic-demo":
        data = synthetic_demo(args.output, args.summary)
    elif args.command == "process-image":
        data = process_image(args.image, args.output, args.summary, tuple(args.cue_ball) if args.cue_ball else None, tuple(args.object_ball) if args.object_ball else None)
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
