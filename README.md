# Billiards Practice

Long-term camera + projector practice aid for a pool/billiards table.

This initial working software is an OpenCV MVP that proves the core loop without needing the real camera or projector:

1. Build a synthetic top-down table image.
2. Detect table geometry and balls.
3. Compute the best pocket and shot line from cue ball to object ball.
4. Render a projector-style overlay with the recommended line.

## Run the demo

```bash
python -m billiardspractice.cli synthetic-demo --output demo_output.png
```

The demo writes `demo_output.png` with the detected balls, selected pocket, and recommended shot line.

## Run tests

```bash
python -m unittest discover -v
```

## Current MVP scope

- Table coordinate system and pocket geometry.
- Cue/object ball shot calculation.
- Best-pocket selection by cut angle.
- Synthetic image generation for repeatable testing.
- OpenCV overlay rendering.

## Planned next milestones

1. **Camera capture**: read from USB/UVC camera or video file.
2. **Table calibration**: mark the four playing-surface corners and save the camera homography.
3. **Projector calibration**: map table coordinates to projector pixels.
4. **Lighting-tolerant detection**: color/shape models for balls, pockets, rails, and cue stick.
5. **Real-time loop**: capture -> detect -> recommend -> project overlay.
