"""Waiting heatmap: where people actually stand still, drawn on a real frame.

The zone report can only tell you how long people waited inside a polygon you
already drew. This answers the prior question -- where are they standing at all
-- which is how you find the spots worth drawing a zone (or putting a screen)
around in the first place.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .analytics import CameraResult
from .config import CameraConfig


def render_stop_heatmap(frame: np.ndarray, result: CameraResult,
                        cam: CameraConfig, blur: int = 99) -> np.ndarray:
    """Blend a dwell-weighted heatmap of stop locations over `frame`.

    Each stop contributes its duration, so a spot where one person stood for
    five minutes outweighs a spot twenty people walked through and paused in.
    """
    h, w = frame.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)

    for s in result.stops:
        x, y = int(round(s.x)), int(round(s.y))
        if 0 <= x < w and 0 <= y < h:
            acc[y, x] += float(s.duration)

    if acc.max() <= 0:
        return frame.copy()

    k = blur if blur % 2 == 1 else blur + 1
    acc = cv2.GaussianBlur(acc, (k, k), 0)
    acc /= acc.max()

    colored = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    # only tint where there is signal, so the empty floor stays readable
    alpha = np.clip(acc * 2.2, 0, 0.85)[..., None]
    out = (frame * (1 - alpha) + colored * alpha).astype(np.uint8)

    # zone outlines for orientation, then a legend
    for z in cam.zones:
        cv2.polylines(out, [z.points], True, (255, 255, 255), 1, cv2.LINE_AA)

    total = sum(s.duration for s in result.stops)
    unzoned = sum(s.duration for s in result.stops if s.zone_id == "off_zone")
    panel = out.copy()
    cv2.rectangle(panel, (0, 0), (w, 78), (18, 18, 18), -1)
    cv2.addWeighted(panel, 0.72, out, 0.28, 0, out)
    cv2.putText(out, f"{cam.label} - where people stand still",
                (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out,
                f"{len(result.stops)} stops | {total/60:.0f} min of standing | "
                f"{unzoned/60:.0f} min of it outside every zone",
                (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 255, 200), 1, cv2.LINE_AA)
    return out


def write_stop_heatmaps(results: list[CameraResult], cameras: list[CameraConfig],
                        out_dir: Path, at_seconds: float = 1200.0) -> list[Path]:
    """One heatmap per camera, over a reference frame from its own video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_name = {c.name: c for c in cameras}
    written: list[Path] = []

    for r in results:
        cam = by_name.get(r.name)
        if cam is None:
            continue
        cap = cv2.VideoCapture(str(cam.video))
        cap.set(cv2.CAP_PROP_POS_MSEC, at_seconds * 1000)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        path = out_dir / f"heatmap_waiting_{r.name}.jpg"
        cv2.imwrite(str(path), render_stop_heatmap(frame, r, cam))
        written.append(path)
    return written
