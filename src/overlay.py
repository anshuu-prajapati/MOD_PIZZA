"""Minimal debug overlay: zones, lines, boxes, anonymous IDs, movement trails."""

from __future__ import annotations

import cv2
import numpy as np

from .config import CameraConfig
from .geometry import polygon_centroid

ZONE_COLOR = {
    "entrance": (80, 220, 255),
    "queue": (60, 90, 255),
    "counter": (255, 170, 60),
    "table": (110, 230, 130),
    "beverage": (255, 120, 230),
    "exit": (0, 140, 255),
    "other": (170, 170, 170),
}
MASK_COLOR = (70, 70, 70)
TRAIL_COLOR = (0, 255, 255)
BOX_COLOR = (0, 230, 120)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, org, color=(255, 255, 255), scale=0.5, thick=1, bg=(0, 0, 0)):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - h - 5), (x + w + 6, y + 3), bg, -1)
    cv2.putText(img, text, (x + 3, y - 2), FONT, scale, color, thick, cv2.LINE_AA)


def draw_static(frame: np.ndarray, cam: CameraConfig, counts: dict[str, int]) -> np.ndarray:
    """Zones, masks and counting lines."""
    overlay = frame.copy()

    for m in cam.masks:
        cv2.fillPoly(overlay, [m.points], MASK_COLOR)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    for z in cam.zones:
        color = ZONE_COLOR.get(z.type, ZONE_COLOR["other"])
        cv2.polylines(frame, [z.points], True, color, 2, cv2.LINE_AA)
        cx, cy = polygon_centroid(z.points)
        n = counts.get(z.id, 0)
        _label(frame, f"{z.label}: {n}", (cx - 55, cy), color, 0.48, 1)

    for ln in cam.lines:
        p1 = tuple(ln.points[0].astype(int))
        p2 = tuple(ln.points[1].astype(int))
        cv2.line(frame, p1, p2, (0, 0, 255), 3, cv2.LINE_AA)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        # short arrow showing the "IN" direction (normal to the line)
        vx, vy = p2[0] - p1[0], p2[1] - p1[1]
        norm = max(1.0, float(np.hypot(vx, vy)))
        sign = 1 if ln.in_side == "right" else -1
        nx, ny = -vy / norm * 45 * sign, vx / norm * 45 * sign
        cv2.arrowedLine(frame, mid, (int(mid[0] + nx), int(mid[1] + ny)),
                        (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.4)
        _label(frame, f"{ln.label} (IN)", (p1[0], p1[1] - 8), (0, 0, 255), 0.5, 1)

    return frame


def draw_tracks(frame, tracks, trails: dict[int, list], masked_ids: set[int]) -> None:
    for tr in tracks:
        x1, y1, x2, y2 = (int(v) for v in tr.bbox)
        if tr.track_id in masked_ids:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        _label(frame, f"ID {tr.track_id}", (x1, y1 - 2), (255, 255, 255), 0.5, 1,
               bg=(0, 120, 60))
        cv2.circle(frame, (int(tr.foot[0]), int(tr.foot[1])), 4, (0, 0, 255), -1)

    for tid, pts in trails.items():
        if len(pts) < 2:
            continue
        arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [arr], False, TRAIL_COLOR, 2, cv2.LINE_AA)


def draw_hud(frame, *, cam_label, clock, t_s, duration_s, in_view, occupancy,
             entries, queue_len, status, progress) -> None:
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 92), (18, 18, 18), -1)
    cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)

    mm, ss = divmod(int(t_s), 60)
    dmm, dss = divmod(int(duration_s), 60)
    line1 = f"{cam_label}   {clock}   [{mm:02d}:{ss:02d} / {dmm:02d}:{dss:02d}]"
    line2 = (f"in view: {in_view}    in this camera's zones: {occupancy}    "
             f"entries: {entries}    queue: {queue_len}    status: {status}")
    cv2.putText(frame, line1, (14, 32), FONT, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (14, 62), FONT, 0.62, (120, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, "anonymous tracking - no faces, no identity   [q] next  [space] pause  [s] snapshot",
                (14, 84), FONT, 0.44, (170, 170, 170), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 92), (int(w * progress), 96), (0, 200, 255), -1)
