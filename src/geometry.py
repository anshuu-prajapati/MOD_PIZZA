"""Pure geometry helpers. Knows nothing about what a zone means."""

from __future__ import annotations

import cv2
import numpy as np


def foot_point(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Bottom-centre of a bbox -- the person's approximate floor contact point."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def line_side(point: tuple[float, float], line: np.ndarray) -> int:
    """Sign of the cross product of A->B and A->P. +1 right, -1 left, 0 on the line."""
    (ax, ay), (bx, by) = line
    cross = (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def segments_intersect(p1, p2, p3, p4) -> bool:
    """True if segment p1-p2 crosses segment p3-p4."""

    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if v == 0 else (1 if v > 0 else -1)

    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    return o1 != o2 and o3 != o4


def crossing_direction(prev_pt, curr_pt, line: np.ndarray, in_side: str) -> str | None:
    """Return 'in', 'out' or None for a movement prev->curr against a counting line."""
    if not segments_intersect(prev_pt, curr_pt, line[0], line[1]):
        return None
    before, after = line_side(prev_pt, line), line_side(curr_pt, line)
    if before == after or after == 0:
        return None
    in_sign = 1 if in_side == "right" else -1
    return "in" if after == in_sign else "out"


def speed_px_per_s(track_points: list[tuple[float, float, float]], window_s: float = 1.0) -> float:
    """Average speed over the last `window_s` of a (t, x, y) trail."""
    if len(track_points) < 2:
        return 0.0
    t_end = track_points[-1][0]
    window = [p for p in track_points if t_end - p[0] <= window_s]
    if len(window) < 2:
        window = track_points[-2:]
    dist = sum(
        float(np.hypot(b[1] - a[1], b[2] - a[2]))
        for a, b in zip(window, window[1:])
    )
    dt = window[-1][0] - window[0][0]
    return dist / dt if dt > 0 else 0.0


def polygon_centroid(polygon: np.ndarray) -> tuple[int, int]:
    m = cv2.moments(polygon.astype(np.int32))
    if m["m00"] == 0:
        return tuple(polygon.mean(axis=0).astype(int))
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
