"""Draw or redraw zones on a reference frame.

    python tools/zone_editor.py front_entrance
    python tools/zone_editor.py pos --at 1200        # grab the frame at t=1200s

Left-click  add a point          Enter/n  finish the current shape
u           undo last point      d        delete the last finished shape
t           cycle shape type     s        save to the camera's zone JSON
l           shape is a LINE      q        quit without saving

The file it writes is exactly the config the pipeline reads, so nothing else
needs to change after editing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_settings  # noqa: E402
from src.overlay import ZONE_COLOR  # noqa: E402

TYPES = ["table", "queue", "counter", "entrance", "beverage", "exit", "other", "mask"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("camera", help="camera name as in settings.yaml")
    ap.add_argument("--at", type=float, default=1200.0,
                    help="seconds into the video to grab the reference frame")
    ap.add_argument("--config", default="config/settings.yaml")
    args = ap.parse_args()

    settings = load_settings(args.config)
    cam = next((c for c in settings.cameras if c.name == args.camera), None)
    if cam is None:
        print(f"Unknown camera '{args.camera}'. Known: "
              f"{[c.name for c in settings.cameras]}")
        return 1

    cap = cv2.VideoCapture(str(cam.video))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.at * 1000)
    ok, base = cap.read()
    cap.release()
    if not ok:
        print(f"Could not read a frame from {cam.video}")
        return 1

    zone_path = ROOT / "config" / "zones" / f"{cam.name}.json"
    existing = json.loads(zone_path.read_text(encoding="utf-8"))

    shapes: list[dict] = []           # newly drawn shapes
    current: list[list[int]] = []
    type_idx = 0
    is_line = False
    win = f"zone editor - {cam.label}"
    scale = 0.7

    def on_mouse(event, x, y, flags, _):
        nonlocal current
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([int(x / scale), int(y / scale)])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        img = base.copy()
        # existing config, dimmed
        for z in cam.zones:
            cv2.polylines(img, [z.points], True, (90, 90, 90), 1)
        for m in cam.masks:
            cv2.polylines(img, [m.points], True, (60, 60, 60), 1)
        for ln in cam.lines:
            cv2.line(img, tuple(ln.points[0].astype(int)),
                     tuple(ln.points[1].astype(int)), (0, 0, 120), 2)
        # new shapes
        for sh in shapes:
            pts = np.asarray(sh["points"], np.int32)
            col = ZONE_COLOR.get(sh["type"], (200, 200, 200))
            if sh.get("line"):
                cv2.line(img, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 3)
            else:
                cv2.polylines(img, [pts], True, col, 2)
            cv2.putText(img, sh["id"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, col, 2)
        if current:
            pts = np.asarray(current, np.int32)
            cv2.polylines(img, [pts], False, (0, 255, 255), 2)
            for p in pts:
                cv2.circle(img, tuple(p), 4, (0, 255, 255), -1)

        kind = "LINE" if is_line else TYPES[type_idx].upper()
        cv2.putText(img, f"type: {kind}   points: {len(current)}   shapes: {len(shapes)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img, "click=add  enter=finish  u=undo  d=drop  t=type  l=line  s=save  q=quit",
                    (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(win, cv2.resize(img, None, fx=scale, fy=scale))
        k = cv2.waitKey(20) & 0xFF

        if k == ord("q"):
            break
        if k == ord("u") and current:
            current.pop()
        if k == ord("t"):
            type_idx = (type_idx + 1) % len(TYPES)
        if k == ord("l"):
            is_line = not is_line
        if k == ord("d") and shapes:
            shapes.pop()
        if k in (13, ord("n")) and len(current) >= 2:
            t = "line" if is_line else TYPES[type_idx]
            shapes.append({
                "id": f"{t}_{len([s for s in shapes if s['type'] == t]) + 1}",
                "type": t,
                "label": f"{t.replace('_',' ').title()} "
                         f"{len([s for s in shapes if s['type'] == t]) + 1}",
                "points": current[:2] if is_line else current[:],
                "line": is_line,
            })
            current = []
        if k == ord("s"):
            for sh in shapes:
                entry = {"id": sh["id"], "label": sh["label"], "points": sh["points"]}
                if sh.get("line"):
                    entry["in_side"] = "right"
                    existing.setdefault("lines", []).append(entry)
                elif sh["type"] == "mask":
                    existing.setdefault("masks", []).append(entry)
                else:
                    entry["type"] = sh["type"]
                    entry["occupancy"] = True
                    existing.setdefault("polygons", []).append(entry)
            zone_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            print(f"appended {len(shapes)} shape(s) to {zone_path}")
            shapes = []

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
