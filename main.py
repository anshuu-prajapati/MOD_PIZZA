"""MOD Pizza - anonymous CCTV analytics MVP.

    python main.py

Processes every enabled camera in config/settings.yaml, shows a live overlay
window while it runs, then prints the analytics summary to the terminal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from src.config import ROOT, load_settings
from src.detector import PersonTracker
from src.pipeline import run_camera
from src.report import export_events, print_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MOD Pizza anonymous CCTV analytics")
    p.add_argument("--config", default="config/settings.yaml",
                   help="path to settings.yaml")
    p.add_argument("--camera", action="append",
                   help="only run this camera (repeatable, e.g. --camera pos)")
    p.add_argument("--minutes", type=float,
                   help="process only the first N minutes of each video")
    p.add_argument("--fps", type=float, help="override analysis sample rate")
    p.add_argument("--no-display", action="store_true",
                   help="headless: no window, much faster")
    p.add_argument("--save-video", action="store_true",
                   help="write the annotated video to the output dir")
    p.add_argument("--check-zones", action="store_true",
                   help="render the configured zones on one frame per camera and exit")
    return p.parse_args()


def check_zones(settings) -> int:
    """Draw the zone config on a reference frame so it can be eyeballed."""
    from src import overlay

    out_dir = ROOT / settings.output.get("dir", ".ai/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam in settings.cameras:
        cap = cv2.VideoCapture(str(cam.video))
        cap.set(cv2.CAP_PROP_POS_MSEC, 20 * 60 * 1000)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"  ! could not read a frame from {cam.video}")
            continue
        vis = overlay.draw_static(frame, cam, {z.id: 0 for z in cam.zones})
        path = out_dir / f"zones_{cam.name}.jpg"
        cv2.imwrite(str(path), vis)
        print(f"  wrote {path}")
    return 0


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)

    if args.camera:
        wanted = set(args.camera)
        settings.cameras = [c for c in settings.cameras if c.name in wanted]
    if args.minutes is not None:
        settings.processing["max_minutes"] = args.minutes
    if args.fps is not None:
        settings.processing["sample_fps"] = args.fps
    if args.no_display:
        settings.processing["display"] = False
    if args.save_video:
        settings.processing["save_annotated_video"] = True

    if not settings.cameras:
        print("No enabled cameras in the config.", file=sys.stderr)
        return 1

    for cam in settings.cameras:
        if not cam.video.exists():
            print(f"Missing video for '{cam.name}': {cam.video}", file=sys.stderr)
            return 1

    if args.check_zones:
        return check_zones(settings)

    out_dir = Path(settings.output.get("dir", ".ai/artifacts"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    print("=" * 84)
    print(" MOD PIZZA - CCTV ANALYTICS  (anonymous person analytics, no identity)")
    print("=" * 84)
    tracker = PersonTracker(settings.model)
    print(f" model: {settings.model.get('weights')}  device: {tracker.device}  "
          f"tracker: {tracker.tracker}")

    results = []
    for cam in settings.cameras:
        result, stop = run_camera(cam, settings, tracker, out_dir)
        results.append(result)
        if stop:
            print("\n[Esc] pressed - stopping early and reporting on what was processed.")
            break

    cv2.destroyAllWindows()

    if not results:
        print("Nothing processed.", file=sys.stderr)
        return 1

    summary = print_report(results, settings)
    if settings.output.get("write_events", True):
        export_events(results, out_dir, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
