"""Per-camera run loop: read -> detect/track -> zones -> analytics -> overlay."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from . import overlay
from .analytics import CameraAnalytics
from .config import CameraConfig, Settings
from .detector import PersonTracker

BUSY_ORDER = ["EMPTY", "QUIET", "STEADY", "BUSY", "PACKED"]


def busy_status(occupancy: float, bands: dict) -> str:
    if occupancy < bands.get("empty", 1):
        return "EMPTY"
    if occupancy < bands.get("quiet", 8):
        return "QUIET"
    if occupancy < bands.get("steady", 18):
        return "STEADY"
    if occupancy < bands.get("busy", 28):
        return "BUSY"
    return "PACKED"


def run_camera(cam: CameraConfig, settings: Settings, tracker: PersonTracker,
               out_dir: Path) -> tuple:
    proc = settings.processing
    an_cfg = settings.analytics

    cap = cv2.VideoCapture(str(cam.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cam.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_duration = total_frames / src_fps if total_frames else 0.0

    max_minutes = proc.get("max_minutes")
    duration = min(src_duration, max_minutes * 60) if max_minutes else src_duration

    sample_fps = float(proc.get("sample_fps", 5))
    step = max(1, int(round(src_fps / sample_fps)))

    wall_start = getattr(cam, "start_time", None)
    t0 = None
    if wall_start:
        try:
            t0 = datetime.strptime(wall_start, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            t0 = None

    tracker.reset()
    analytics = CameraAnalytics(cam, an_cfg, wall_start=wall_start)

    trails: dict[int, deque] = {}
    trail_len = int(proc.get("trail_length", 40))
    display = bool(proc.get("display", True))
    scale = float(proc.get("display_scale", 0.6))
    win = f"MOD CCTV - {cam.label}"
    writer = None

    if display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    entries_in = 0
    frame_idx = 0
    started = time.time()
    paused = False
    quit_all = False

    print(f"\n[{cam.label}] {cam.video.name} | {total_frames} frames @ {src_fps:.0f} fps "
          f"| sampling every {step} frames (~{sample_fps:.0f} fps) "
          f"| {duration/60:.1f} min of footage")

    while True:
        ok = cap.grab()
        if not ok:
            break
        t = frame_idx / src_fps
        if duration and t > duration:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break

        tracks = tracker.update(frame)
        masked_ids = {tr.track_id for tr in tracks if analytics.in_mask(tr.foot)}
        customer_tracks = [tr for tr in tracks if tr.track_id not in masked_ids]

        snap = analytics.update(t, customer_tracks)

        entries_in = sum(1 for e in analytics.line_events if e.direction == "in")
        queue_len = sum(
            snap["counts"].get(z.id, 0) for z in cam.zones if z.type == "queue"
        )

        if display or writer is not None:
            for tr in customer_tracks:
                dq = trails.setdefault(tr.track_id, deque(maxlen=trail_len))
                dq.append((int(tr.foot[0]), int(tr.foot[1])))
            for tid in list(trails):
                if tid not in {tr.track_id for tr in customer_tracks}:
                    trails.pop(tid, None)

            vis = overlay.draw_static(frame, cam, snap["counts"])
            overlay.draw_tracks(vis, tracks, {k: list(v) for k, v in trails.items()},
                                masked_ids)
            clock = (t0 + timedelta(seconds=t)).strftime("%H:%M:%S") if t0 else "--:--:--"
            overlay.draw_hud(
                vis,
                cam_label=cam.label,
                clock=clock,
                t_s=t,
                duration_s=duration,
                in_view=snap["in_view"],
                occupancy=snap["occupancy"],
                entries=entries_in,
                queue_len=queue_len,
                status=busy_status(snap["occupancy"], an_cfg.get("busy_thresholds", {})),
                progress=(t / duration) if duration else 0.0,
            )

            if proc.get("save_annotated_video") and writer is None:
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"annotated_{cam.name}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         sample_fps, (vis.shape[1], vis.shape[0]))
            if writer is not None:
                writer.write(vis)

            if display:
                small = cv2.resize(vis, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
                cv2.imshow(win, small)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 27:  # Esc -> stop everything
                    quit_all = True
                    break
                if key == ord("s"):
                    out_dir.mkdir(parents=True, exist_ok=True)
                    snap_path = out_dir / f"snapshot_{cam.name}_{int(t)}s.jpg"
                    cv2.imwrite(str(snap_path), vis)
                    print(f"  saved {snap_path}")
                if key == ord(" "):
                    paused = True
                while paused:
                    k2 = cv2.waitKey(50) & 0xFF
                    if k2 in (ord(" "), ord("q"), 27):
                        paused = False

        if analytics.frames % 250 == 0:
            el = time.time() - started
            pct = (t / duration * 100) if duration else 0
            print(f"  [{cam.label}] {pct:5.1f}%  t={t/60:5.1f}min  "
                  f"in view={snap['in_view']:2d}  zone-occ={snap['occupancy']:2d}  "
                  f"entries={entries_in:3d}  ({analytics.frames/max(el,1e-6):.1f} fps)",
                  flush=True)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyWindow(win)

    result = analytics.finalize()
    el = time.time() - started
    print(f"[{cam.label}] done: {result.frames_processed} frames analysed in "
          f"{el/60:.1f} min ({result.frames_processed/max(el,1e-6):.1f} fps), "
          f"{result.tracks_total} anonymous tracks")
    return result, quit_all
