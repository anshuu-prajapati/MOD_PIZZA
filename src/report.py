"""Terminal analytics summary + event export.

Everything printed here is derived from anonymous, temporary track IDs.
"""

from __future__ import annotations

import json
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .analytics import CameraResult
from .pipeline import busy_status

W = 84
BAR = "#"


# --------------------------------------------------------------------- helpers
def _h(title: str) -> str:
    return f"\n{'-' * W}\n {title}\n{'-' * W}"


def _bar(value: float, vmax: float, width: int = 28) -> str:
    if vmax <= 0:
        return ""
    return BAR * max(0, int(round(value / vmax * width)))


def _fmt_dur(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(round(seconds)), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _clock(t0: datetime | None, secs: float) -> str:
    if t0 is None:
        m, s = divmod(int(secs), 60)
        return f"t+{m:02d}:{s:02d}"
    return (t0 + timedelta(seconds=secs)).strftime("%H:%M")


def _zone_label(results: list[CameraResult], zone_id: str) -> str:
    for r in results:
        for z in r.zones:
            if z.id == zone_id:
                return z.label
    return zone_id.replace("_", " ")


def _zone_type(results: list[CameraResult], zone_id: str) -> str:
    for r in results:
        for z in r.zones:
            if z.id == zone_id:
                return z.type
    return "other"


def _combined_occupancy(results: list[CameraResult]) -> list[tuple[float, float]]:
    """Store occupancy = sum of each camera's own (non-overlapping) zone counts."""
    merged: dict[float, float] = defaultdict(float)
    for r in results:
        for t, v in r.occupancy_series:
            merged[t] += v
    return sorted(merged.items())


def _window_means(series: list[tuple[float, float]], window_s: int):
    buckets: dict[int, list[float]] = defaultdict(list)
    for t, v in series:
        buckets[int(t // window_s)].append(v)
    return sorted((b * window_s, float(np.mean(v))) for b, v in buckets.items())


# ---------------------------------------------------------------------- report
def print_report(results: list[CameraResult], settings) -> dict:
    an = settings.analytics
    bands = an.get("busy_thresholds", {})
    t0 = None
    for r in results:
        if r.wall_start:
            try:
                t0 = datetime.strptime(r.wall_start, "%Y-%m-%d %H:%M:%S")
                break
            except ValueError:
                pass

    duration = max((r.duration_s for r in results), default=0.0)
    frames = sum(r.frames_processed for r in results)
    tracks = sum(r.tracks_total for r in results)
    occ = _combined_occupancy(results)
    summary: dict = {}

    print("\n" + "=" * W)
    print(" MOD PIZZA - CCTV CUSTOMER ANALYTICS SUMMARY")
    print(" Anonymous only: no faces, no names, no purchases, no customer profiles.")
    print("=" * W)
    span = (f"{_clock(t0, 0)} -> {_clock(t0, duration)}" if t0 else
            f"0 -> {_fmt_dur(duration)}")
    if t0:
        span = f"{t0.strftime('%Y-%m-%d')}  {span}"
    print(f" Footage window   : {span}   ({duration/60:.1f} min per camera)")
    print(f" Cameras          : {', '.join(r.label for r in results)}")
    print(f" Frames analysed  : {frames:,}  |  anonymous tracks: {tracks}")
    print(" Track IDs are temporary, camera-local, and discarded at the end of the run.")
    print(" A track is a continuous sighting, NOT a person: one person who is occluded")
    print(" and re-detected becomes a second track, so track counts exceed head count.")

    # ---------------------------------------------------------- 1. entries
    print(_h("1. HOW MANY PEOPLE ARE ENTERING"))
    ins = [e for r in results for e in r.line_events if e.direction == "in"]
    outs = [e for r in results for e in r.line_events if e.direction == "out"]
    entries = len(ins)
    print(f"  Front-door crossings IN     : {entries}")
    print(f"  Front-door crossings OUT    : {len(outs)}")
    print(f"  Net flow (in - out)         : {entries - len(outs):+d}")
    if duration:
        print(f"  Entry rate                  : {entries / (duration/3600):.0f} people/hour")

    if ins:
        block = 600
        per_block: Counter = Counter(int(e.t // block) for e in ins)
        vmax = max(per_block.values())
        print("\n  Entries per 10-minute block:")
        n_blocks = max(1, int(round(duration / block)), max(per_block) + 1)
        for b in range(n_blocks):
            n = per_block.get(b, 0)
            print(f"    {_clock(t0, b*block)}-{_clock(t0, (b+1)*block)}  "
                  f"{_bar(n, vmax, 24):<24} {n}")
    else:
        print("  (no door-line crossings recorded - check the front_door line in "
              "config/zones/front_entrance.json)")
    print("  NOTE: front door only. The side door has no camera, so this UNDERCOUNTS.")
    summary["entries_in"] = entries
    summary["entries_out"] = len(outs)

    # ------------------------------------------------------- 2. busy / empty
    print(_h("2. WHEN THE STORE IS BUSY OR EMPTY"))
    if occ:
        shown, step_s = occ, int(an.get("occupancy_bin_seconds", 60))
        if len(occ) > 75:                       # a full day would scroll forever
            step_s = int(np.ceil(len(occ) / 75)) * step_s
            shown = _window_means(occ, step_s)
        vmax = max(v for _, v in shown)
        print(f"  Store occupancy every {max(1, step_s // 60)} min "
              f"(sum of non-overlapping camera zones):")
        for t, v in shown:
            print(f"    {_clock(t0, t)}  {_bar(v, vmax, 32):<32} {v:5.1f}  "
                  f"{busy_status(v, bands)}")

        tenmin = _window_means(occ, 600)
        busiest = max(tenmin, key=lambda kv: kv[1])
        quietest = min(tenmin, key=lambda kv: kv[1])
        print(f"\n  Busiest 10 min  : {_clock(t0, busiest[0])}-{_clock(t0, busiest[0]+600)}"
              f"   avg {busiest[1]:.1f} people  ({busy_status(busiest[1], bands)})")
        print(f"  Quietest 10 min : {_clock(t0, quietest[0])}-{_clock(t0, quietest[0]+600)}"
              f"   avg {quietest[1]:.1f} people  ({busy_status(quietest[1], bands)})")
        print(f"  Peak minute     : {_clock(t0, max(occ, key=lambda kv: kv[1])[0])}"
              f"   {vmax:.1f} people")
        print(f"  Average         : {np.mean([v for _, v in occ]):.1f} people")

        share: Counter = Counter(busy_status(v, bands) for _, v in occ)
        tot = sum(share.values())
        print("  Time spent      : " + " | ".join(
            f"{k} {share.get(k,0)/tot*100:.0f}%"
            for k in ["EMPTY", "QUIET", "STEADY", "BUSY", "PACKED"]))
        summary["peak_occupancy"] = round(vmax, 1)
        summary["avg_occupancy"] = round(float(np.mean([v for _, v in occ])), 1)
        summary["busiest_window"] = _clock(t0, busiest[0])
    else:
        print("  no occupancy samples")

    # ------------------------------------------------------ 3. crowded now
    print(_h("3. HOW CROWDED THE STORE IS RIGHT NOW (last processed frame)"))
    now = 0
    for r in results:
        now += r.live_occupancy
        print(f"  {r.label:<18} {r.live_occupancy:>3} people in its own zones")
    print(f"  {'STORE TOTAL':<18} {now:>3} people   ->  STATUS: {busy_status(now, bands)}")
    print(f"  Bands: EMPTY <{bands.get('empty',1)} | QUIET <{bands.get('quiet',8)} | "
          f"STEADY <{bands.get('steady',18)} | BUSY <{bands.get('busy',28)} | "
          f"PACKED >={bands.get('busy',28)}   (tune in config/settings.yaml)")
    summary["occupancy_now"] = now
    summary["status_now"] = busy_status(now, bands)

    # ------------------------------------------------- 4. tables + dwell
    print(_h("4. WHICH TABLES ARE BEING USED AND FOR HOW LONG"))
    seat_s = float(an.get("table_seat_seconds", 60.0))
    print(f"  A 'sitting' = one anonymous track present in a table zone for >= "
          f"{seat_s:.0f}s.")
    header = (f"  {'TABLE ZONE':<26}{'CAMERA':<18}{'SITTINGS':>9}{'AVG DWELL':>11}"
              f"{'LONGEST':>10}{'PEAK':>6}{'% OF TIME IN USE':>18}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    table_rows = []
    for r in results:
        table_ids = [z.id for z in r.zones if z.type == "table"]
        for zid in table_ids:
            v = [x for x in r.visits if x.zone_id == zid and x.dwell >= seat_s]
            if not v:
                continue
            occupied = r.zone_occupied_samples.get(zid, 0)
            dwells = [x.dwell for x in v]
            row = (_zone_label(results, zid), r.label, len(v),
                   float(np.mean(dwells)), max(dwells),
                   occupied / max(1, r.frames_processed) * 100,
                   r.zone_peaks.get(zid, 0))
            table_rows.append(row)
    table_rows.sort(key=lambda x: -x[2])
    for label, cam, n, avg, mx, pct, peak in table_rows:
        print(f"  {label:<26}{cam:<18}{n:>9}{_fmt_dur(avg):>11}{_fmt_dur(mx):>10}"
              f"{peak:>6}{pct:>17.0f}%")
    if table_rows:
        all_d = [d for r in results for x in r.visits
                 if _zone_type(results, x.zone_id) == "table" and x.dwell >= seat_s
                 for d in [x.dwell]]
        print(f"\n  Total sittings across all tables : {sum(r[2] for r in table_rows)}")
        print(f"  Average dwell at a table         : {_fmt_dur(float(np.mean(all_d)))}")
        print(f"  Longest single sitting           : {_fmt_dur(max(all_d))}")
        print(f"  Most-used table                  : {table_rows[0][0]} "
              f"({table_rows[0][2]} sittings)")
        idle = [z.label for r in results for z in r.zones if z.type == "table"
                and z.label not in {t[0] for t in table_rows}]
        print(f"  Tables never used in this window : "
              f"{', '.join(idle) if idle else 'none - every table was used'}")
        summary["total_sittings"] = sum(r[2] for r in table_rows)
        summary["avg_table_dwell_s"] = round(float(np.mean(all_d)), 1)
    else:
        print("  (no table sittings recorded)")

    # ------------------------------------------------------- 5. movement
    print(_h("5. HOW PEOPLE MOVE THROUGH THE STORE"))
    for r in results:
        if not r.transitions:
            continue
        print(f"\n  {r.label} - most common zone-to-zone moves:")
        for (a, b), n in r.transitions.most_common(8):
            print(f"    {_zone_label(results,a):<24} -> {_zone_label(results,b):<24} {n:>4}")
    flows = Counter()
    for r in results:
        flows.update(r.transitions)
    print("\n  Dominant flows store-wide (per-camera paths; not stitched across cameras):")
    for (a, b), n in flows.most_common(5):
        print(f"    {_zone_label(results,a)} -> {_zone_label(results,b)}  ({n} moves)")
    dur_all = [d for r in results for d in r.track_durations.values()]
    if dur_all:
        print(f"\n  Average time a person stays visible to one camera : "
              f"{_fmt_dur(float(np.mean(dur_all)))}")
        print(f"  Longest continuous track                          : "
              f"{_fmt_dur(max(dur_all))}")

    # --------------------------------------------------------- 6. returns
    print(_h("6. WHETHER PEOPLE APPEAR TO RETURN"))
    reent = sum(r.reentries for r in results)
    print(f"  Short-term door re-entries (same live track crossed IN again "
          f"after >= {an.get('reentry_gap_seconds',60):.0f}s): {reent}")
    revisits = Counter()
    for r in results:
        revisits.update(r.zone_revisits)
    if revisits:
        print(f"  In-store return trips (left a zone and came back after >= "
              f"{an.get('revisit_gap_seconds',45):.0f}s):")
        for zid, n in revisits.most_common(8):
            print(f"    back to {_zone_label(results, zid):<26} {n:>4} times")
        print(f"  Total return trips: {sum(revisits.values())}")
    else:
        print("  No in-store return trips detected.")
    print("  LIMIT: this is within-session only. Recognising a returning customer on")
    print("         another day would need face/appearance ID - deliberately NOT built.")
    summary["reentries"] = reent
    summary["zone_return_trips"] = sum(revisits.values())

    # ---------------------------------------------------- 7. where they enter
    print(_h("7. WHERE PEOPLE ENTER"))
    by_line: Counter = Counter()
    for r in results:
        for e in r.line_events:
            if e.direction == "in":
                by_line[f"{r.label} / {e.line_id}"] += 1
    if by_line:
        for k, n in by_line.most_common():
            print(f"  {k:<44} {n:>4} entries")
    else:
        print("  no door-line crossings recorded")
    print("\n  Where a track is first seen (entry point into each camera's view).")
    print("  This includes tracks re-acquired after an occlusion, so mid-room zones")
    print("  show up too. The door line above is the entry number that matters.")
    for r in results:
        top = r.origin_zones.most_common(5)
        if not top:
            continue
        print(f"    {r.label}:")
        for zid, n in top:
            name = "outside any zone (frame edge)" if zid == "off_zone" else _zone_label(results, zid)
            print(f"      {name:<34} {n:>4} tracks")
    print("  NOTE: only the front door is covered. The side door is invisible to the system.")

    # -------------------------------------------------- 8. stop / wait spots
    print(_h("8. WHERE THEY STOP OR WAIT"))
    print(f"  A 'stop' = moving slower than {an.get('stop_speed_px_per_s',22):.0f} px/s "
          f"for >= {an.get('stop_min_seconds',5):.0f}s.")
    print("  Table zones are excluded here - people sitting down are section 4, not a")
    print("  queue. This list is standing-and-waiting on the open floor only.")
    stop_count: Counter = Counter()
    stop_time: Counter = Counter()
    seated_time: Counter = Counter()
    for r in results:
        for s in r.stops:
            if _zone_type(results, s.zone_id) == "table":
                seated_time[s.zone_id] += s.duration   # that is sitting, not waiting
                continue
            stop_count[s.zone_id] += 1
            stop_time[s.zone_id] += s.duration
    if stop_count:
        vmax = max(stop_time.values())
        print(f"  {'ZONE':<30}{'STOPS':>8}{'TOTAL WAITING':>16}{'AVG':>10}")
        print("  " + "-" * 62)
        for zid, n in stop_time.most_common(12):
            name = "unzoned floor" if zid == "off_zone" else _zone_label(results, zid)
            print(f"  {name:<30}{stop_count[zid]:>8}{_fmt_dur(n):>16}"
                  f"{_fmt_dur(n/max(1,stop_count[zid])):>10}")
        hot = stop_time.most_common(1)[0]
        if hot[0] == "off_zone":
            print("\n  Most waiting happens on floor no polygon covers. Add zones there")
            print("  (--check-zones shows the gaps) to learn what people are waiting for.")
            hot = next(((z, n) for z, n in stop_time.most_common() if z != "off_zone"),
                       hot)
        print(f"\n  Biggest waiting hotspot inside a zone: "
              f"{_zone_label(results, hot[0])} ({_fmt_dur(hot[1])} of standing still)")
    else:
        print("  no stop events recorded")

    # --------------------------------------------------------- 9. queues
    print(_h("9. WHERE QUEUES FORM"))
    any_q = False
    for r in results:
        for z in r.zones:
            if z.type != "queue":
                continue
            any_q = True
            series = r.zone_occupancy_series.get(z.id, [])
            visits = [v for v in r.visits if v.zone_id == z.id]
            counts = [c for _, c in series]
            peak = r.zone_peaks.get(z.id, 0)
            avg = float(np.mean(counts)) if counts else 0.0
            busy_min = sum(1 for c in counts if c >= 2)
            waits = [v.dwell for v in visits]
            print(f"\n  {z.label}  ({r.label})")
            print(f"    People who passed through : {len(visits)}")
            print(f"    Peak queue length         : {peak} people (instantaneous)")
            print(f"    Average queue length      : {avg:.1f} people")
            print(f"    Minutes with 2+ waiting   : {busy_min}")
            if waits:
                print(f"    Average time in the queue : {_fmt_dur(float(np.mean(waits)))}")
                print(f"    Longest wait              : {_fmt_dur(max(waits))}")
            if counts:
                pk = max(series, key=lambda kv: kv[1])
                print(f"    Queue peaked at           : {_clock(t0, pk[0])}")
    if not any_q:
        print("  no queue zones configured")
    print("\n  NOTE: staff-side floor is masked out of the POS camera so employees are")
    print("        not counted as customers.")

    # ------------------------------------------------- 10. areas visited
    print(_h("10. WHICH AREAS THEY VISIT"))
    visits_by_zone: Counter = Counter()
    people_by_zone: dict[str, set] = defaultdict(set)
    time_by_zone: Counter = Counter()
    for r in results:
        for v in r.visits:
            visits_by_zone[v.zone_id] += 1
            people_by_zone[v.zone_id].add((r.name, v.track_id))
            time_by_zone[v.zone_id] += v.dwell
    if visits_by_zone:
        vmax = max(visits_by_zone.values())
        print(f"  {'ZONE':<28}{'TYPE':<11}{'VISITS':>8}{'TRACKS':>8}{'TOTAL TIME':>13}  TRAFFIC")
        print("  " + "-" * 80)
        for zid, n in visits_by_zone.most_common():
            print(f"  {_zone_label(results,zid):<28}{_zone_type(results,zid):<11}{n:>8}"
                  f"{len(people_by_zone[zid]):>8}{_fmt_dur(time_by_zone[zid]):>13}  "
                  f"{_bar(n, vmax, 18)}")
        cold = [z.label for r in results for z in r.zones if z.id not in visits_by_zone]
        print(f"\n  Zones with no recorded visits: {', '.join(cold) if cold else 'none'}")
    else:
        print("  no zone visits recorded")

    # -------------------------------------- 11. toward counter vs seating
    print(_h("11. WHERE THEY MOVE TOWARD THE COUNTER / SEATING"))

    # (a) The robust answer: how the whole room's traffic splits between the
    #     ordering side and the seating side. Every confirmed visit counts, so
    #     a broken track still contributes.
    counter_visits = sum(n for z, n in visits_by_zone.items()
                         if _zone_type(results, z) in ("counter", "queue"))
    seat_visits = sum(n for z, n in visits_by_zone.items()
                      if _zone_type(results, z) in ("table", "beverage"))
    split_tot = max(1, counter_visits + seat_visits)
    print("  Traffic split across the whole hour (every zone visit counts):")
    print(f"    counter / order queue   {counter_visits:>5} visits  "
          f"({counter_visits/split_tot*100:.0f}%)")
    print(f"    seating / beverage      {seat_visits:>5} visits  "
          f"({seat_visits/split_tot*100:.0f}%)")
    summary["visits_counter_side"] = counter_visits
    summary["visits_seating_side"] = seat_visits

    # (b) The direct answer: what a person did straight after walking in. Only
    #     tracks that survive from the door to their destination qualify, and
    #     ByteTrack drops a lot of them behind the booth, so the sample is small
    #     and is reported as a count, not sold as a percentage of all customers.
    dest: Counter = Counter()
    for r in results:
        dest.update(r.first_destination)
    if dest:
        to_order = sum(n for z, n in dest.items()
                       if _zone_type(results, z) in ("counter", "queue"))
        to_seat = sum(n for z, n in dest.items()
                      if _zone_type(results, z) in ("table", "beverage"))
        tot = sum(dest.values())
        print(f"\n  Of the {tot} arrivals whose track survived from the door to a")
        print("  destination (walkways skipped), the first place they went was:")
        print(f"    -> counter / order queue   {to_order:>4}")
        print(f"    -> seating / beverage      {to_seat:>4}")
        print("  Broken down by destination zone:")
        for zid, n in dest.most_common(8):
            print(f"    {_zone_label(results,zid):<28} {n:>4}")
        print(f"  CAUTION: {tot} arrivals is a small sample next to the door count")
        print("  above. Treat the traffic split as the reliable figure of the two.")
        summary["toward_counter"] = to_order
        summary["toward_seating"] = to_seat
    else:
        print("\n  No arrival kept a single track from the door to a destination in")
        print("  this window, so the per-arrival breakdown is unavailable.")

    # ----------------------------------------------------------- limitations
    print(_h("KNOWN LIMITATIONS (do not read the numbers without these)"))
    limitations = [
        "Front door only - the side door has no camera, so entries undercount.",
        "The front door is distant and oblique; groups walking in together merge.",
        "Track IDs are per-camera. Someone walking between cameras becomes a new "
        "anonymous track in each view, so store-wide paths are coarse.",
        "Returning customers across visits or days are NOT computed. Only "
        "within-session re-entry and in-store return trips are reported.",
        "Staff are excluded by masking fixed floor areas. A staff member stepping "
        "onto the customer floor can still be counted.",
        "Seated people are detected from the torso, so their floor point is "
        "approximate; table polygons are drawn generously to compensate.",
        "Occupancy sums each camera's own zones. Overlapping floor is owned by "
        "exactly one camera, so editing the zone files changes the totals.",
        "Tracking is appearance-free, so a person hidden behind another person or a "
        "booth comes back as a NEW track. Track totals therefore overstate head "
        "count; occupancy and dwell are far more reliable than track counts.",
        "Entry counts have not been ground-truthed on this footage yet. Hand-count "
        "two or three 10-minute windows before quoting the number to anyone.",
    ]
    for i, txt in enumerate(limitations, 1):
        for j, ln in enumerate(textwrap.wrap(txt, W - 8)):
            print(f"  {i}. {ln}" if j == 0 else f"     {ln}")
    print("\n" + "=" * W)
    print(" End of report. No personal data was produced, stored, or inferred.")
    print("=" * W + "\n")

    return summary


# ---------------------------------------------------------------------- export
def export_events(results: list[CameraResult], out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "zone_visits.csv").open("w", encoding="utf-8") as f:
        f.write("camera,track_id,zone_id,enter_s,exit_s,dwell_s\n")
        for r in results:
            for v in r.visits:
                f.write(f"{r.name},{v.track_id},{v.zone_id},{v.enter_t:.2f},"
                        f"{v.exit_t:.2f},{v.dwell:.2f}\n")

    with (out_dir / "line_events.csv").open("w", encoding="utf-8") as f:
        f.write("camera,track_id,line_id,direction,t_s\n")
        for r in results:
            for e in r.line_events:
                f.write(f"{r.name},{e.track_id},{e.line_id},{e.direction},{e.t:.2f}\n")

    with (out_dir / "occupancy_timeline.csv").open("w", encoding="utf-8") as f:
        f.write("camera,bin_start_s,mean_people\n")
        for r in results:
            for t, v in r.occupancy_series:
                f.write(f"{r.name},{t:.0f},{v:.2f}\n")

    with (out_dir / "stop_events.csv").open("w", encoding="utf-8") as f:
        f.write("camera,track_id,zone_id,start_s,duration_s\n")
        for r in results:
            for s in r.stops:
                f.write(f"{r.name},{s.track_id},{s.zone_id},{s.start_t:.2f},"
                        f"{s.duration:.2f}\n")

    (out_dir / "summary.json").write_text(
        json.dumps({
            "generated": datetime.now().isoformat(timespec="seconds"),
            "cameras": [r.name for r in results],
            **summary,
        }, indent=2),
        encoding="utf-8",
    )
    print(f" Events + summary written to {out_dir}/")
