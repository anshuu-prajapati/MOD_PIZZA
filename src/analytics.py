"""Metric engine.

Consumes (timestamp, tracks) and produces the store-intelligence metrics.
It never touches pixels and never learns anything about who a person is --
only where an anonymous, temporary track ID was, and for how long.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .config import CameraConfig, Zone
from .geometry import crossing_direction, line_side, point_in_polygon, speed_px_per_s

# When a foot point falls in several zones, the more "meaningful" type wins.
ZONE_PRIORITY = {
    "queue": 0,
    "counter": 1,
    "entrance": 2,
    "beverage": 3,
    "table": 4,
    "exit": 5,
    "other": 6,
}


@dataclass
class ZoneVisit:
    track_id: int
    zone_id: str
    enter_t: float
    exit_t: float

    @property
    def dwell(self) -> float:
        return self.exit_t - self.enter_t


@dataclass
class StopEvent:
    track_id: int
    zone_id: str
    start_t: float
    duration: float
    x: float = 0.0          # where the person was standing, for the waiting heatmap
    y: float = 0.0


@dataclass
class OccupancyEvent:
    """One continuous spell of a zone being occupied.

    Counted from the per-frame headcount inside the polygon, never from track
    identity. That is the whole point: a person who is occluded and re-detected
    becomes a new track ID, which wrecks any metric built on tracks, but the
    headcount in the polygon does not change. So a party of four seated for ten
    minutes is one event of size four, however many track IDs it churned through.
    """
    zone_id: str
    start_t: float
    end_t: float
    peak: int               # most people present at once
    size: int               # the plateau -- the headcount held for longest
    samples: int

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t


@dataclass
class LineEvent:
    track_id: int
    line_id: str
    direction: str
    t: float


@dataclass
class TrackState:
    track_id: int
    first_t: float
    last_t: float
    points: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)
    open_zones: dict[str, float] = field(default_factory=dict)
    pending_exit: dict[str, float] = field(default_factory=dict)
    last_exit_t: dict[str, float] = field(default_factory=dict)
    revisits: Counter = field(default_factory=Counter)
    sequence: list[str] = field(default_factory=list)
    line_state: dict[str, int] = field(default_factory=dict)
    door_in_times: list[float] = field(default_factory=list)
    stop_start: float | None = None
    stop_zone: str = "other"
    stop_xy: tuple[float, float] = (0.0, 0.0)
    origin_zone: str | None = None
    distance: float = 0.0


@dataclass
class CameraResult:
    name: str
    label: str
    duration_s: float
    frames_processed: int
    wall_start: str | None
    zones: list[Zone]
    tracks_total: int
    visits: list[ZoneVisit]
    stops: list[StopEvent]
    line_events: list[LineEvent]
    occupancy_series: list[tuple[float, float]]       # (bin_start_s, mean people)
    zone_occupancy_series: dict[str, list[tuple[float, float]]]
    zone_raw_series: dict[str, list[tuple[float, int]]]   # per-frame headcount
    occupancy_events: list[OccupancyEvent]
    live_counts: dict[str, int]
    live_occupancy: int
    zone_peaks: dict[str, int]
    zone_occupied_samples: dict[str, int]
    transitions: Counter
    origin_zones: Counter
    first_destination: Counter                        # after the entrance, where next
    reentries: int
    reentry_tracks: list[int]
    zone_revisits: Counter
    track_durations: dict[int, float]
    peak_in_view: int


class CameraAnalytics:
    def __init__(self, camera: CameraConfig, settings: dict, wall_start: str | None = None):
        self.cam = camera
        self.s = settings
        self.wall_start = wall_start

        self.tracks: dict[int, TrackState] = {}
        self.visits: list[ZoneVisit] = []
        self.stops: list[StopEvent] = []
        self.line_events: list[LineEvent] = []
        self.transitions: Counter = Counter()
        self.origin_zones: Counter = Counter()
        self.first_destination: Counter = Counter()
        self.zone_revisits: Counter = Counter()

        self._occ_bins: dict[int, list[int]] = defaultdict(list)
        self._zone_bins: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        # the raw per-frame headcount per zone -- the source of truth for every
        # metric that must survive track fragmentation
        self._zone_series: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self.live_counts: dict[str, int] = {z.id: 0 for z in camera.zones}
        self.live_occupancy = 0
        self.zone_peaks: Counter = Counter()
        self.zone_occupied_samples: Counter = Counter()
        self.peak_in_view = 0
        self.frames = 0
        self.last_t = 0.0

        self._sorted_zones = sorted(
            camera.zones, key=lambda z: ZONE_PRIORITY.get(z.type, 9)
        )
        self._bin = int(settings.get("occupancy_bin_seconds", 60))

    # ------------------------------------------------------------------ utils
    def _zones_at(self, pt: tuple[float, float]) -> list[str]:
        return [z.id for z in self._sorted_zones if point_in_polygon(pt, z.points)]

    def in_mask(self, pt: tuple[float, float]) -> bool:
        return any(point_in_polygon(pt, m.points) for m in self.cam.masks)

    def _zone_type(self, zone_id: str) -> str:
        z = self.cam.zone(zone_id)
        return z.type if z else "other"

    # ----------------------------------------------------------------- update
    def update(self, t: float, tracks) -> dict:
        """Feed one sampled frame. Returns a live snapshot for the overlay."""
        self.frames += 1
        self.last_t = t
        counts = {z.id: 0 for z in self.cam.zones}
        seen_ids = set()
        occupancy_ids = set()   # distinct people, so overlapping zones never double-count

        for tr in tracks:
            tid = tr.track_id
            seen_ids.add(tid)
            pt = tr.foot

            st = self.tracks.get(tid)
            if st is None:
                st = TrackState(track_id=tid, first_t=t, last_t=t)
                self.tracks[tid] = st
            else:
                prev = st.points[-1]
                st.distance += float(np.hypot(pt[0] - prev[1], pt[1] - prev[2]))
            st.last_t = t
            st.points.append((t, pt[0], pt[1]))

            here = self._zones_at(pt)
            for zid in here:
                counts[zid] += 1
                z = self.cam.zone(zid)
                if z is not None and z.occupancy:
                    occupancy_ids.add(tid)
            if st.origin_zone is None:
                st.origin_zone = here[0] if here else "off_zone"

            self._update_zone_membership(st, t, here)
            self._update_lines(st, t, pt)
            self._update_stops(st, t, here)

        # close out tracks that disappeared this frame
        for tid, st in list(self.tracks.items()):
            if tid in seen_ids or st.last_t == t:
                continue
            if st.open_zones or st.stop_start is not None:
                self._close_track(st, st.last_t)

        for zid, c in counts.items():
            self._zone_bins[zid][int(t // self._bin)].append(c)
            self._zone_series[zid].append((t, c))
            if c > self.zone_peaks[zid]:
                self.zone_peaks[zid] = c
            if c > 0:
                self.zone_occupied_samples[zid] += 1
        total = len(occupancy_ids)
        self._occ_bins[int(t // self._bin)].append(total)
        self.live_counts = counts
        self.live_occupancy = total
        self.peak_in_view = max(self.peak_in_view, len(seen_ids))

        return {"counts": counts, "occupancy": total, "in_view": len(seen_ids)}

    def _update_zone_membership(self, st: TrackState, t: float, here: list[str]) -> None:
        """Open/close zone visits with hysteresis.

        A foot point sitting on a zone boundary jitters in and out many times a
        minute. Without a grace period that single person produces dozens of
        fake visits and hundreds of fake A->B->A transitions, which swamps the
        movement metrics. So a zone is only really left once the track has been
        outside it continuously for `zone_exit_grace_seconds`.
        """
        gap = float(self.s.get("revisit_gap_seconds", 45.0))
        min_dwell = float(self.s.get("zone_min_dwell_seconds", 3.0))
        grace = float(self.s.get("zone_exit_grace_seconds", 2.5))

        for zid in here:
            if zid in st.open_zones:
                st.pending_exit.pop(zid, None)   # came back before the grace ran out
                continue
            st.open_zones[zid] = t
            st.pending_exit.pop(zid, None)
            if zid in st.last_exit_t and t - st.last_exit_t[zid] >= gap:
                st.revisits[zid] += 1
                self.zone_revisits[zid] += 1
            if not st.sequence or st.sequence[-1] != zid:
                st.sequence.append(zid)

        for zid, enter_t in list(st.open_zones.items()):
            if zid in here:
                continue
            first_absent = st.pending_exit.setdefault(zid, t)
            if t - first_absent < grace:
                continue
            del st.open_zones[zid]
            del st.pending_exit[zid]
            st.last_exit_t[zid] = first_absent
            if first_absent - enter_t >= min_dwell:
                self.visits.append(ZoneVisit(st.track_id, zid, enter_t, first_absent))

    def _update_lines(self, st: TrackState, t: float, pt: tuple[float, float]) -> None:
        for ln in self.cam.lines:
            side = line_side(pt, ln.points)
            prev = st.points[-2][1:] if len(st.points) >= 2 else None
            if prev is not None:
                direction = crossing_direction(prev, pt, ln.points, ln.in_side)
                if direction:
                    self.line_events.append(LineEvent(st.track_id, ln.id, direction, t))
                    if direction == "in":
                        st.door_in_times.append(t)
            st.line_state[ln.id] = side

    def _update_stops(self, st: TrackState, t: float, here: list[str]) -> None:
        thr = float(self.s.get("stop_speed_px_per_s", 22.0))
        min_s = float(self.s.get("stop_min_seconds", 5.0))
        spd = speed_px_per_s(st.points, window_s=1.0)

        if spd < thr:
            if st.stop_start is None:
                st.stop_start = t
                st.stop_zone = here[0] if here else "off_zone"
                st.stop_xy = (st.points[-1][1], st.points[-1][2])
        elif st.stop_start is not None:
            dur = t - st.stop_start
            if dur >= min_s:
                x, y = st.stop_xy
                self.stops.append(
                    StopEvent(st.track_id, st.stop_zone, st.stop_start, dur, x, y)
                )
            st.stop_start = None

    def _close_track(self, st: TrackState, t: float) -> None:
        min_dwell = float(self.s.get("zone_min_dwell_seconds", 3.0))
        for zid, enter_t in st.open_zones.items():
            st.last_exit_t[zid] = t
            if t - enter_t >= min_dwell:
                self.visits.append(ZoneVisit(st.track_id, zid, enter_t, t))
        st.open_zones.clear()
        st.pending_exit.clear()
        if st.stop_start is not None:
            dur = t - st.stop_start
            if dur >= float(self.s.get("stop_min_seconds", 5.0)):
                x, y = st.stop_xy
                self.stops.append(
                    StopEvent(st.track_id, st.stop_zone, st.stop_start, dur, x, y)
                )
            st.stop_start = None

    # ------------------------------------------------------- occupancy events
    def _event_gap(self, zone_type: str) -> float:
        """How long a zone must read empty before the spell is really over.

        Seated people get occluded for several seconds at a time, so a table
        needs a generous bridge. A register turns over in ~12s, so the same
        bridge there would merge two customers into one -- it needs a short one.
        """
        cfg = self.s.get("occupancy_event_gap_seconds", {})
        if isinstance(cfg, (int, float)):
            return float(cfg)
        return float(cfg.get(zone_type, cfg.get("default", 3.0)))

    def _split_on_clearing(self, spell: list[tuple[float, int]],
                           ) -> list[list[tuple[float, int]]]:
        """Cut one spell into separate parties wherever the table nearly clears.

        A busy table never reads zero -- the next party sits down before the
        polygon empties, and flicker fills the rest. Left unsplit, an hour of a
        popular table collapses into a single endless "sitting", which is worse
        than the per-track count it replaced. So a sustained dip to the clearing
        floor is treated as one party leaving and the next arriving.

        Only applied where it can mean something: a spell whose peak never
        exceeds the floor (a lone diner, or one customer at a register) is left
        whole, otherwise every solo visit would be split away to nothing.
        """
        cfg = self.s.get("party_split", {}) or {}
        floor = int(cfg.get("floor", 1))
        min_low = float(cfg.get("seconds", 15.0))

        peak = max(c for _, c in spell)
        if peak <= floor:
            return [spell]

        segments: list[list[tuple[float, int]]] = []
        cur: list[tuple[float, int]] = []
        low_since: float | None = None

        for t, c in spell:
            if c <= floor:
                if low_since is None:
                    low_since = t
                elif t - low_since >= min_low and cur:
                    segments.append(cur)      # that party has gone
                    cur = []
                    continue
            else:
                low_since = None
            cur.append((t, c))
        if cur:
            segments.append(cur)
        return [sg for sg in segments if sg and max(c for _, c in sg) > floor] or [spell]

    def _occupancy_events(self) -> list[OccupancyEvent]:
        events: list[OccupancyEvent] = []
        for zone in self.cam.zones:
            series = self._zone_series.get(zone.id, [])
            if not series:
                continue
            gap = self._event_gap(zone.type)
            split = zone.type == "table"

            spells: list[list[tuple[float, int]]] = []
            cur: list[tuple[float, int]] = []
            last_seen = None
            for t, c in series:
                if c > 0:
                    cur.append((t, c))
                    last_seen = t
                elif cur and t - last_seen >= gap:
                    spells.append(cur)
                    cur = []
            if cur:
                spells.append(cur)

            for spell in spells:
                parts = self._split_on_clearing(spell) if split else [spell]
                for part in parts:
                    counts = [c for _, c in part]
                    busy = [c for c in counts if c > 0]
                    events.append(OccupancyEvent(
                        zone_id=zone.id,
                        start_t=part[0][0],
                        end_t=part[-1][0],
                        peak=max(counts),
                        size=Counter(busy or counts).most_common(1)[0][0],
                        samples=len(part),
                    ))
        return events

    # --------------------------------------------------------------- finalize
    def finalize(self) -> CameraResult:
        for st in self.tracks.values():
            self._close_track(st, st.last_t)

        min_track = float(self.s.get("min_track_seconds", 1.5))
        valid = {
            tid: st for tid, st in self.tracks.items()
            if (st.last_t - st.first_t) >= min_track
        }
        valid_ids = set(valid)

        visits = [v for v in self.visits if v.track_id in valid_ids]
        stops = [s for s in self.stops if s.track_id in valid_ids]
        line_events = [e for e in self.line_events if e.track_id in valid_ids]

        # Movement path, built from confirmed visits rather than live membership.
        # Two zones that overlap on the floor are occupied at the same instant --
        # that is one person standing still, not a move -- so a transition is only
        # counted between visits that do NOT overlap in time.
        by_track: dict[int, list[ZoneVisit]] = defaultdict(list)
        for v in visits:
            by_track[v.track_id].append(v)
        self.transitions = Counter()
        ordered_path: dict[int, list[str]] = {}
        for tid, vs in by_track.items():
            vs.sort(key=lambda v: v.enter_t)
            path: list[ZoneVisit] = []
            for v in vs:
                if path and v.enter_t < path[-1].exit_t:
                    continue                      # overlapping -> same moment
                if path and path[-1].zone_id == v.zone_id:
                    continue
                if path:
                    self.transitions[(path[-1].zone_id, v.zone_id)] += 1
                path.append(v)
            ordered_path[tid] = [v.zone_id for v in path]

        # origin + first destination after the entrance
        origins: Counter = Counter()
        first_dest: Counter = Counter()
        for st in valid.values():
            origins[st.origin_zone or "off_zone"] += 1
            seq = ordered_path.get(st.track_id, st.sequence)
            start = None
            for i, zid in enumerate(seq):
                if self._zone_type(zid) == "entrance":
                    start = i
                    break
            if start is None and st.door_in_times:
                # crossed the door line without registering inside the entrance
                # polygon -- treat the door crossing as the start of the journey
                t_in = st.door_in_times[0]
                seen: list[str] = []
                for (tt, x, y) in st.points:
                    if tt < t_in:
                        continue
                    for z in self._zones_at((x, y)):
                        if z not in seen:
                            seen.append(z)
                seq, start = seen, -1
            if start is not None:
                # "Did they head for the counter or for a seat?" -- so skip the
                # walkways and the entrance itself and take the first zone that
                # actually answers the question.
                nxt = next(
                    (z for z in seq[start + 1:]
                     if self._zone_type(z) in ("counter", "queue", "table", "beverage")),
                    None,
                )
                if nxt:
                    first_dest[nxt] += 1

        # short-term re-entry: the SAME live track crosses the door inward twice
        gap = float(self.s.get("reentry_gap_seconds", 60.0))
        reentry_tracks = []
        for tid, st in valid.items():
            ins = st.door_in_times
            if any(b - a >= gap for a, b in zip(ins, ins[1:])):
                reentry_tracks.append(tid)

        occ_series = sorted(
            (b * self._bin, float(np.mean(v))) for b, v in self._occ_bins.items() if v
        )
        zone_series = {
            zid: sorted((b * self._bin, float(np.mean(v))) for b, v in bins.items() if v)
            for zid, bins in self._zone_bins.items()
        }

        return CameraResult(
            name=self.cam.name,
            label=self.cam.label,
            duration_s=self.last_t,
            frames_processed=self.frames,
            wall_start=self.wall_start,
            zones=self.cam.zones,
            tracks_total=len(valid),
            visits=visits,
            stops=stops,
            line_events=line_events,
            occupancy_series=occ_series,
            zone_occupancy_series=zone_series,
            zone_raw_series={z: list(v) for z, v in self._zone_series.items()},
            occupancy_events=self._occupancy_events(),
            live_counts=dict(self.live_counts),
            live_occupancy=self.live_occupancy,
            zone_peaks=dict(self.zone_peaks),
            zone_occupied_samples=dict(self.zone_occupied_samples),
            transitions=self.transitions,
            origin_zones=origins,
            first_destination=first_dest,
            reentries=len(reentry_tracks),
            reentry_tracks=reentry_tracks,
            zone_revisits=self.zone_revisits,
            track_durations={tid: st.last_t - st.first_t for tid, st in valid.items()},
            peak_in_view=self.peak_in_view,
        )
