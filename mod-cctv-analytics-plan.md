# MOD Pizza — CCTV Customer-Intelligence System
## Technical Architecture Plan (v1)

**Status:** Planning / pre-build. No code exists yet — this is the blueprint we build against.
**Privacy stance:** Anonymous store intelligence only. No facial recognition, no names, no purchase data, no cross-day identity.

---

## 1. What we are trying to achieve

Answer one question in near-real-time, and later trigger screen campaigns from the answer:

> **"What is happening inside this MOD store right now?"**

Concretely, the system should produce metrics such as:

- Current occupancy and a busy/empty status
- Entries over time (by hour / day)
- Table usage and dwell time
- Where people stop, wait, and queue
- Which zones get visited and for how long
- Dominant movement directions (e.g. entrance → counter → seating)

Everything is derived from **anonymous** person detection. Each tracked person gets a throwaway ID (`person_001`, `person_002`) that is valid only within a single camera and a single session. It never represents a real identity.

---

## 2. Scope for this phase

### In scope — 3 cameras
| Camera | Role in the system |
|---|---|
| **Front Entrance** | Primary camera. Front-door entries, main seating, counter-facing direction, occupancy baseline. |
| **Beverage Station** | Beverage/soda station, online-pickup counter, right-side high-tables + booths, movement in that half of the room. |
| **POS** | Order point and the queue floor in front of the counter. Staff area behind the line is masked out. |

### Out of scope (for now)
- **Secondary Entrance camera** — dropped. The side door is therefore invisible to the system.
- **Safe camera** — back-office/storage, zero customer value. Never enters the analytics pipeline.

### Delivery approach
Build and validate on **recorded video first** — one full day, then one full week — before any live-stream/real-time work. Offline footage can be re-run, ground-truthed, and tuned; real-time cannot.

---

## 3. Camera angles — capability read

Assessment based on the actual reference frames (empty store, early morning — ideal for drawing zones).

### Front Entrance (primary)
- Elevated dining-room view. **Excellent table/seating coverage.**
- The front glass doors sit **far and oblique** in the background — usable for a line-crossing entry counter, but not a clean head-on doorway. Expect modest accuracy and group-entry undercounting.
- Sees the top edge of the service counter → supports "toward counter vs toward seating" direction.

### Beverage Station
- Wide view of the beverage/soda station, pickup counter, and the right-side seating block the Front Entrance angle misses.
- Strong for beverage-zone activity and that seating cluster. No entrance value.

### POS
- The updated busy-hour frame confirms a **clear stretch of open customer floor directly in front of the counter** — this is where a queue forms, so **queue detection is feasible here.**
- Condition: the **staff work area behind the line must be masked** so employees are never counted as customers.

---

## 4. The 11 metrics — how each is derived

| Metric | Method | Source camera | Confidence |
|---|---|---|---|
| People entering | Line-crossing events at the front-door line | Front Entrance | ⚠️ Modest (door is distant/oblique; front door only) |
| Busy / empty status | Threshold bands on total occupancy | derived | ✅ High |
| How crowded right now | Sum of per-camera zone occupancy (non-overlapping) | all three | ⚠️ Medium (depends on zone partition discipline) |
| Tables used + how long | Per-table polygon; foot-point inside → occupied; dwell = duration inside | Front Entrance + Beverage | ✅ High |
| How people move | Per-camera foot-point trajectories over time | all three | ⚠️ Per-camera solid; store-wide coarse |
| Whether people return | **Not computed** — impossible without face/appearance re-ID | none | ❌ Documented limitation |
| Where people enter | Which door-line fired | Front Entrance only | ⚠️ Front door only |
| Where they stop / wait | Dwell-per-zone heatmap | all three | ✅ High |
| Where queues form | People inside queue polygon + duration | POS | ✅ Feasible (needs staff mask) |
| Areas visited | Zone-entry event counts | all three | ✅ High |
| Toward counter / seating | Transition sequences counter-zone ↔ seating-zone | Front Entrance | ✅ Good |

---

## 5. System architecture

Six independent stages. Each is a separate module so any one can be swapped or debugged without touching the others. This keeps **detection/tracking separate from analytics, and analytics separate from campaign rules.**

```
[1] INGEST        recorded video per camera/day   →  sampled frames
        ↓
[2] DETECT        person detection per frame       →  boxes + confidence
        ↓
[3] TRACK         per-camera tracker               →  temporary IDs + paths
        ↓
[4] ZONE MAP      foot-point → polygons + lines     →  who is in which zone, when
        ↓
[5] ANALYTICS     derive the 11 metrics
        ↓
[6] STORE+REPORT  event log → hourly/daily rollups → dashboard
        ↓
   (later)  BUSINESS RULES → CAMPAIGN TRIGGERS  (reads metrics only)
```

**Design rules**
- Detection/tracking never knows what a zone *means*.
- Zones never know business meaning.
- Analytics never touches pixels.
- Campaign rules sit downstream and read only metrics.

### Double-counting control
All three cameras see overlapping floor. To avoid counting one person twice, **each physical area is owned by exactly one camera** (its "authority"). Occupancy is the sum of non-overlapping zone counts, never a raw sum of detections across cameras.

---

## 6. Zones are configuration, not code

One JSON file per camera, drawn once on the empty reference frames. Editing a zone never requires editing the pipeline.

```json
{
  "camera": "front_entrance",
  "lines": [
    { "id": "front_door", "points": [[x1,y1],[x2,y2]] }
  ],
  "polygons": [
    { "id": "seating_a", "type": "table",   "points": [[..],[..],[..]] },
    { "id": "counter",   "type": "counter", "points": [[..],[..],[..]] },
    { "id": "queue",     "type": "queue",   "points": [[..],[..],[..]] }
  ],
  "masks": [
    { "id": "staff_area", "points": [[..],[..],[..]] }
  ]
}
```

**Zone vocabulary:** `entrance_front`, `counter`, `queue`, `seating`, `beverage`, `exit`, `other` (plus `staff_area` masks). Each camera has its own file.

---

## 7. Data model

Append-only events, rolled up into aggregates.

| Table | Fields |
|---|---|
| `detections` | ts, camera, track_id, bbox, foot_x, foot_y, confidence |
| `zone_events` | camera, track_id, zone_id, enter_ts, exit_ts, dwell_s |
| `line_events` | camera, track_id, line_id, direction, ts |
| `occupancy_ts` | ts, camera/zone, count |
| `aggregates` | hour, day, metric, value |

- Raw events → **Parquet** (or SQLite for a small first run).
- Rollups → **pandas / DuckDB**.
- `track_id` is temporary and camera-local. Never an identity.

---

## 8. Technology stack

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.11** | Ecosystem for CV + data |
| Video IO | **OpenCV** | Read recorded files, sample frames |
| Detection | **Ultralytics YOLO** (v8/v11, person class only) | Fast, accurate, well supported |
| Tracking | **ByteTrack** | Fast, appearance-free, no identity — fits privacy stance |
| Zones / lines | **`supervision`** (Roboflow) | Ready-made polygon-zone + line-crossing utilities and debug annotators |
| Storage | **DuckDB + Parquet** | Cheap columnar analytics on event logs |
| Rollups | **pandas** | Hourly/daily aggregation, trends, peaks |
| Dashboard | **Streamlit** (or static HTML) | Simple day/week report views |

Optional / experimental later: appearance **re-ID** for cross-camera linking — imperfect, not required for v1.

---

## 9. Day → Week workflow

### Day 1 — validation run
1. Process one full day of the three recorded streams.
2. Sample at **~5 fps** (optionally higher only in the door region for cleaner entry counts). A full day at 5 fps across 3 cameras is on the order of a million frames — run it as an **overnight batch**; budget for a GPU.
3. **Ground-truth check:** manually count entries in two or three 10-minute windows and compare to the system. This is the gate before trusting any number.

### Week run
1. Once zones are tuned and Day 1 looks right, batch the full week.
2. Produce hourly/daily aggregates, peak-period detection, and trend graphs.
3. Feed the metrics into the dashboard.

---

## 10. Dashboard targets

- Current people inside
- Today's total entries + front-entrance count (side door not covered)
- Busy / quiet status
- Occupied vs empty tables, average dwell time
- Queue size + duration, peak queue periods
- Zone traffic and dwell heatmap
- Dominant movement flows (e.g. entrance → counter → seating)
- Entries-by-hour graph

---

## 11. Known limitations (must stay visible in output)

1. **Front-door entries only** — the side door is uncovered this phase, so total entries **undercount** by the side-door share.
2. **Entry-line accuracy is modest** — the front door is distant/oblique; groups entering together will be undercounted.
3. **Cross-camera person paths are coarse.** Within one camera, trajectories are solid; stitching one person across cameras needs re-ID and is optional/experimental. No promise of seamless store-wide paths.
4. **Returning customers are not computed** — impossible without face/appearance identifiers, which are deliberately excluded.
5. **Staff vs customer** on POS/counter relies on spatial masking; a staff member stepping onto the customer floor can be miscounted.
6. **Empty-store validation only so far** — accuracy figures are estimates until a busy-period clip is processed.

---

## 12. Campaign-trigger layer (later phase)

Kept strictly downstream of analytics. The camera only detects conditions; a **rules engine** decides the campaign.

```
Examples
  3–5 PM  AND occupancy low        → slow-hour campaign
  lunchtime AND occupancy high     → lunch campaign
  large group in seating           → shareable-items campaign
  queue high                       → quick-service messaging
```

Rules read metrics only — they never see pixels or tracks.

---

## 13. What's needed to start building

1. **A busy-period sample clip (10–15 min) per camera** — to tune detection/tracking and confirm the queue zone before committing.
2. **Hardware confirmation** — is a GPU available for the batch runs?
3. **Sign-off on the stack** above (or a stated preference).

### First concrete deliverable
Zone-config files + a **detection/tracking debug overlay** on the Front Entrance camera — so the pipeline can be *seen* working before the analytics layer is built on top.
