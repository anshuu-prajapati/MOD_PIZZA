# MOD Pizza — Anonymous CCTV Store Analytics (MVP)

Turns recorded MOD Pizza outlet CCTV into store-intelligence numbers: how many
people come in, how busy the room is, which tables are used and for how long,
where people stop, where the queue forms, and how they move from the door
toward the counter or the seating.

It runs on recorded video, draws a live debug overlay while it processes, and
prints a full analytics summary in the terminal when it finishes.

## Privacy stance

This is **anonymous** analytics. The system deliberately does **not** implement:

- facial recognition or any biometric matching
- names, identities, or customer profiles
- purchase or order inference
- reliable re-identification of a person across cameras, visits, or days

Every person is a temporary integer track ID (`ID 42`) produced by ByteTrack,
which matches boxes on motion and overlap alone — it never looks at appearance.
IDs restart at 1 for every camera and every run and are thrown away when the
process exits. Nothing that could identify a person is computed or stored.

---

## Project purpose

Answer, from footage alone:

| # | Question | Where it is answered |
|---|---|---|
| 1 | How many people are entering | Door line-crossing on Front Entrance |
| 2 | When the store is busy or empty | Per-minute occupancy timeline + busy bands |
| 3 | How crowded the store is right now | Live occupancy at the last processed frame |
| 4 | Which tables are used and for how long | Per-table zone dwell, sittings, utilisation |
| 5 | How people move through the store | Zone-to-zone transition counts per camera |
| 6 | Whether people appear to return | Within-session re-entry + in-store return trips |
| 7 | Where people enter | Which door line fired + where tracks first appear |
| 8 | Where they stop or wait | Low-speed "stop" events by zone, seating excluded |
| 9 | Where queues form | Queue-zone length, wait time, peak time |
| 10 | Which areas they visit | Zone visit counts, distinct tracks, total time |
| 11 | Where they move toward counter/seating | First counter/queue vs seating zone reached after the door |

---

## Folder structure

```
MOD_PIZZA/
├── main.py                        entry point — python main.py
├── requirements.txt
├── README.md
├── mod-cctv-analytics-plan.md     the original architecture plan
│
├── config/
│   ├── settings.yaml              video paths, model, sampling, thresholds
│   └── zones/
│       ├── front_entrance.json    polygons + the front-door counting line
│       ├── beverage_station.json  polygons + staff mask
│       └── pos.json               queue/register polygons + staff masks
│
├── src/
│   ├── config.py                  loads settings.yaml + the zone files
│   ├── geometry.py                point-in-polygon, line crossing, speed
│   ├── detector.py                YOLO person detection + ByteTrack
│   ├── analytics.py               the metric engine (zones, dwell, flows)
│   ├── overlay.py                 the live debug drawing
│   ├── pipeline.py                per-camera read → detect → analyse loop
│   └── report.py                  terminal summary + CSV/JSON export
│
├── tools/
│   └── zone_editor.py             click-to-draw new zones on a real frame
│
├── .gitignore                     keeps .ai/, weights and footage out of git
├── models/                        YOLO weights (downloaded on first run)
├── Data/                          the source videos and reference frames
└── .ai/artifacts/                 output: events, summary.json, snapshots
```

---

## Requirements

- Python 3.10+ (developed on 3.11)
- ~2 GB disk for model weights and outputs
- An NVIDIA GPU is strongly recommended. On CPU the same run takes roughly
  10–20× longer; use `--minutes` to keep it manageable.

Python packages are listed in `requirements.txt`: `ultralytics`, `torch`,
`opencv-python`, `numpy`, `PyYAML`, `lap`. The pipeline runs on CPU or GPU
without any code change — `model.device: auto` picks whichever is present.

---

## Installation

```bash
cd MOD_PIZZA
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 1. GPU users: install torch FIRST, from pytorch.org, for your CUDA version.
#    Skipping this step is the single most common mistake - see the box below.
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. everything else
pip install -r requirements.txt
```

> **Install torch before requirements.txt, or you get the CPU build.**
> `pip install -r requirements.txt` on its own pulls the default PyPI wheel,
> which is CPU-only, and nothing warns you. The pipeline still runs correctly —
> it detects the CPU and says `device: cpu` on startup — but at roughly
> **1.5 frames/s instead of 20**, turning a 25-minute job into an overnight one.
>
> Check which one you have:
> ```bash
> python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
> ```
> `2.14.0+cpu False` means CPU-only. Reinstall with the cu128 index URL above.
> `2.11.0+cu128 True` means you are on the GPU.

The YOLO weights (`models/yolo11m.pt`, ~40 MB) download automatically the first
time you run the pipeline. No other setup is needed.

---

## How to configure the video

Everything lives in `config/settings.yaml`. Each camera is one block:

```yaml
cameras:
  - name: front_entrance                     # internal id, also the CLI name
    label: Front Entrance                    # shown in the window and report
    video: Data/Front Entrance.mp4           # relative to the project root
    zones: config/zones/front_entrance.json  # its zone file
    enabled: true                            # false = skip this camera
    role: primary
    start_time: "2026-09-20 18:00:00"        # wall clock at video t=0
```

- **Point at different footage** — change `video`. Any file OpenCV can open
  works. An RTSP/HTTP stream URL also works, but this MVP is built and tuned
  for recorded files.
- **`start_time`** is only used to print real clock times in the report. It was
  read off the burned-in timestamp in the footage. Remove it and the report
  falls back to `t+MM:SS` offsets.
- **Add a camera** — add a block and a matching zone file.

Useful knobs in the same file:

| Setting | Meaning |
|---|---|
| `processing.sample_fps` | frames analysed per second of footage (default 3) |
| `processing.max_minutes` | `null` for the whole file, or a number to cut it short |
| `processing.display` | `false` to run headless (much faster) |
| `model.conf` | person-detection confidence threshold |
| `model.imgsz` | inference resolution — lower is faster, worse on distant people |
| `analytics.table_seat_seconds` | how long in a table zone counts as a "sitting" |
| `analytics.stop_speed_px_per_s` | below this speed a person counts as stopped |
| `analytics.busy_thresholds` | the occupancy bands for EMPTY→PACKED |

---

## How to configure camera zones

Zones are pure configuration — one JSON file per camera in `config/zones/`.
Editing one never requires touching the pipeline.

```json
{
  "camera": "front_entrance",
  "frame_size": [1920, 1134],
  "lines": [
    { "id": "front_door", "label": "Front door",
      "points": [[788, 332], [958, 278]], "in_side": "right" }
  ],
  "polygons": [
    { "id": "table_center", "type": "table", "label": "Centre tables",
      "points": [[622,456],[1288,408],[1128,624],[648,712]],
      "seats": 10, "occupancy": true }
  ],
  "masks": [
    { "id": "staff_floor", "label": "Staff work floor",
      "points": [[1290,0],[1920,0],[1920,1134],[950,1134]] }
  ]
}
```

- **`points`** are pixel coordinates in the source frame (not the scaled
  display window).
- **`type`** drives the analytics: `entrance`, `queue`, `counter`, `table`,
  `beverage`, `exit`, `other`. Only `table` zones produce sitting/dwell rows;
  only `queue` zones produce queue metrics.
- **`occupancy`** — set `false` to exclude a zone from the store occupancy
  total. Because the three cameras see overlapping floor, **each physical area
  should be owned by exactly one camera**. That is why the POS camera masks the
  left seating: the Front Entrance camera already counts it.
- **`masks`** — any detection whose floor point lands inside a mask is dropped
  before analytics. This is how staff are kept out of the customer numbers.
- **`in_side`** on a line is `"right"` or `"left"` — which side of the A→B
  vector counts as walking *into* the store. Flip it if entries and exits come
  out swapped.

### Why zones must not overlap

Two polygons that cover the same patch of floor are occupied at the same
instant by the same person. Occupancy is de-duplicated by track ID so the head
count stays right, but before this was handled the movement section reported
hundreds of phantom `Left window bar -> Left high-tops` moves — one person's
foot point drifting across a shared edge. Two defences are in place:

- a zone visit only ends once the track has been outside it for
  `analytics.zone_exit_grace_seconds`, so boundary jitter does not split one
  visit into twenty;
- a move between two zones is only counted when the two visits do **not**
  overlap in time.

Even so, draw zones that touch rather than overlap. `--check-zones` makes this
easy to see.

### Checking and drawing zones

```bash
python main.py --check-zones          # renders the zones onto a real frame
                                      # -> .ai/artifacts/zones_<camera>.jpg

python tools/zone_editor.py pos --at 1200   # click new shapes onto the frame
```

In the editor: left-click adds a point, `Enter` finishes a shape, `t` cycles the
zone type, `l` toggles line mode, `u` undoes, `d` drops the last shape, `s`
appends everything to the camera's JSON, `q` quits.

---

## How to run

```bash
python main.py
```

That processes all three enabled cameras end to end and prints the summary.

Useful variations:

```bash
python main.py --minutes 5                  # first 5 minutes of each camera
python main.py --camera front_entrance      # one camera only (repeatable)
python main.py --no-display                 # headless, ~2x faster
python main.py --fps 2                      # analyse 2 frames/sec instead of 3
python main.py --save-video                 # also write the annotated video
python main.py --check-zones                # just render the zone config
```

While a window is open: `q` skips to the next camera, `Esc` stops everything
and reports on what was processed, `space` pauses, `s` saves a snapshot.

**Runtime.** At the default 3 fps sampling, one hour of footage is ~10,800
frames per camera. Throughput depends almost entirely on how much power the GPU
is allowed to draw:

| Situation | Measured on an RTX 5050 laptop GPU | 1 hour x 3 cameras |
|---|---|---|
| GPU power-capped (Windows power-saver, on battery) | 2.5 frames/s | ~3.5 hours |
| Same GPU unthrottled (plugged in, performance plan) | 14–20 frames/s | ~30 minutes |
| CPU only | well under 1 frame/s | overnight |

That is a 7x difference from a Windows power setting alone. If a run feels far
too slow, check:

```bash
nvidia-smi -q -d PERFORMANCE     # look for "SW Power Cap : Active"
```

Plugging the laptop in and switching Windows to a performance power plan is
worth more here than any code change.

**Always start with `python main.py --minutes 5 --no-display`** to confirm the
whole thing works before committing to a full run.

---

## Expected output

**While running** — one window per camera showing the source video with:

- zone polygons colour-coded by type, each labelled with its live count
- the front-door counting line in red with an arrow showing the "IN" direction
- staff masks shaded out; detections inside them drawn grey and ignored
- green boxes with `ID n` for every tracked person, a red dot at their floor point
- yellow trails showing recent movement paths
- a HUD with the clock, people in view, occupancy, entries so far, queue length,
  and the busy/quiet status

Example frames from each camera are saved in `.ai/artifacts/overlay_example_*.jpg`.
The POS one is worth a look: the MOD staff at the make line and the register are
drawn grey because they fall inside a staff mask, while the customer standing at
the register gets a green box and a movement trail.

**In the terminal when it finishes** — one section per question. This is the
real output from the shipped config on the supplied hour of footage
(2026-09-20, 18:00-19:00, dinner service), trimmed for length:

```
====================================================================================
 MOD PIZZA - CCTV CUSTOMER ANALYTICS SUMMARY
 Anonymous only: no faces, no names, no purchases, no customer profiles.
====================================================================================
 Footage window   : 2026-09-20  18:00 -> 19:00   (60.0 min per camera)
 Cameras          : Front Entrance, Beverage Station, POS
 Frames analysed  : 30,834  |  anonymous tracks: 2068

 1. HOW MANY PEOPLE ARE ENTERING
  Front-door crossings IN     : 45          OUT : 60      rate: 45 people/hour
    18:00-18:10  ####                     4
    18:20-18:30  ######################## 27      <- the dinner rush arriving
    18:50-19:00  #                        1

 2. WHEN THE STORE IS BUSY OR EMPTY
    18:00  ##########################        21.7  BUSY
    18:14  ################################  26.4  BUSY     <- peak minute
    18:42  ##############                    11.6  STEADY
    18:57  ########                           6.6  QUIET
  Busiest 10 min : 18:10-18:20  avg 24.0     Quietest : 18:50-19:00  avg 9.4
  Time spent     : EMPTY 0% | QUIET 5% | STEADY 43% | BUSY 52% | PACKED 0%

 3. HOW CROWDED THE STORE IS RIGHT NOW      10 people  ->  STATUS: STEADY

 4. WHICH TABLES ARE BEING USED AND FOR HOW LONG
  TABLE ZONE              CAMERA          SITTINGS  AVG DWELL  LONGEST  PEAK  IN USE
  Left window seating     Front Entrance        37     2m 40s   8m 01s     8     98%
  Centre tables           Front Entrance        18     3m 14s  21m 06s     6     68%
  Booth east              Front Entrance        12     1m 55s   3m 04s     5     91%
  Total sittings 92 | average dwell 2m 39s | longest 21m 06s
  Tables never used in this window : Long bar table

 5. HOW PEOPLE MOVE THROUGH THE STORE
    Order queue / menu  -> Counter walkway   20      Booth north -> Centre tables  12
    Order queue (make line) -> Register      10      (POS: order, then pay)

 6. WHETHER PEOPLE APPEAR TO RETURN
  Short-term door re-entries: 0    In-store return trips: 66
    back to Centre tables 11 | Order queue 8 | Communal high table 8 | Register 7

 7. WHERE PEOPLE ENTER        Front Entrance / front_door   45 entries

 8. WHERE THEY STOP OR WAIT   (seating excluded - that is section 4)
  Order queue / menu   170 stops   41m 08s waiting   avg 15s
  Beverage station      92 stops   24m 03s waiting   avg 16s

 9. WHERE QUEUES FORM
  Order queue / menu (Front Entrance) : 284 through, peak 6, 29 min with 2+ waiting
  Order queue (make line) (POS)       : 133 through, peak 3, avg wait 13s

10. WHICH AREAS THEY VISIT
  Left window seating  641 visits  3h 19m   Counter walkway  336 visits  1h 12m
  Right window seating 376 visits  1h 18m   Order queue      284 visits  1h 15m

11. WHERE THEY MOVE TOWARD THE COUNTER / SEATING
  Traffic split: counter/queue 912 visits (29%)  vs  seating/beverage 2258 (71%)

 KNOWN LIMITATIONS  ... printed in full at the end of every run
====================================================================================
```

**Files written to `.ai/artifacts/`:**

| File | Contents |
|---|---|
| `zone_visits.csv` | camera, track_id, zone_id, enter_s, exit_s, dwell_s |
| `line_events.csv` | camera, track_id, line_id, direction, t_s |
| `occupancy_timeline.csv` | camera, bin_start_s, mean_people |
| `stop_events.csv` | camera, track_id, zone_id, start_s, duration_s |
| `summary.json` | the headline numbers, machine-readable |
| `terminal_report.txt` | the full terminal summary, saved verbatim |
| `zones_<camera>.jpg` | the zone config rendered on a real frame |
| `overlay_example_<camera>.jpg` | a sample frame with boxes, IDs, trails and HUD |

---

## Known limitations

These are real and they matter when reading the numbers. They are also printed
at the bottom of every run.

1. **Front door only.** The side door has no camera in this phase, so total
   entries **undercount** by whatever share uses that door.
2. **The front door is distant and oblique.** Two or three people walking in
   together can merge into one detection, so groups undercount. The entry number
   is an indicator, not an audited count.
3. **Track IDs are per-camera and temporary.** A person moving from the Front
   Entrance view into the Beverage Station view becomes a *new* anonymous track.
   Movement paths are solid within one camera and coarse store-wide; they are
   not stitched together, because stitching would need appearance re-ID.
4. **Returning customers are not computed.** Section 6 reports only
   within-session behaviour: the same live track re-crossing the door, and
   people leaving a zone and coming back (a drink refill, a second trip to the
   counter). Recognising someone who returns tomorrow is out of scope by design.
5. **Staff exclusion is spatial, not personal.** Employees are removed by
   masking fixed floor areas on the POS and Beverage cameras. A staff member who
   steps onto the customer floor will be counted as a customer.
6. **Seated people are detected from the torso.** Their box bottom sits at the
   seat or table edge rather than the floor, so table polygons are drawn
   generously to compensate. A polygon drawn too tightly will miss sittings.
7. **Occupancy depends on zone discipline.** The store total is the sum of each
   camera's own zones, with overlapping floor assigned to exactly one camera.
   Redrawing the zone files changes the totals — that is expected, and it is why
   `--check-zones` exists.
8. **Sampling is 3 fps, not 20.** Fast movement between samples can break a
   track into two IDs, which slightly inflates the track count and can miss a
   very fast door crossing. Raise `sample_fps` for accuracy, lower it for speed.
9. **Not validated against ground truth yet.** Before trusting the entry count,
   hand-count two or three 10-minute windows and compare, as the plan document
   specifies.
10. **Occupancy bands are guesses.** `analytics.busy_thresholds` were set for a
    store of this size and should be tuned once a few days of footage exist.
