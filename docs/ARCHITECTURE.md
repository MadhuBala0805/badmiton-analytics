# Architecture Deep-Dive

## Pipeline overview

```
Video
  │
  ▼
Frame Extraction         backend/utils/video_utils.py
  │
  ▼
Court Detection           backend/court/court_detector.py
  │
  ▼
Player Detection           backend/detection/player_detector.py  (standalone; tracker.py wraps detection+tracking together)
  │
  ▼
Player Tracking              backend/tracking/tracker.py
  │
  ▼
Person Re-Identification      backend/reid/embedder.py
  │
  ▼
Player Identity Assignment     backend/reid/identity_matcher.py
  │
  ▼
Pose Estimation                  backend/pose/pose_estimator.py
  │
  ▼
Shuttle Detection                  backend/shuttle/shuttle_detector.py
  │
  ▼
Court Mapping                        backend/court/court_detector.py (image_to_court)
  │
  ▼
Rally Detection                        backend/scoring/rally_detector.py
  │
  ▼
Point / Service / Score Update           backend/scoring/point_winner.py, backend/service/service_detector.py
  │
  ▼
Analytics Engine                           backend/analytics/stats_engine.py, heatmap.py
  │
  ▼
Dashboard                                    pages/4_Analytics_Dashboard.py
```

`backend/pipeline.py::MatchPipeline.run()` is the only place that encodes
this ORDER. Every arrow above is a plain-Python function/dataclass boundary
— no stage reaches into another stage's internals — so any box can be
replaced independently.

## Why some modules are "heuristic" and that's intentional

Per this project's engineering principles, we do not pretend a hard
computer-vision problem is solved just because doing so would simplify the
demo. Three sub-problems in this pipeline do not have small, reliable,
general-purpose open-source pretrained models, and the code says so loudly
in the relevant docstring rather than silently under-delivering:

### 1. Court line/keypoint detection (`backend/court/court_detector.py`)

Detecting a badminton court's exact boundary and service lines from
arbitrary, unconstrained YouTube footage (any camera angle, broadcast
overlays, lighting, partial occlusion by players) is a specialized
keypoint-detection problem. No generic pretrained model handles it well.
We ship a classical-CV heuristic (white-line HSV threshold → Hough
transform → convex-hull quadrilateral) that works acceptably on clean,
static, elevated broadcast shots and degrades gracefully (low confidence)
on messier footage — with a one-click manual 4-corner calibration in the
UI as the reliable fallback, which is what most real sports-analytics
tools use in practice for exactly this reason.

### 2. Shuttlecock detection (`backend/shuttle/shuttle_detector.py`)

The published state of the art (TrackNet / TrackNetV2 / V3) is a model
that must be trained on labeled shuttle-trajectory data from a specific
camera setup — there is no generic pretrained checkpoint that generalizes
across arbitrary amateur YouTube footage. We ship a background-subtraction
+ blob-filtering heuristic (small, bright, roughly-circular, fast-moving,
not overlapping a player) as a working baseline, documented with its known
failure modes (motion blur on smashes, false positives on other small
bright objects). The interface (`ShuttleDetector.detect()`) is stable so a
properly trained TrackNet checkpoint can be dropped in later without
touching any other module.

### 3. Rally / point-winner / service inference (`backend/scoring/`)

These require reasoning over noisy, compounding upstream signals (shuttle
position, court boundary, player positions) — there's no off-the-shelf
model for "did this rally end because of a fault or a winner." We ship a
transparent, weighted multi-signal fusion for rally boundaries and a
rule-based indicator system for point winners, and **both are conservative
by design**: `RallyStateMachine` and `infer_point_winner()` return explicit
confidence scores, and `point_winner.py` returns `winner=None,
needs_review=True` whenever the evidence isn't strong rather than emitting
a guess. The UI surfaces "Needs Review" states rather than silently
"deciding" for the user.

## Persistent player identity — the primary objective, in detail

This is the single most important property of the system, so it's worth
walking through explicitly.

**Two-level design** (`backend/reid/identity_matcher.py`):

1. **Enrollment gallery** — `player_id -> (K, D) reference embeddings`,
   generated once from the 2-3 uploaded images per player
   (`backend/reid/gallery.py`). This is ground truth we always compare
   against; it never drifts during the match.

2. **Track-identity memory** — while processing video, each tracker
   `track_id` (from `backend/tracking/tracker.py`, BoT-SORT/ByteTrack)
   accumulates a running, EMA-smoothed appearance embedding. Every frame,
   that running embedding is compared against the enrollment gallery via
   cosine similarity, and the best match becomes the track's `player_id` —
   but only if it clears both a similarity threshold AND a margin over the
   second-best candidate (`reid.match_threshold`, `reid.needs_review_margin`
   in `configs/config.yaml`). This prevents confidently mislabeling one
   player as another when they look similar.

**Handling breaks in tracking** (occlusion, crossing, temporary
disappearance): the underlying tracker's `track_id` WILL sometimes change
across a hard occlusion — that's a tracking limitation, not an identity
one. `IdentityMatcher.reconcile_reappearance()` compares a brand-new
`track_id`'s embedding against recently-lost tracks' running embeddings; if
it matches well enough, the new track is merged into the old one's history
so a player's stats/identity don't fragment across the gap. This is the
mechanism that fulfills "players temporarily disappear and reappear" and
"players partially occlude one another" from the spec.

**Court position is never used for identity** — `zone_of_point()` in
`court_detector.py` is only ever fed INTO analytics (front/back/left/right
court coverage stats), never used as an input to `IdentityMatcher`. This
directly satisfies "the player's identity should never depend only on
court position" and "side change handling."

**Confidence is always visible** — every `IdentityAssignment` carries a
`confidence` and `is_confident` flag, rendered in the Processing page's
live player table (✅ Confident / ⚠️ Needs Review) and stored per-frame so
the dashboard can audit low-confidence stretches later.

## Data flow at a glance

* `Detection` (detection/) → consumed only by `tracker.py`
* `TrackedBox` (tracking/) → consumed by `pipeline.py` for crops → `embedder.py`
* Embeddings (reid/) → consumed by `identity_matcher.py` only
* `IdentityAssignment` → consumed by `pipeline.py` to tag boxes with
  `player_id`, and by `stats_engine.py` to attribute positions/strokes
* `CourtModel.image_to_court()` → consumed by `pipeline.py` to convert
  pixel foot-points into court-space meters before handing to `stats_engine`
* `ShuttleObservation` → consumed by `rally_detector.py` (presence/motion
  signal) and `point_winner.py` (landing-position indicator)
* `RallyEvent` / `PointResult` / `ServiceState` → consumed by `pipeline.py`
  to update the running score and timeline, persisted via `database/store.py`
* Final `PlayerStats` → persisted as JSON, read by `pages/4_Analytics_Dashboard.py`
  and `reports/report_generator.py`

## Known POC limitations (by design, documented rather than hidden)

* Score updates in `pipeline.py`'s live loop are wired to rally
  start/end events but full point-winner attribution (calling
  `infer_point_winner` + `ServiceTracker.on_point_won` together with
  side-to-player mapping for singles vs. doubles) is left as the next
  integration step once real match footage is available to tune the
  heuristic thresholds in `configs/config.yaml` — the scoring modules
  themselves are complete and unit-testable in isolation.
* Stroke-type classification (smash/drop/clear/net/drive) has a data
  structure (`StrokeEvent`) and a stats aggregator ready
  (`stats_engine.record_stroke`), but the pose-to-stroke-type classifier
  itself is a placeholder seam — `pose_estimator.py::estimate_body_state`
  shows the pattern (simple geometric heuristic on keypoints) to extend.
* The database layer is intentionally JSON-file-based for POC simplicity;
  swap `backend/database/store.py` for a real DB without touching callers.
