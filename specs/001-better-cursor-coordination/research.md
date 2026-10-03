# Research: Better Cursor Coordination

**Feature**: 001-better-cursor-coordination
**Date**: 2026-10-03

## Decisions

### 1. Pipeline Stage Ordering

**Decision**: MediaPipe Landmarks → Virtual Plane Projection → Smoothing (One Euro Filter on plane u,v) → Velocity Limiting → Screen Mapping → uinput

**Rationale**: 
- Architecture audit Bug 1 proved smoothing before projection destroys head-relative geometry
- Projection converts camera coords → plane coords using eye midpoint + fingertip ray intersection
- One Euro Filter on plane coordinates preserves geometric relationship while providing adaptive smoothing
- Velocity limiting on screen-space deltas prevents runaway cursor
- This ordering matches the "Correct Flow" documented in ARCHITECTURE_AUDIT.md

**Alternatives Considered**:
- Smoothing in camera space then projection: Rejected (breaks head-relative, causes Bug 1)
- Projection → EMA only: Rejected (One Euro adapts to speed, EMA is fixed)
- No smoothing: Rejected (unacceptable jitter per Principle VII)

---

### 2. Virtual Plane Projection Mathematics

**Decision**: Ray-plane intersection using eye midpoint as origin, index fingertip as direction vector, virtual plane at configurable distance/width in head coordinate space.

**Rationale**:
- MediaPipe provides 3D landmarks (x,y,z normalized) for both face and hand
- Eye midpoint = average of left/right eye landmarks
- Ray direction = fingertip_3D - eye_midpoint_3D
- Plane: centered at head forward vector * distance, sized by width/height
- Intersection gives (u,v) in [0,1] plane coordinates
- This is implemented in `virtual_plane.py` and `head_coords.py` but was being bypassed

**Alternatives Considered**:
- Simple homography camera→screen: Rejected (no head-relative invariance)
- Kalman filter on camera coords: Rejected (same Bug 1 problem)
- Direct landmark mapping: Rejected (no perspective correction)

---

### 3. Reference Point Update Logic

**Decision**: 
- Head-relative mode: reference_point = current_projection_position every frame
- Legacy mode: reference_point updates only when leaving dead zone (existing behavior)

**Rationale**:
- Architecture audit Bug 2: reference point was only updated when leaving dead zone
- In head-relative mode, relative movement should be from current projected position
- Legacy mode uses dead zone as "anchor" for absolute positioning

**Alternatives Considered**:
- Always update reference point: Rejected (breaks legacy mode UX)
- Never update reference point: Rejected (cursor drifts)

---

### 4. Velocity Limiter Correction

**Decision**: New `VelocityLimiter` class that:
- Accepts position (x,y) and timestamp
- Computes velocity internally: v = (pos - prev_pos) / dt
- Limits velocity magnitude to max_velocity
- Returns limited position delta

**Rationale**:
- Architecture audit Bug 3: existing `VelocityLimiter.limit(dx, dt)` treated dx as position
- Correct approach: track position, compute velocity, limit, return delta
- Separate class avoids confusion with old API

**Alternatives Considered**:
- Fix existing VelocityLimiter in place: Rejected (breaking change, unclear contract)
- Remove velocity limiting: Rejected (safety requirement Principle XVIII)

---

### 5. Coordinate Contract Enforcement

**Decision**: Extend `coordinate_contract.py` with explicit pipeline stage validation:
- Stage 1 output: camera coords [0,1] (validated)
- Stage 2 output: plane coords [0,1] (validated)
- Stage 3 output: plane coords [0,1] smoothed (validated)
- Stage 4 output: screen pixels (clamped)

**Rationale**:
- Constitution Principle XVI: explicit contracts between pipeline stages
- Enables independent testing of each stage
- Catches contract violations early

**Alternatives Considered**:
- Runtime assertions only: Rejected (no compile-time/IDE support)
- Pydantic models for each stage: Rejected (overhead for real-time loop)

---

### 6. Graceful Degradation Strategy

**Decision**: When face tracking unavailable (no face, low confidence, lost):
- Log warning once per session
- Automatically switch to legacy camera-coordinate mode
- Show UI notification (non-blocking)
- Re-attempt head-relative when face confidence recovers

**Rationale**:
- Constitution Principle XIX: graceful failure on device unavailability
- Face tracking can fail due to lighting, occlusion, angle
- Legacy mode still functional for basic use

**Alternatives Considered**:
- Hard fail: Rejected (violates Principle XIX)
- Disable cursor entirely: Rejected (too aggressive)
- Synthetic head pose: Rejected (unreliable, complex)

---

### 7. Configuration Parameters

**Decision**: Add to `TrackingConfig` (schema.py):
```python
use_head_relative: bool = True
virtual_plane_distance: float = 0.30  # meters (30cm in front)
virtual_plane_width: float = 0.40     # meters (40cm)
virtual_plane_height: float = 0.25    # meters (25cm)
head_confidence_threshold: float = 0.5
projection_smoothing: SmoothingAlgorithm = ONE_EURO
one_euro_min_cutoff: float = 1.0
one_euro_beta: float = 0.0
one_euro_d_cutoff: float = 1.0
reference_point_update_mode: Literal["every_frame", "dead_zone_exit"] = "every_frame"
```

**Rationale**:
- All parameters profile-configurable per Principle XIII
- Defaults from IDEA2.md benchmarks
- `reference_point_update_mode` makes Bug 2 fix configurable

**Alternatives Considered**:
- Hardcoded constants: Rejected (violates Principle XIII)
- Separate config file: Rejected (existing schema/profiles system works)

---

### 8. Testing Strategy

**Decision**: 
- Unit tests: each pipeline stage with synthetic inputs
- Property tests: coordinate transform round-trip, head-movement invariance
- Integration: full pipeline camera→cursor with synthetic landmarks
- Benchmarks: latency, FPS, CPU, jitter metrics

**Rationale**:
- Constitution Principle XVII: testability and production quality
- Synthetic tests are deterministic and fast
- Head-movement invariance is the key correctness property

**Alternatives Considered**:
- Only integration tests: Rejected (hard to debug, slow)
- Manual testing only: Rejected (not repeatable)

---

## Open Questions (None - all resolved)

All technical decisions made. Ready for Phase 1 design.