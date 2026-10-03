# AirMouse Debugging — Internal Task Log

## Phase 1 — Architecture Map

Key modules and their roles:
- `airmouse/vision/hand_tracker.py` — MediaPipe HandLandmarker (IMAGE mode). Produces `Hand` with 21 landmarks + `index_tip` 3D (camera-normalized [0,1]).
- `airmouse/vision/face_tracker.py` — MediaPipe FaceLandmarker (VIDEO mode). Produces `Face` with `eye_midpoint`, `nose_tip`, `forehead` landmarks.
- `airmouse/vision/head_coords.py` — `HeadCoordinateSystem`: origin=eye midpoint, +Z forward (nose), +X right, +Y up. Transform matrices camera↔head.
- `airmouse/vision/virtual_plane.py` — `VirtualDisplayPlane`: 40×25cm plane at z=0.30m in head coords. `ray_plane_intersection_head()` and `ray_plane_intersection()`. `point_to_normalized()` returns (u,v) in [0,1] or None.
- `airmouse/vision/tracking_processor.py` — Pipeline orchestrator. `OneEuroFilter` on (u,v) AFTER projection. `VelocityLimiter` for deltas. `get_cursor_movement()` returns (dx,dy).
- `airmouse/control/coordinate_contract.py` — `CoordinateTransformer`: camera[0,1] → plane[0,1] → screen[0,1] → pixels.
- `airmouse/vision/confidence.py` + `tracking_status.py` — Confidence states HIGH/MEDIUM/LOW/LOST. `TrackingStatus` with phases STARTING/TRACKING/DEGRADED/FROZEN/LOST/REACQUIRING/STABILIZING.
- `airmouse/ui/safety.py` — `SafetyManager` with corner_escape, focus_loss_pause, velocity_limit, inactivity, emergency_disable.
- `airmouse/control/main_loop.py` — `AirMouseController` ties it all together. On `NO_HAND` or `LOST_TRACK`: release_all, reset cursor/gestures/tracking_processor → hard freeze.
- `airmouse/input/linux_input.py`, `uinput_mouse.py`, `ydotool.py` — input backends.

## Phase 2 — Root Cause of projection_intersection_failed

**The log shows u/v out of bounds on every frame. Two issues identified:**

### Issue A: Geometry correctness for rays pointing away from plane
In `RayPlaneIntersectionHead` (virtual_plane.py:199-225), plane is at z=+0.30m, normal=(0,0,-1).
The ray originates at eye midpoint (origin→z≈0 in head coords) pointing toward fingertip.
- In MediaPipe **camera** coords: +Z is INTO the screen (away from camera). The nose tip has z≈0 or slightly negative. A fingertip in front of the face has a MORE NEGATIVE z than the eye.
- When transformed to **head** coords: forward=+Z. The fingertip in front of the face should have z>0 in head coords, i.e. positive z — pointing TOWARD the plane at z=0.30.
- BUT: `HeadCoordinateSystem.forward = normalize(nose - eye)`. In camera coords nose.z < eye.z (both negative, nose more negative). After transform, the head forward vector ends up with z-component direction that depends on the camera→head matrix.

The repro script shows: for a centered face with eye=(0.5,0.4,-0.3), nose=(0.5,0.45,-0.35), the head forward ends up as (0, 0.707, -0.707) in CAMERA coords. This means head forward z = -0.707 in camera coords (pointing INTO the scene, away from user). 

In head coordinates, the fingertip (0.95, 0.05, -0.9) camera maps to head tip (0.45, 0.67, 0.177) — z is POSITIVE (forward). Good. The plane is at z=+0.30, the fingertip at z≈0.18. Ray from eye(z≈0) through fingertip(z≈0.18) has positive-z direction → intersects plane. u/v computed correctly.

**The REAL problem (Issue B):** The out_of_bounds values like u=-3.339, v=2.000 occur when the hand is moved FAR outside the plane's physical extent (hand reaches beyond the 40×25cm virtual plane). This is **normal** behavior — the hand can literally be outside the plane's x/y bounds while still having a valid ray that intersects the plane's z=constant surface. The clamping in `point_to_normalized` is correct, but it logs a WARNING on every single frame, creating log spam.

### Issue B: CONFIDENCE-DROP causing full freeze
In `main_loop.py` lines 1005-1024: `LOST_TRACK` state (returned by TrackingProcessor when projection returns None) causes a **full reset** of cursor_controller, gesture_recognizer, and tracking_processor. This happens on every frame where the ray is parallel to the plane or points away — e.g., hand at same depth as eyes, or hand above/below the eye plane.

`ray_plane_intersection_head` returns None when:
- `abs(ray_direction_head[2]) <= 1e-6` (ray parallel to plane — hand at eye depth)
- `t < 0` (intersection behind ray origin — hand between eye and plane, or behind)

These conditions happen frequently with natural hand movements (hand moving up/down while keeping distance, or slight depth changes). Each None → LOST_TRACK → full reset → next frame re-creates everything → ping-pong freeze.

### Issue C: No hysteresis in confidence
`confidence.py` `_classify` switches state every frame based on blended value. No temporal hysteresis. Combined with MediaPipe border confidence (~0.5), this causes rapid LOST↔TRACKING oscillation.

## Phase 3 — Fix Plan

### Fix 1: Distinguish "out of bounds" from "invalid geometry" in virtual_plane.py
- When the ray-plane intersection SUCCEEDS (valid t>0) but the point is outside the [−w/2,w/2]×[−h/2,h/2] bounds → this is a NORMAL condition (hand reached beyond virtual plane). Clamp silently to boundary, log at DEBUG only.
- When the intersection is truly invalid (parallel ray, t<0) → return a structured invalid result WITH the clamped fallback position, so the tracking loop degrades gracefully instead of freezing.

### Fix 2: Don't hard-freeze on LOST_TRACK, use hold-and-retry
In main_loop.py: when LOST_TRACK but hands were detected with adequate confidence, HOLD the last valid cursor position rather than resetting everything. Only full-reset on NO_HAND (no detection at all).

### Fix 3: Confidence hysteresis
Add consecutive-frame requirements before state transitions: require N consecutive LOW before LOST, N consecutive HIGH before TRACKING.

