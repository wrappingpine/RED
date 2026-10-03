# Feature Specification: Better Cursor Coordination

**Feature Branch**: `001-better-cursor-coordination`

**Created**: 2026-10-03

**Status**: Draft

**Input**: User description: "Better cursor coordination - eliminate mirrored movement and fix projection pipeline ordering"

## Clarifications

### Session 2026-10-03

- **Q:** What exactly counts as the start and end points for measuring end-to-end latency in SC-003 ("camera → cursor")? (SC-003) → **A:** Measure from camera frame capture timestamp to uinput write completion timestamp — full pipeline latency including frame acquisition, landmark inference, projection, smoothing, velocity limiting, and cursor injection.
- **Q:** When face tracking degrades (FR-010), what mechanism should notify the user and should recovery be automatic? (FR-010) → **A:** Log a structured WARNING event, set internal mode flag, emit a SafetyEvent for tray notification, and auto-recover when face confidence exceeds threshold for 5 consecutive frames.
- **Q:** How should the system switch between head-relative and legacy mapping modes at runtime? (FR-006) → **B:** Mode is determined by config.use_head_relative at startup; runtime toggle via global hotkey Super+Alt+M per Principle X.
- **Q:** When the virtual plane intersection fails (ray parallel to plane), what should the system do beyond clamping and logging? → **A:** Clamp to nearest plane boundary, log structured WARNING event with tag `projection_intersection_failed`, and continue with the clamped position without interrupting cursor tracking.
- **Q:** Should the One Euro Filter and EMA smoothing be implemented as interchangeable classes behind a common interface, or as two separate code paths selected at startup? (FR-002) → **A:** A common `SmoothingFilter` abstract base class with `OneEuroFilter` and `EmaFilter` concrete implementations, selected at startup via config and swappable at runtime for testing.
- **Q:** What angular ranges should SC-002 head-movement invariance tests cover for yaw and pitch? (SC-002) → **B:** ±30° yaw, ±20° pitch (more restrictive pitch limit reflects real-world face tracking degradation; matches quickstart.md validation Scenario 2).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Natural Hand-to-Cursor Mapping (Priority: P1)

When the user moves their index finger to the right, the cursor moves to the right. When the user moves their index finger up, the cursor moves up. The cursor follows the hand naturally without mirroring or inversion.

**Why this priority**: This is the fundamental UX requirement. If cursor movement doesn't match hand movement, the system is unusable regardless of other features.

**Independent Test**: Can be fully tested by moving hand in four cardinal directions (up, down, left, right) and verifying cursor moves in the same direction. Delivers basic usable cursor control.

**Acceptance Scenarios**:
1. **Given** hand tracking is active and user raises index finger, **When** user moves index finger right, **Then** cursor moves right on screen
2. **Given** hand tracking is active, **When** user moves index finger up, **Then** cursor moves up on screen
3. **Given** hand tracking is active, **When** user moves index finger diagonally, **Then** cursor follows same diagonal direction
4. **Given** user enables "invert_x" or "invert_y" in config, **When** hand moves, **Then** cursor moves in inverted axis as configured

---

### User Story 2 - Head-Relative Virtual Plane Projection (Priority: P1)

The cursor position is computed by projecting a ray from the eye midpoint through the index fingertip onto a virtual plane positioned in front of the user's head. This projection happens BEFORE any smoothing, so the geometric relationship between head and hand is preserved.

**Why this priority**: The architecture audit (Bug 1) identified that smoothing was applied to raw camera coordinates before projection, destroying the head-relative geometry. This is the core architectural fix.

**Independent Test**: Can be tested by fixing hand position relative to head, moving head, and verifying cursor stays stable (head-movement invariance). Delivers head-relative tracking that works regardless of head position.

**Acceptance Scenarios**:
1. **Given** head-relative mode enabled, **When** user moves head left while keeping hand fixed relative to head, **Then** cursor stays stationary on screen
2. **Given** head-relative mode enabled, **When** user moves head right while keeping hand fixed relative to head, **Then** cursor stays stationary on screen
3. **Given** head-relative mode enabled, **When** user moves hand relative to head, **Then** cursor moves proportionally on virtual plane
4. **Given** head-relative mode disabled (legacy), **When** user moves hand, **Then** cursor maps directly from camera coordinates

---

### User Story 3 - Smooth Cursor with One Euro Filter on Plane Coordinates (Priority: P1)

Smoothing (One Euro Filter) is applied to the normalized (u,v) coordinates on the virtual plane AFTER projection, not to raw camera landmarks. The filter adapts to movement speed: more smoothing at low speed for stability, less at high speed for responsiveness.

**Why this priority**: Fixes the pipeline ordering bug (Bug 1). The One Euro Filter must operate on plane coordinates to preserve head-relative geometry while providing adaptive smoothing.

**Independent Test**: Can be tested with synthetic landmarks: feed known noisy input, verify output is smoothed on plane coordinates, and that filter adapts to speed changes. Delivers smooth, responsive cursor without jitter.

**Acceptance Scenarios**:
1. **Given** noisy hand landmarks, **When** pipeline processes frames, **Then** cursor movement is smooth without visible jitter
2. **Given** slow hand movement, **When** One Euro Filter processes, **Then** output has high smoothing (low cutoff)
3. **Given** fast hand movement, **When** One Euro Filter processes, **Then** output has low smoothing (high cutoff) for responsiveness
4. **Given** One Euro Filter disabled, **When** EMA smoothing enabled, **Then** EMA applies to plane coordinates

---

### User Story 4 - Correct Reference Point Update for Head-Relative Mode (Priority: P2)

In head-relative mode, the reference point for relative cursor movement updates every frame to the current projection position. In legacy mode, it updates only when leaving the dead zone.

**Why this priority**: Architecture audit Bug 2 - the reference point was only updated when leaving dead zone, breaking relative movement in head-relative mode.

**Independent Test**: Can be tested by moving hand in small circles within dead zone in head-relative mode - cursor should track smoothly. Delivers correct relative movement behavior.

**Acceptance Scenarios**:
1. **Given** head-relative mode, **When** hand moves small amount within dead zone, **Then** reference point updates each frame, cursor moves smoothly
2. **Given** legacy mode, **When** hand moves within dead zone, **Then** reference point stays fixed until dead zone exited
3. **Given** head-relative mode, **When** hand exits dead zone, **Then** movement is relative to current projection position

---

### User Story 5 - Velocity Limiter Fix (Priority: P2)

The VelocityLimiter receives deltas (dx, dy) but incorrectly treats them as positions. A corrected velocity limiter operates on positions or properly handles deltas.

**Why this priority**: Architecture audit Bug 3 - VelocityLimiter data contract mismatch causes incorrect velocity limiting.

**Independent Test**: Can be tested by feeding known velocities, verifying limiter caps at configured max_velocity. Delivers correct velocity limiting without artifacts.

**Acceptance Scenarios**:
1. **Given** velocity limiter configured with max_velocity=2000 px/s, **When** hand moves producing 5000 px/s equivalent, **Then** cursor velocity capped at 2000 px/s
2. **Given** sudden hand jerk, **When** velocity limiter processes, **Then** no overshoot beyond max_velocity
3. **Given** precision mode (max_velocity=500 px/s), **When** hand moves fast, **Then** cursor capped at 500 px/s

---

### Edge Cases

- What happens when face tracking is lost but hand tracking continues? → Fall back to legacy camera-coordinate mapping with warning (structured WARNING event with tag `face_tracking_lost`, tray notification via SafetyEvent). Auto-recover when face confidence exceeds threshold for 5 consecutive frames.
- What happens when virtual plane intersection fails (ray parallel to plane)? → Clamp to nearest plane boundary, log structured WARNING event with tag `projection_intersection_failed`, and continue with the clamped position without interrupting cursor tracking.
- How does system handle sudden camera resolution change? → Recompute normalization, reset filters
- What if head landmarks are low confidence? → Disable head-relative mode automatically, notify user
- How does system behave on multi-monitor setups? → Virtual plane maps to primary monitor; cursor clamping at screen edges

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST project index fingertip through eye midpoint onto virtual plane before any smoothing
- **FR-002**: System MUST apply One Euro Filter (or EMA) to normalized plane coordinates (u,v), not camera coordinates
- **FR-003**: System MUST update reference point every frame in head-relative mode
- **FR-004**: System MUST update reference point only when leaving dead zone in legacy mode
- **FR-005**: VelocityLimiter MUST correctly limit cursor velocity given position deltas (dx, dy) and time delta
- **FR-006**: System MUST support both head-relative and legacy (camera-coordinate) mapping modes. Mode is determined by config.use_head_relative at startup; runtime toggle via global hotkey Super+Alt+M per Principle X.
- **FR-007**: System MUST invert X/Y axes when configured via invert_x/invert_y settings
- **FR-008**: System MUST clamp cursor to screen boundaries
- **FR-009**: System MUST log pipeline stage timings for latency measurement as structured JSON with fields: `stage_name` (string), `duration_ms` (float), `timestamp` (ISO 8601), `frame_id` (int). Logs written to `airmouse/debug/diagnostics.py` via non-blocking async handler.
- **FR-010**: System MUST gracefully degrade to legacy mode when face tracking unavailable. On degradation: log structured WARNING event with tag `face_tracking_lost`, set internal mode flag to legacy, emit SafetyEvent for tray/UI notification. Auto-recover to head-relative mode when face confidence exceeds threshold for 5 consecutive frames.

### Key Entities

- **TrackedHand**: Hand landmarks, confidence, handedness, smoothed plane position, velocity, reference point
- **VirtualPlane**: Distance from head, width/height, coordinate transform matrix
- **ProjectionResult**: Plane coordinates (u,v), screen coordinates (x,y), validity flag
- **CursorState**: Current screen position, velocity, smoothing state, mode (head-relative/legacy)
- **SmoothingFilter**: Abstract base class with `OneEuroFilter` and `EmaFilter` concrete implementations. Selected at startup via config, swappable at runtime for testing. Both operate on normalized plane (u,v) coordinates.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Cursor direction matches hand direction in all 4 cardinal directions (100% accuracy in synthetic test)
- **SC-002**: Head-movement invariance: cursor drift <2 pixels (RMS, per-axis) when head moves ±30° yaw, ±20° pitch with hand fixed relative to head
- **SC-003**: End-to-end latency <50ms at 30 FPS, measured from camera frame capture timestamp to uinput write completion timestamp (full pipeline: frame acquisition → landmark inference → projection → smoothing → velocity limiting → cursor injection)
- **SC-004**: Jitter reduction: cursor position std dev <1 pixel with stationary hand (synthetic noise test)
- **SC-005**: Velocity limiting: max cursor speed respects configured max_velocity ±5%
- **SC-006**: No regression in legacy mode: existing camera-coordinate mapping works identically
- **SC-007**: All existing tests pass (test_projection.py, test_cursor_smoothing.py, test_virtual_plane.py)

## Assumptions

- MediaPipe provides 21 hand landmarks + 468 face landmarks at 30 FPS minimum
- Camera provides 640x480 or 1280x720 frames (configurable)
- User's head is roughly facing the camera (frontal face detection works)
- Virtual plane distance default 0.30m, width 0.40m, height 0.25m (configurable via profiles)
- Linux uinput for cursor injection works (tested in current codebase)
- Existing config system (profiles.py, schema.py) supports new parameters
- PySide6 GUI exists for testing but not required for core functionality