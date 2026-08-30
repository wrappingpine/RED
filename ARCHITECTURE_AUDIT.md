# AirMouse Architecture Audit & Fix Plan

## Root Cause: "Cursor driven by raw hand movement instead of head-relative virtual-display/ray-projection system"

After reading all 11 core files and 3 test files, I've identified **9 critical architectural bugs** that cause the head-relative system to be bypassed or malfunction.

---

## Bug 1: Projection BEFORE Smoothing (tracking_processor.py:356-396)

**Location:** `TrackedHand.update()` lines 356-396

**Problem:** One Euro Filter and Velocity Limiter are applied to **raw MediaPipe landmarks** BEFORE projection. The ray-plane intersection happens AFTER smoothing in the pipeline.

**Current Flow (WRONG):**
```
MediaPipe Landmarks → OneEuro/VelocityLimit (in camera coords) → Projection → Cursor
```

**Correct Flow:**
```
MediaPipe Landmarks → Projection (ray-plane in head coords) → OneEuro/VelocityLimit (on normalized u,v) → Cursor
```

**Why it breaks head-relative:** Smoothing in camera coordinates destroys the geometric relationship between eye midpoint and fingertip that the ray projection depends on.

**Fix:** Move smoothing to AFTER projection - smooth the normalized (u,v) coordinates on the virtual plane.

---

## Bug 2: Reference Point Never Updated for Head-Relative Mode (tracking_processor.py:891-917)

**Location:** `TrackingProcessor.get_cursor_movement()` lines 891-917

**Problem:** The `_reference_point` is initialized once but **only updated when leaving dead zone** (line 917). In head-relative mode, the reference should be the **current projection position** every frame for relative movement.

**Current Code:**
```python
# Line 917 - ONLY updates when NOT in dead zone
self._reference_point = current_pos
```

**Fix:** Update reference point every frame in head-relative mode:
```python
if self.config.use_head_relative:
    self._reference_point = current_pos  # Always update for head-relative
else:
    # Legacy behavior - update when leaving dead zone
    if not self._dead_zone_active:
        self._reference_point = current_pos
```

---

## Bug 3: VelocityLimiter Data Contract Mismatch (tracking_processor.py:178-223, 912-913)

**Location:** `VelocityLimiter.limit()` vs `TrackingProcessor.get_cursor_movement()`

**Problem:** `VelocityLimiter.limit(value, t)` expects a **position** but receives a **delta (dx, dy)** at lines 912-913:
```python
limited_dx = self._cursor_vel_limiter.limit(dx, current_time)  # dx is a delta!
limited_dy = self._cursor_vel_limiter_y.limit(dy, current_time)
```

The VelocityLimiter computes `desired_velocity = (value - self.prev_value) / dt` treating `value` as position, but `dx` is already a velocity/delta.

**Fix:** Create a `VelocityLimiterDelta` class or fix the call to pass positions, not deltas.

---

## Bug 4: Two-Hand Tracking Enabled in Config But Not Runtime (tracking_processor.py:99, 680)

**Location:** `TrackingConfig.enable_two_hand = True` (line 99) but runtime check at line 680

**Problem:** Config default is `enable_two_hand = True` but the code at line 680 only processes secondary hand if `self.config.enable_two_hand` - however the CLI/main loop may not pass this through properly. Need to verify main_loop.py passes config correctly.

---

## Bug 5: Pinch Hysteresis Missing Confirm/Release Thresholds (gestures.py)

**Location:** `GestureRecognizer` class - needs three-threshold hysteresis

**Spec Required:**
- Pinch ENTER: 0.045 (start tracking pinch)
- Pinch CONFIRM: 0.040 (confirm pinch/click)  
- Pinch RELEASE: 0.070 (release click - wider for hysteresis)

**Current:** Only single threshold likely implemented.

---

## Bug 6: Missing Head-Movement Invariance Tests (tests/)

**Gap:** No tests verifying that when head moves but hand stays fixed relative to head, cursor doesn't move.

**Required Tests:**
1. `test_head_movement_invariance()` - Rotate/translate head, verify cursor stable
2. `test_projection_only_mode()` - Bypass camera, feed synthetic landmarks, verify u,v
3. `test_two_hand_precision_mode()` - Secondary hand enables precision mode

---

## Bug 7: CursorController Mixes Absolute and Relative APIs (cursor.py:258-352)

**Location:** `CursorController.map_hand_to_cursor()` (absolute) vs `get_relative_movement()` (relative)

**Problem:** 
- `map_hand_to_cursor()` returns **absolute screen coordinates** (pixels)
- `get_relative_movement()` computes delta from previous absolute position
- But TrackingProcessor outputs **relative normalized movement** (dx, dy)

**Data Flow Mismatch:**
```
TrackingProcessor.get_cursor_movement() → (dx, dy) normalized
    ↓
CursorController expects absolute hand landmarks
    ↓
VirtualMouse expects relative (dx, dy) for uinput
```

**Fix:** Choose ONE architecture:
- Option A: Pure relative - TrackingProcessor → CursorController.get_relative_movement_from_plane() → VirtualMouse
- Option B: Pure absolute - TrackingProcessor → CursorController.map_hand_to_cursor() → VirtualMouse (with absolute moves)

**Recommendation:** Option A (relative) matches uinput and avoids screen coordinate dependency.

---

## Bug 8: Main Loop Pipeline Order (main_loop.py)

**Need to verify:** The main loop calls components in correct order:
1. Camera frame
2. HandTracker.process()
3. FaceTracker.process() 
4. TrackingProcessor.process(hands, faces)
5. GestureRecognizer.process(tracking_result)
6. CursorController.get_relative_movement_from_plane() ← NEW method needed
7. VirtualMouse.move(dx, dy)

---

## Bug 9: No Debug Overlay for Projection Verification

**Spec Required:** Press 'd' to toggle debug overlay showing:
- Virtual plane corners projected to camera image
- Ray from eye to fingertip
- Intersection point on plane
- Normalized (u,v) coordinates
- Head coordinate axes

---

## Fix Implementation Order

### Phase 1: Core Pipeline Fixes (Critical)
1. **Fix Bug 1** - Move smoothing AFTER projection in TrackingProcessor
2. **Fix Bug 2** - Update reference point every frame for head-relative
3. **Fix Bug 3** - Fix VelocityLimiter contract for deltas
4. **Fix Bug 7** - Align CursorController to pure relative architecture

### Phase 2: Gesture & Two-Hand Fixes
5. **Fix Bug 5** - Implement three-threshold pinch hysteresis
6. **Fix Bug 4** - Verify/enable two-hand tracking at runtime

### Phase 3: Testing & Debug
7. **Fix Bug 6** - Add head-movement invariance tests + projection-only test mode
8. **Fix Bug 8** - Verify main_loop.py pipeline order
9. **Fix Bug 9** - Add debug overlay ('d' key)

---

## Files to Modify (Priority Order)

| Priority | File | Changes |
|----------|------|---------|
| 1 | `tracking_processor.py` | Move smoothing after projection, fix reference point, fix velocity limiter |
| 2 | `cursor.py` | Add `get_relative_movement_from_plane()`, remove absolute/relative confusion |
| 3 | `gestures.py` | Implement 3-threshold pinch hysteresis |
| 4 | `main_loop.py` | Verify pipeline order, connect CursorController relative API |
| 5 | `tests/test_projection.py` | Add head-movement invariance & projection-only tests |
| 6 | `vision/hand_tracker.py` | Add debug overlay support |
| 7 | `vision/face_tracker.py` | Add debug overlay support |

---

## Verification Checklist

After fixes, verify:
- [ ] Cursor moves only when hand moves RELATIVE TO HEAD (not camera)
- [ ] Head rotation/translation doesn't move cursor if hand fixed to head
- [ ] Smooth cursor movement (One Euro on u,v plane coords)
- [ ] Dead zone works on plane coordinates
- [ ] Pinch click works with hysteresis (enter 0.045, confirm 0.040, release 0.070)
- [ ] Two-hand mode: secondary hand enables precision mode
- [ ] Fist hold 0.5s pauses tracking
- [ ] All existing tests pass + new tests pass
- [ ] Debug overlay shows plane, ray, intersection correctly