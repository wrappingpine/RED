# Pipeline Stage Contracts

**Feature**: 001-better-cursor-coordination
**Version**: 1.0

## Stage 1: Camera Landmark Input → Camera Coordinates

**Input**: Raw MediaPipe hand landmarks (21 landmarks, each with x,y,z,visibility)

**Output**: `CameraLandmarks`
```python
@dataclass
class CameraLandmarks:
    landmarks: List[Landmark3D]  # 21 landmarks, each x,y,z in [0,1], visibility in [0,1]
    handedness: Literal["Left", "Right"]
    confidence: float  # [0,1]
    timestamp: float  # monotonic seconds
```

**Contract**:
- `len(landmarks) == 21`
- `all(0.0 <= lm.x <= 1.0 for lm in landmarks)`
- `all(0.0 <= lm.y <= 1.0 for lm in landmarks)`
- `all(lm.visibility >= 0.0 for lm in landmarks)`
- `0.0 <= confidence <= 1.0`
- `timestamp > 0`

**Error Handling**: If contract violated, log error, return `None`, caller handles gracefully

---

## Stage 2: Camera Coordinates → Virtual Plane Projection

**Input**: 
- `CameraLandmarks` (from Stage 1)
- `HeadPose` (from face tracker)
- `VirtualPlane` (configuration)

**Output**: `ProjectionResult`
```python
@dataclass
class ProjectionResult:
    plane_u: float  # [0,1]
    plane_v: float  # [0,1]
    valid: bool
    intersection_point: Vector3D  # in head coordinates
    ray_origin: Vector3D  # eye midpoint in head coordinates
    ray_direction: Vector3D  # normalized
```

**Contract**:
- If `valid == True`:
  - `0.0 <= plane_u <= 1.0`
  - `0.0 <= plane_v <= 1.0`
  - `intersection_point.z < 0` (in front of head)
  - `ray_direction` is normalized (|ray_direction| ≈ 1.0)
- If `valid == False`:
  - `plane_u, plane_v` undefined (caller must check `valid`)
  - Reason logged: "no_face", "low_confidence", "ray_parallel", "out_of_bounds"

**Algorithm**:
1. Transform eye midpoint and fingertip to head coordinate space using `HeadPose`
2. Compute ray: origin = eye_midpoint, direction = normalize(fingertip - eye_midpoint)
3. Intersect ray with plane: t = -ray_origin.z / ray_direction.z
4. If t <= 0: invalid (ray points away from plane)
5. intersection = ray_origin + t * ray_direction
6. plane_u = (intersection.x - plane_left) / plane_width
7. plane_v = (intersection.y - plane_bottom) / plane_height
8. Clamp to [0,1], valid = (0 <= u <= 1 and 0 <= v <= 1)

**Error Handling**: On any math error (division by zero, NaN), return `valid=False`

---

## Stage 3: Plane Coordinates → Smoothed Plane Coordinates

**Input**: 
- `ProjectionResult` (from Stage 2)
- `SmoothingState` (persistent per-hand)
- `SmoothingConfig` (algorithm, parameters)

**Output**: `SmoothedPlanePosition`
```python
@dataclass
class SmoothedPlanePosition:
    u: float  # [0,1]
    v: float  # [0,1]
    valid: bool  # same as input.valid
```

**Contract**:
- If `input.valid == False`: output `valid=False`, `u,v` undefined
- If `input.valid == True`:
  - `0.0 <= u <= 1.0`
  - `0.0 <= v <= 1.0`
  - Output varies smoothly from previous frame
  - One Euro Filter: adaptive cutoff based on velocity estimate

**Algorithms**:
- `ONE_EURO`: One Euro Filter with `min_cutoff`, `beta`, `d_cutoff`
- `EMA`: Exponential Moving Average with `alpha`
- `NONE`: Pass-through

**Error Handling**: Filter internal errors → fall back to pass-through, log warning

---

## Stage 4: Smoothed Plane Coordinates → Screen Coordinates

**Input**:
- `SmoothedPlanePosition` (from Stage 3)
- `CursorConfig` (screen dimensions, sensitivity, acceleration, dead zone, invert flags)

**Output**: `ScreenPosition`
```python
@dataclass
class ScreenPosition:
    x: float  # pixels, [0, screen_width]
    y: float  # pixels, [0, screen_height]
    valid: bool
```

**Contract**:
- If `input.valid == False`: output `valid=False`
- If `input.valid == True`:
  - Apply dead zone: if distance from reference < dead_zone_radius → no movement
  - Apply sensitivity: delta = (plane_pos - reference) * sensitivity * screen_size
  - Apply acceleration: delta *= |delta|^(acceleration - 1)
  - Apply inversion: if invert_x: delta.x *= -1; if invert_y: delta.y *= -1
  - New position = current_position + delta
  - Clamp to [0, screen_width] × [0, screen_height]
  - Update reference point per mode (head-relative: every frame; legacy: dead zone exit)

**Error Handling**: Clamping guarantees valid output if input valid

---

## Stage 5: Screen Coordinates → Velocity Limited Deltas

**Input**:
- `ScreenPosition` (from Stage 4)
- `VelocityLimiterState` (persistent)
- `max_velocity` (px/s from config)

**Output**: `VelocityLimitedDelta`
```python
@dataclass
class VelocityLimitedDelta:
    dx: int  # pixels
    dy: int  # pixels
    limited: bool  # True if velocity was capped
```

**Contract**:
- Computes velocity: `v = sqrt(dx^2 + dy^2) / dt`
- If `v <= max_velocity`: `dx, dy` unchanged, `limited=False`
- If `v > max_velocity`: scale `dx, dy` by `max_velocity / v`, `limited=True`
- `dt` = current_timestamp - last_timestamp (must be > 0)
- Updates internal state: `last_position = current_position`, `last_timestamp = now`

**Error Handling**: 
- `dt <= 0`: return `(0, 0, False)`, log warning
- NaN position: return `(0, 0, False)`, reset state

---

## Stage 6: Velocity Limited Deltas → uinput Events

**Input**: `VelocityLimitedDelta` (from Stage 5)

**Output**: Linux uinput REL_X, REL_Y events + SYN_REPORT

**Contract**:
- Write `REL_X = dx`, `REL_Y = dy`, `SYN_REPORT` atomically
- Non-blocking write
- On `EAGAIN`/`EWOULDBLOCK`: drop frame, log warning
- On other error: log error, attempt reconnect

---

## Full Pipeline Contract

**Input**: Camera frame + MediaPipe results

**Output**: uinput events (cursor movement, clicks, scroll)

**Latency Budget**:
- Stage 1 (MediaPipe): ~15-25ms
- Stage 2 (Projection): <1ms
- Stage 3 (Smoothing): <1ms
- Stage 4 (Screen mapping): <1ms
- Stage 5 (Velocity limit): <1ms
- Stage 6 (uinput): <1ms
- **Total**: <30ms processing + camera latency

**Monitoring**: Each stage logs `stage_name_ms` for diagnostics