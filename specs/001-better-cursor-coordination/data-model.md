# Data Model: Better Cursor Coordination

**Feature**: 001-better-cursor-coordination
**Date**: 2026-10-03

## Entities

### TrackedHand
Represents a detected hand with landmarks and derived tracking state.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `handedness` | `Literal["Left", "Right"]` | MediaPipe handedness classification | Required |
| `confidence` | `float` | Detection confidence [0,1] | ≥ `min_hand_confidence` |
| `landmarks` | `List[Landmark3D]` | 21 hand landmarks in camera coordinates | Exactly 21, each in [0,1]³ |
| `index_tip_3d` | `Vector3D` | Landmark 8 (index fingertip) in camera 3D | Derived from landmarks[8] |
| `palm_center_3d` | `Vector3D` | Palm center (landmark 0) in camera 3D | Derived from landmarks[0] |
| `plane_position` | `Vector2D` | Projected (u,v) on virtual plane [0,1]² | Valid if projection succeeded |
| `plane_position_smoothed` | `Vector2D` | After One Euro Filter on plane coords | In [0,1]² |
| `screen_position` | `Vector2D` | Final cursor position in screen pixels | Clamped to screen bounds |
| `velocity` | `Vector2D` | Screen-space velocity (px/s) | Magnitude ≤ `max_velocity` |
| `reference_point` | `Vector2D` | Reference for relative movement | Updated per mode |
| `is_primary` | `bool` | Primary (controlling) hand | True for first/preferred hand |
| `tracking_state` | `TrackingState` | Current tracking state | Enum |
| `lost_frames` | `int` | Consecutive frames without detection | Reset on detection |

### Landmark3D
MediaPipe landmark with 3D coordinates.

| Field | Type | Description |
|-------|------|-------------|
| `x` | `float` | Normalized camera X [0,1] |
| `y` | `float` | Normalized camera Y [0,1] |
| `z` | `float` | Normalized camera Z (depth, negative=forward) |
| `visibility` | `float` | Visibility confidence [0,1] |

### Vector3D / Vector2D
Simple coordinate vectors.

```python
@dataclass
class Vector3D:
    x: float
    y: float
    z: float

@dataclass
class Vector2D:
    x: float
    y: float
```

### VirtualPlane
Defines the projection plane in head coordinate space.

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `distance` | `float` | Distance from head origin along -Z (meters) | 0.30 |
| `width` | `float` | Plane width (meters) | 0.40 |
| `height` | `float` | Plane height (meters) | 0.25 |
| `center` | `Vector3D` | Plane center in head coords | (0, 0, -distance) |
| `normal` | `Vector3D` | Plane normal (toward user) | (0, 0, 1) |
| `u_axis` | `Vector3D` | Plane U axis (right) | (1, 0, 0) |
| `v_axis` | `Vector3D` | Plane V axis (up) | (0, 1, 0) |

### HeadPose
Head position and orientation from face landmarks.

| Field | Type | Description |
|-------|------|-------------|
| `eye_midpoint` | `Vector3D` | Midpoint between eyes in camera 3D |
| `forward_vector` | `Vector3D` | Head forward direction (nose to eye midpoint) |
| `up_vector` | `Vector3D` | Head up direction |
| `right_vector` | `Vector3D` | Head right direction (cross product) |
| `confidence` | `float` | Face detection confidence [0,1] |
| `landmarks` | `List[Landmark3D]` | 468 face landmarks (optional, for debug) |

### ProjectionResult
Output of virtual plane projection stage.

| Field | Type | Description |
|-------|------|-------------|
| `plane_u` | `float` | U coordinate on plane [0,1] |
| `plane_v` | `float` | V coordinate on plane [0,1] |
| `valid` | `bool` | Projection succeeded (ray intersects plane) |
| `intersection_point` | `Vector3D` | 3D intersection point in head coords |
| `ray_origin` | `Vector3D` | Eye midpoint in head coords |
| `ray_direction` | `Vector3D` | Normalized ray direction |

### SmoothingState
Internal state for One Euro Filter per axis.

| Field | Type | Description |
|-------|------|-------------|
| `x_filter` | `OneEuroFilter` | Filter for U coordinate |
| `y_filter` | `OneEuroFilter` | Filter for V coordinate |
| `last_value` | `Vector2D` | Last filtered output |
| `last_timestamp` | `float` | Last update time |

### VelocityLimiterState
State for corrected velocity limiter.

| Field | Type | Description |
|-------|------|-------------|
| `last_position` | `Vector2D` | Last cursor position (screen pixels) |
| `last_timestamp` | `float` | Last update time |
| `max_velocity` | `float` | Configured max velocity (px/s) |

### TrackingConfig (Extended)
Additional config fields for this feature.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_head_relative` | `bool` | `True` | Enable head-relative projection |
| `virtual_plane_distance` | `float` | `0.30` | Plane distance from head (m) |
| `virtual_plane_width` | `float` | `0.40` | Plane width (m) |
| `virtual_plane_height` | `float` | `0.25` | Plane height (m) |
| `head_confidence_threshold` | `float` | `0.5` | Min face confidence for head-relative |
| `projection_smoothing` | `SmoothingAlgorithm` | `ONE_EURO` | Smoothing on plane coords |
| `one_euro_min_cutoff` | `float` | `1.0` | One Euro min cutoff (Hz) |
| `one_euro_beta` | `float` | `0.0` | One Euro beta |
| `one_euro_d_cutoff` | `float` | `1.0` | One Euro derivative cutoff (Hz) |
| `reference_point_update_mode` | `Literal["every_frame", "dead_zone_exit"]` | `"every_frame"` | Reference update strategy |

### CursorConfig (Extended)
Existing config with clarified semantics.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_velocity` | `int` | `2000` | Max cursor speed normal mode (px/s) |
| `max_velocity_precision` | `int` | `500` | Max cursor speed precision mode (px/s) |
| `dead_zone_radius` | `float` | `0.02` | Dead zone in plane coords [0,1] |
| `stabilization_frames` | `int` | `10` | Frames to stabilize after tracking loss |

## Relationships

```
TrackingProcessor
    ├── tracks: List[TrackedHand] (0-2)
    ├── head_pose: Optional[HeadPose]
    ├── virtual_plane: VirtualPlane
    ├── smoothing_state: Dict[hand_id, SmoothingState]
    ├── velocity_limiter: VelocityLimiterState
    └── config: TrackingConfig

TrackedHand
    ├── landmarks: List[Landmark3D] (21)
    ├── plane_position: Vector2D (from ProjectionResult)
    ├── plane_position_smoothed: Vector2D (from SmoothingState)
    ├── screen_position: Vector2D (final output)
    └── reference_point: Vector2D (for relative movement)

ProjectionResult ← VirtualPlane.project(eye_midpoint, fingertip_3d)
SmoothingState.filter(ProjectionResult.plane_u, plane_v) → plane_position_smoothed
VelocityLimiter.limit(screen_position, dt) → limited_delta
screen_position + limited_delta → uinput
```

## State Transitions

### TrackedHand.tracking_state
```
NO_HAND → TRACKING_ONE_HAND (hand detected)
TRACKING_ONE_HAND → TRACKING_TWO_HANDS (second hand detected)
TRACKING_* → LOST_TRACK (lost_frames > max_lost_frames)
LOST_TRACK → NO_HAND (reset)
TRACKING_* → FROZEN (fist gesture)
FROZEN → TRACKING_* (open palm or timeout)
TRACKING_* → PRECISION_MODE (two-hand gesture or hotkey)
PRECISION_MODE → TRACKING_* (exit gesture or hotkey)
```

### Projection Validity
```
VALID → INVALID (face confidence < threshold OR ray parallel to plane)
INVALID → VALID (face confidence recovered AND ray intersects)
```

## Validation Rules

1. **Coordinate Contract**: Every pipeline stage validates input/output ranges
   - Camera coords: [0,1] × [0,1] × ℝ
   - Plane coords: [0,1] × [0,1]
   - Screen coords: [0, screen_width] × [0, screen_height]

2. **Projection Validity**: 
   - Ray must intersect plane in front of head (t > 0)
   - Intersection must be within plane bounds
   - Face confidence ≥ threshold

3. **Smoothing Output**: 
   - Smoothed coords remain in [0,1]²
   - Filter state persists across frames

4. **Velocity Limiting**:
   - |velocity| ≤ max_velocity
   - Position delta scaled proportionally if exceeded

5. **Reference Point**:
   - Head-relative: updated every frame to plane_position_smoothed
   - Legacy: updated only when leaving dead zone