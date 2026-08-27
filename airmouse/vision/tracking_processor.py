"""
Tracking Processor Module for Air Mouse

Implements a robust tracking pipeline for head-anchored virtual display:

1. Confidence filtering - reject low-confidence detections
2. Hand selection - pick the best hand (highest confidence, preferred handedness)
3. Landmark validation - check for physically plausible landmarks
4. Outlier rejection - detect and reject sudden unrealistic movements
5. Adaptive smoothing - One Euro Filter for stable yet responsive tracking
6. Dead zone - ignore tiny movements when hand is stationary
7. Velocity limiting - cap maximum cursor speed
8. Head-relative projection - ray from eye midpoint through fingertip to virtual plane

Supports both legacy 2D mapping and new 3D head-relative mapping.
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np

from .hand_tracker import Hand, Landmark, HandLandmark
from .face_tracker import Face, FaceLandmark
from .head_coords import HeadCoordinateSystem
from .virtual_plane import VirtualDisplayPlane
from .projection import HandProjector, ProjectionResult, create_projector
from .gestures import TrackingState

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """Result of tracking processor containing all tracking state information."""
    tracked_hands: List[Hand] = field(default_factory=list)
    primary_hand: Optional[Hand] = None
    secondary_hand: Optional[Hand] = None
    tracking_state: TrackingState = TrackingState.NO_HAND
    projection: Optional[ProjectionResult] = None
    secondary_projection: Optional[ProjectionResult] = None
    timestamp: float = 0.0

    def get_cursor_movement(self) -> Tuple[float, float]:
        """Get cursor movement from primary projection."""
        if self.projection and self.projection.valid:
            return (self.projection.u, self.projection.v)
        return (0.0, 0.0)


@dataclass
class TrackingConfig:
    """Configuration for the tracking processor."""
    # Confidence thresholds
    min_hand_confidence: float = 0.65
    min_landmark_visibility: float = 0.5
    min_face_confidence: float = 0.5

    # Hand selection
    preferred_handedness: str = "Right"  # "Right", "Left", or "Any"

    # Landmark validation - more lenient for wrist which moves more
    max_landmark_jump: float = 0.15  # Max normalized distance a landmark can move per frame (for fingertips)
    max_wrist_jump: float = 0.5      # Wrist can move more (normalized)
    max_hand_center_jump: float = 0.25  # Max normalized distance hand center can move per frame

    # One Euro Filter parameters
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.007
    one_euro_d_cutoff: float = 1.0

    # Dead zone
    dead_zone_radius: float = 0.015  # Normalized radius

    # Velocity limiting
    max_velocity: float = 0.5  # Normalized units per frame
    velocity_smoothing: float = 0.3

    # Tracking recovery
    max_lost_frames: int = 15  # Frames to hold position before resetting
    reset_on_large_jump: bool = True
    stabilization_frames: int = 5  # Frames to skip jump check during stabilization

    # Coordinate mapping
    use_index_tip: bool = True  # True = index tip, False = palm center
    invert_x: bool = False
    invert_y: bool = False

    # Head-relative tracking (new 3D system)
    use_head_relative: bool = True  # Enable head-anchored virtual plane
    virtual_plane_distance: float = 0.30  # meters (30cm)
    virtual_plane_width: float = 0.40     # meters (40cm)
    virtual_plane_height: float = 0.25    # meters (25cm)
    use_head_coords_for_ray: bool = True  # Compute ray in head coordinates

    # Two-hand tracking
    enable_two_hand: bool = True      # Enable stable two-hand tracking by handedness
    track_primary_only: bool = False  # If True, only track primary hand for cursor


class OneEuroFilter:
    """
    One Euro Filter for adaptive smoothing.
    Based on: https://cristal.univ-lille.fr/~casiez/1euro/

    Adapts cutoff frequency based on signal velocity:
    - Slow movement → low cutoff → heavy smoothing
    - Fast movement → high cutoff → responsive tracking
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: Optional[float] = None
        self.t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        """Compute alpha for exponential smoothing."""
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: Optional[float] = None) -> float:
        """Filter a value with timestamp."""
        if t is None:
            t = time.time()

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0:
            dt = 1e-3

        # Estimate derivative
        dx = (x - self.x_prev) / dt

        # Filter derivative
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Compute adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter signal
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    def reset(self):
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def set_params(self, min_cutoff: float = None, beta: float = None, d_cutoff: float = None):
        """Update filter parameters."""
        if min_cutoff is not None:
            self.min_cutoff = min_cutoff
        if beta is not None:
            self.beta = beta
        if d_cutoff is not None:
            self.d_cutoff = d_cutoff


class VelocityLimiter:
    """Limits the rate of change to prevent sudden jumps."""

    def __init__(self, max_velocity: float = 0.5, smoothing: float = 0.3):
        self.max_velocity = max_velocity
        self.smoothing = smoothing
        self.prev_value: Optional[float] = None
        self.prev_time: Optional[float] = None

    def limit(self, value: float, t: Optional[float] = None) -> float:
        """Limit velocity and apply smoothing."""
        if t is None:
            t = time.time()

        if self.prev_value is None:
            self.prev_value = value
            self.prev_time = t
            return value

        dt = t - self.prev_time
        if dt <= 0:
            dt = 1e-3

        # Calculate desired velocity
        desired_velocity = (value - self.prev_value) / dt

        # Clamp velocity
        max_vel = self.max_velocity / dt  # Convert to per-frame
        if abs(desired_velocity) > max_vel:
            desired_velocity = max_vel if desired_velocity > 0 else -max_vel

        # Apply limited velocity
        limited_value = self.prev_value + desired_velocity * dt

        # Apply additional smoothing
        smoothed = self.smoothing * limited_value + (1 - self.smoothing) * self.prev_value

        self.prev_value = smoothed
        self.prev_time = t

        return smoothed

    def reset(self):
        self.prev_value = None
        self.prev_time = None


class LandmarkValidator:
    """Validates hand landmarks for physical plausibility."""

    # Expected bone lengths (normalized, approximate ratios)
    BONE_RATIOS = {
        # (start, end): (min_ratio, max_ratio) relative to hand size
        (HandLandmark.WRIST, HandLandmark.INDEX_MCP): (0.15, 0.25),
        (HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP): (0.1, 0.18),
        (HandLandmark.INDEX_PIP, HandLandmark.INDEX_DIP): (0.08, 0.15),
        (HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP): (0.06, 0.12),
        (HandLandmark.WRIST, HandLandmark.THUMB_CMC): (0.1, 0.2),
        (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP): (0.08, 0.15),
        (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP): (0.06, 0.12),
        (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP): (0.05, 0.1),
    }

    @staticmethod
    def estimate_hand_size(landmarks: List[Landmark]) -> float:
        """Estimate hand size from wrist to middle MCP."""
        if len(landmarks) <= HandLandmark.MIDDLE_MCP.value:
            return 0.2  # Default
        wrist = landmarks[HandLandmark.WRIST.value]
        middle_mcp = landmarks[HandLandmark.MIDDLE_MCP.value]
        return math.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2)

    @classmethod
    def validate_landmarks(cls, landmarks: List[Landmark], hand_size: float = None) -> Tuple[bool, List[str]]:
        """
        Validate landmark positions for physical plausibility.

        Returns:
            (is_valid, list_of_issues)
        """
        if len(landmarks) < 21:
            return False, ["Insufficient landmarks"]

        issues = []
        if hand_size is None:
            hand_size = cls.estimate_hand_size(landmarks)

        # Check bone lengths - more lenient ranges
        for (start, end), (min_ratio, max_ratio) in cls.BONE_RATIOS.items():
            if start.value >= len(landmarks) or end.value >= len(landmarks):
                continue
            lm_start = landmarks[start.value]
            lm_end = landmarks[end.value]
            dist = math.sqrt((lm_start.x - lm_end.x)**2 + (lm_start.y - lm_end.y)**2)
            min_dist = hand_size * min_ratio * 0.5  # More lenient
            max_dist = hand_size * max_ratio * 2.0  # More lenient

            if dist < min_dist or dist > max_dist:
                issues.append(f"Bone {start.name}-{end.name}: length {dist:.3f} outside [{min_dist:.3f}, {max_dist:.3f}]")

        # Check landmark ordering (fingers should be roughly ordered) - more lenient
        # Index tip should be above (smaller y than) PIP - only reject if very far off
        if (landmarks[HandLandmark.INDEX_TIP.value].y > landmarks[HandLandmark.INDEX_PIP.value].y + 0.1):
            issues.append("Index tip below PIP (possibly folded or invalid)")

        # Check all landmarks are in valid range
        for i, lm in enumerate(landmarks):
            if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1):
                issues.append(f"Landmark {i} out of bounds: ({lm.x:.3f}, {lm.y:.3f})")
            if lm.visibility < 0.05:  # More lenient
                issues.append(f"Landmark {i} very low visibility: {lm.visibility:.3f}")

        # Only reject if there are severe issues (out of bounds or very low visibility)
        # Bone ratio issues are just warnings
        severe_issues = [i for i in issues if "out of bounds" in i or "very low visibility" in i]
        return len(severe_issues) == 0, issues


@dataclass
class TrackedHand:
    """Hand with tracking state for temporal consistency."""
    hand: Hand
    smoothed_landmarks: List[Landmark] = field(default_factory=list)
    frame_count: int = 0
    lost_frames: int = 0
    last_position: Tuple[float, float] = (0.0, 0.0)
    last_time: float = 0.0

    # One Euro Filters for each landmark (only for key landmarks)
    _filters: Dict[int, OneEuroFilter] = field(default_factory=dict)
    _vel_limiters: Dict[int, VelocityLimiter] = field(default_factory=dict)

    # Pre-allocated arrays to avoid allocation in hot path
    _smoothed_landmarks_cache: List[Landmark] = field(default_factory=lambda: [Landmark(0.0, 0.0, 0.0) for _ in range(21)])
    _temp_landmark: Landmark = field(default_factory=lambda: Landmark(0.0, 0.0, 0.0))

    # Key landmark indices to smooth
    KEY_LANDMARKS = [
        HandLandmark.WRIST.value,
        HandLandmark.INDEX_MCP.value,
        HandLandmark.INDEX_TIP.value,
        HandLandmark.THUMB_TIP.value,
        HandLandmark.MIDDLE_TIP.value,
    ]

    def __post_init__(self):
        self.last_time = time.time()

    def _get_filter(self, idx: int) -> OneEuroFilter:
        """Get or create One Euro Filter for landmark."""
        if idx not in self._filters:
            self._filters[idx] = OneEuroFilter()
        return self._filters[idx]

    def _get_vel_limiter(self, idx: int) -> VelocityLimiter:
        """Get or create Velocity Limiter for landmark."""
        if idx not in self._vel_limiters:
            self._vel_limiters[idx] = VelocityLimiter()
        return self._vel_limiters[idx]

    def update(self, new_hand: Hand, config: TrackingConfig) -> bool:
        """
        Update tracked hand with new detection.

        Returns:
            True if update successful, False if detection rejected
        """
        current_time = time.time()

        # Validate new landmarks
        hand_size = LandmarkValidator.estimate_hand_size(new_hand.landmarks)
        is_valid, issues = LandmarkValidator.validate_landmarks(new_hand.landmarks, hand_size)

        if not is_valid:
            logger.debug(f"Landmark validation failed: {issues}")
            self.lost_frames += 1
            return False

        # Apply One Euro Filter to key landmarks FIRST (smooth before jump check)
        # Reuse pre-allocated cache to avoid allocations
        cache = self._smoothed_landmarks_cache
        new_landmarks = new_hand.landmarks

        for i in range(21):
            lm = new_landmarks[i]
            if i in self.KEY_LANDMARKS:
                # Apply velocity limiting first
                vel_limiter = self._get_vel_limiter(i)
                lx = vel_limiter.limit(lm.x, current_time)
                ly = vel_limiter.limit(lm.y, current_time)
                lz = vel_limiter.limit(lm.z, current_time)

                # Apply One Euro Filter
                filt_x = self._get_filter(i * 3)
                filt_y = self._get_filter(i * 3 + 1)
                filt_z = self._get_filter(i * 3 + 2)

                sx = filt_x.filter(lx, current_time)
                sy = filt_y.filter(ly, current_time)
                sz = filt_z.filter(lz, current_time)

                cache[i].x = sx
                cache[i].y = sy
                cache[i].z = sz
                cache[i].visibility = lm.visibility
            else:
                # For non-key landmarks, just copy with slight smoothing toward previous
                if self.smoothed_landmarks and i < len(self.smoothed_landmarks):
                    prev = self.smoothed_landmarks[i]
                    alpha = 0.3
                    cache[i].x = alpha * lm.x + (1 - alpha) * prev.x
                    cache[i].y = alpha * lm.y + (1 - alpha) * prev.y
                    cache[i].z = alpha * lm.z + (1 - alpha) * prev.z
                    cache[i].visibility = lm.visibility
                else:
                    cache[i].x = lm.x
                    cache[i].y = lm.y
                    cache[i].z = lm.z
                    cache[i].visibility = lm.visibility

        # Check for large jumps (outlier rejection) on SMOOTHED landmarks - skip during stabilization
        if self.smoothed_landmarks and config.max_landmark_jump > 0 and self.frame_count >= config.stabilization_frames:
            for idx in self.KEY_LANDMARKS:
                if idx < 21:
                    old_lm = self.smoothed_landmarks[idx]
                    new_lm = cache[idx]
                    jump = math.sqrt((new_lm.x - old_lm.x)**2 + (new_lm.y - old_lm.y)**2)

                    # Use different thresholds: wrist can move more than fingertips
                    threshold = config.max_wrist_jump if idx == 0 else config.max_landmark_jump
                    if jump > threshold:
                        logger.warning(f"Large jump detected at landmark {idx} ({HandLandmark(idx).name}): {jump:.3f} > {threshold} (frame={self.frame_count})")
                        if config.reset_on_large_jump:
                            self.lost_frames += 1
                            return False

        # Check hand center jump on SMOOTHED landmarks - skip during stabilization
        if self.smoothed_landmarks and config.max_hand_center_jump > 0 and self.frame_count >= config.stabilization_frames:
            old_center = self._get_palm_center(self.smoothed_landmarks)
            new_center = self._get_palm_center(cache)
            if old_center and new_center:
                jump = math.sqrt((new_center.x - old_center.x)**2 + (new_center.y - old_center.y)**2)
                if jump > config.max_hand_center_jump:
                    logger.warning(f"Large hand center jump: {jump:.3f} > {config.max_hand_center_jump} (frame={self.frame_count})")
                    if config.reset_on_large_jump:
                        self.lost_frames += 1
                        return False

        # All checks passed - commit smoothed landmarks (swap references)
        self.smoothed_landmarks, cache = cache, self.smoothed_landmarks
        self.hand = new_hand
        self.hand.landmarks = self.smoothed_landmarks  # Replace with smoothed landmarks
        # Recompute derived properties after landmark smoothing
        self.hand._compute_derived()
        self.frame_count += 1
        self.lost_frames = 0
        self.last_time = current_time

        # Update last position
        ref_point = self.hand.index_tip if config.use_index_tip else self.hand.palm_center
        if ref_point:
            self.last_position = (ref_point.x, ref_point.y)

        return True

    def _get_palm_center(self, landmarks: List[Landmark]) -> Optional[Landmark]:
        """Calculate palm center from landmarks."""
        if len(landmarks) < 21:
            return None
        # Use temp landmark to avoid allocation
        result = self._temp_landmark
        result.x = (landmarks[HandLandmark.INDEX_MCP.value].x +
                    landmarks[HandLandmark.MIDDLE_MCP.value].x +
                    landmarks[HandLandmark.RING_MCP.value].x +
                    landmarks[HandLandmark.PINKY_MCP.value].x) / 4
        result.y = (landmarks[HandLandmark.INDEX_MCP.value].y +
                    landmarks[HandLandmark.MIDDLE_MCP.value].y +
                    landmarks[HandLandmark.RING_MCP.value].y +
                    landmarks[HandLandmark.PINKY_MCP.value].y) / 4
        result.z = (landmarks[HandLandmark.INDEX_MCP.value].z +
                    landmarks[HandLandmark.MIDDLE_MCP.value].z +
                    landmarks[HandLandmark.RING_MCP.value].z +
                    landmarks[HandLandmark.PINKY_MCP.value].z) / 4
        return result

    def is_lost(self, config: TrackingConfig) -> bool:
        """Check if tracking is lost."""
        return self.lost_frames >= config.max_lost_frames

    def get_cursor_position(self, config: TrackingConfig) -> Optional[Tuple[float, float]]:
        """Get current cursor position (normalized)."""
        if self.is_lost(config):
            return None
        ref_point = self.hand.index_tip if config.use_index_tip else self.hand.palm_center
        if ref_point:
            x, y = ref_point.x, ref_point.y
            if config.invert_x:
                x = 1.0 - x
            if config.invert_y:
                y = 1.0 - y
            return (x, y)
        return None

    def reset(self):
        """Reset tracking state."""
        self.smoothed_landmarks = []
        self.frame_count = 0
        self.lost_frames = 0
        self._filters.clear()
        self._vel_limiters.clear()


class TrackingProcessor:
    """
    Main tracking processor that coordinates the full pipeline.

    Pipeline (Head-Relative Mode):
    Camera → MediaPipe Hand Landmarker + Face Landmarker
    → Confidence filtering (both hand and face)
    → Hand selection (single best hand)
    → Face selection (primary face)
    → Head coordinate system from face
    → Virtual display plane anchored to head
    → Ray from eye midpoint through fingertip
    → Ray-plane intersection
    → Normalized coordinates on plane
    → One Euro Filter smoothing
    → Dead zone
    → Velocity limiting
    → Cursor movement output

    Pipeline (Legacy 2D Mode):
    Camera → MediaPipe Hand Landmarker
    → Confidence filtering
    → Hand selection
    → Landmark validation
    → Outlier rejection
    → One Euro Filter smoothing
    → Dead zone
    → Velocity limiting
    → 2D coordinate mapping
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()
        # Backward compatibility property
        self._config = self.config
        # Two-hand tracking: primary (cursor control) and secondary (gestures/precision)
        self._primary_hand: Optional[TrackedHand] = None
        self._secondary_hand: Optional[TrackedHand] = None
        self._tracked_face: Optional[Face] = None
        self._head_coords: Optional[HeadCoordinateSystem] = None
        self._virtual_plane: Optional[VirtualDisplayPlane] = None
        self._projector: Optional[HandProjector] = None
        self._last_cursor_pos: Optional[Tuple[float, float]] = None
        self._cursor_vel_limiter = VelocityLimiter(
            max_velocity=self.config.max_velocity,
            smoothing=self.config.velocity_smoothing
        )
        self._cursor_vel_limiter_y = VelocityLimiter(
            max_velocity=self.config.max_velocity,
            smoothing=self.config.velocity_smoothing
        )
        self._dead_zone_active = False
        self._reference_point: Optional[Tuple[float, float]] = None
        self._last_projection: Optional[ProjectionResult] = None
        self._last_secondary_projection: Optional[ProjectionResult] = None

    @property
    def _plane(self):
        """Backward compatibility property for _virtual_plane."""
        return self._virtual_plane

    def process(self, hands: List[Hand], faces: List[Face] = None) -> Optional[TrackingResult]:
        """
        Process detected hands (and faces) through the full tracking pipeline.

        Two-hand tracking with stable identity by handedness:
        - Primary hand: controls cursor (preferred_handedness, default Right)
        - Secondary hand: enables precision mode / gestures

        Args:
            hands: List of hands from HandTracker
            faces: List of faces from FaceTracker (required for head-relative mode)

        Returns:
            TrackingResult with tracking state, hands, and projections, or None if no valid hand
        """
        current_time = time.time()

        if not hands:
            # No hands detected - increment lost frames for both hands
            self._update_lost_frames()
            return TrackingResult(
                tracked_hands=[],
                primary_hand=None,
                secondary_hand=None,
                tracking_state=TrackingState.NO_HAND,
                projection=None,
                secondary_projection=None,
                timestamp=current_time
            )

        # Select primary and secondary hands by handedness for stable identity
        primary_hand, secondary_hand = self._select_hands_by_handedness(hands)

        if primary_hand is None:
            self._update_lost_frames()
            return TrackingResult(
                tracked_hands=[],
                primary_hand=None,
                secondary_hand=None,
                tracking_state=TrackingState.NO_HAND,
                projection=None,
                secondary_projection=None,
                timestamp=current_time
            )

        # Check confidence threshold for primary hand
        if primary_hand.confidence < self.config.min_hand_confidence:
            logger.debug(f"Primary hand confidence too low: {primary_hand.confidence:.2f}")
            self._update_lost_frames()
            return TrackingResult(
                tracked_hands=[],
                primary_hand=None,
                secondary_hand=None,
                tracking_state=TrackingState.NO_HAND,
                projection=None,
                secondary_projection=None,
                timestamp=current_time
            )

        # For head-relative mode, we need a valid face
        if self.config.use_head_relative:
            if not faces:
                logger.debug("Head-relative mode requires face tracking but no faces provided")
                self._update_lost_frames()
                return TrackingResult(
                    tracked_hands=[],
                    primary_hand=None,
                    secondary_hand=None,
                    tracking_state=TrackingState.LOST_TRACK,
                    projection=None,
                    secondary_projection=None,
                    timestamp=current_time
                )

            # Select best face (first one with sufficient confidence)
            best_face = None
            for face in faces:
                if face.confidence >= self.config.min_face_confidence:
                    best_face = face
                    break

            if best_face is None:
                logger.debug("No face with sufficient confidence")
                self._update_lost_frames()
                return TrackingResult(
                    tracked_hands=[],
                    primary_hand=None,
                    secondary_hand=None,
                    tracking_state=TrackingState.LOST_TRACK,
                    projection=None,
                    secondary_projection=None,
                    timestamp=current_time
                )

            # Initialize or update head coordinate system and virtual plane
            self._update_head_tracking(best_face)

            if not self._head_coords or not self._head_coords.is_valid():
                logger.debug("Invalid head coordinate system")
                self._update_lost_frames()
                return TrackingResult(
                    tracked_hands=[],
                    primary_hand=None,
                    secondary_hand=None,
                    tracking_state=TrackingState.LOST_TRACK,
                    projection=None,
                    secondary_projection=None,
                    timestamp=current_time
                )

            # Project primary hand to virtual plane
            projection = self._projector.project(primary_hand, best_face) if self._projector else None

            if not projection or not projection.valid:
                logger.debug(f"Primary projection failed: {projection.error_message if projection else 'No projector'}")
                self._update_lost_frames()
                return TrackingResult(
                    tracked_hands=[],
                    primary_hand=None,
                    secondary_hand=None,
                    tracking_state=TrackingState.LOST_TRACK,
                    projection=None,
                    secondary_projection=None,
                    timestamp=current_time
                )

            self._last_projection = projection

            # Project secondary hand if present and two-hand mode enabled
            if secondary_hand and self.config.enable_two_hand:
                sec_projection = self._projector.project(secondary_hand, best_face)
                if sec_projection and sec_projection.valid:
                    self._last_secondary_projection = sec_projection
                else:
                    self._last_secondary_projection = None

        # Initialize or update primary tracked hand
        if self._primary_hand is None:
            self._primary_hand = TrackedHand(hand=primary_hand)
            self._primary_hand.smoothed_landmarks = primary_hand.landmarks.copy()
            ref_point = primary_hand.index_tip if self.config.use_index_tip else primary_hand.palm_center
            if ref_point:
                self._reference_point = (ref_point.x, ref_point.y)
                self._last_cursor_pos = self._reference_point
            logger.info(f"Primary tracking started: {primary_hand.handedness} hand (confidence: {primary_hand.confidence:.2f})")
        else:
            success = self._primary_hand.update(primary_hand, self.config)
            if not success:
                pass

        # Initialize or update secondary tracked hand
        if self.config.enable_two_hand and secondary_hand:
            if self._secondary_hand is None:
                self._secondary_hand = TrackedHand(hand=secondary_hand)
                self._secondary_hand.smoothed_landmarks = secondary_hand.landmarks.copy()
                logger.info(f"Secondary tracking started: {secondary_hand.handedness} hand (confidence: {secondary_hand.confidence:.2f})")
            else:
                self._secondary_hand.update(secondary_hand, self.config)
        else:
            # No secondary hand or two-hand disabled
            if self._secondary_hand and self._secondary_hand.is_lost(self.config):
                self._secondary_hand = None

        # Check if primary tracking is lost
        if self._primary_hand and self._primary_hand.is_lost(self.config):
            logger.info("Primary tracking lost - resetting")
            self._primary_hand = None
            self._secondary_hand = None
            self._reference_point = None
            self._last_cursor_pos = None
            self._cursor_vel_limiter.reset()
            self._cursor_vel_limiter_y.reset()
            return TrackingResult(
                tracked_hands=[],
                primary_hand=None,
                secondary_hand=None,
                tracking_state=TrackingState.LOST_TRACK,
                projection=None,
                secondary_projection=None,
                timestamp=current_time
            )

        # Determine tracking state
        tracking_state = TrackingState.TRACKING_ONE_HAND
        if self._secondary_hand and not self._secondary_hand.is_lost(self.config):
            tracking_state = TrackingState.TRACKING_TWO_HANDS

        # Build list of tracked hands
        tracked_hands_list = []
        if self._primary_hand:
            tracked_hands_list.append(self._primary_hand.hand)
        if self._secondary_hand and not self._secondary_hand.is_lost(self.config):
            tracked_hands_list.append(self._secondary_hand.hand)

        return TrackingResult(
            tracked_hands=tracked_hands_list,
            primary_hand=self._primary_hand.hand if self._primary_hand else None,
            secondary_hand=self._secondary_hand.hand if self._secondary_hand and not self._secondary_hand.is_lost(self.config) else None,
            tracking_state=tracking_state,
            projection=self._last_projection,
            secondary_projection=self._last_secondary_projection,
            timestamp=current_time
        )

    def _update_lost_frames(self):
        """Increment lost frames for both hands and reset if needed."""
        if self._primary_hand:
            self._primary_hand.lost_frames += 1
        if self._secondary_hand:
            self._secondary_hand.lost_frames += 1

    def _select_hands_by_handedness(self, hands: List[Hand]) -> Tuple[Optional[Hand], Optional[Hand]]:
        """
        Select primary and secondary hands by handedness for stable identity.

        Primary hand: preferred_handedness (default "Right"), highest confidence
        Secondary hand: opposite handedness, highest confidence

        Returns:
            Tuple of (primary_hand, secondary_hand) or (None, None)
        """
        if not hands:
            return None, None

        # Filter by minimum confidence
        valid_hands = [h for h in hands if h.confidence >= self.config.min_hand_confidence]
        if not valid_hands:
            return None, None

        # Group by handedness
        right_hands = [h for h in valid_hands if h.handedness == "Right"]
        left_hands = [h for h in valid_hands if h.handedness == "Left"]

        primary = None
        secondary = None

        # Select primary based on preferred handedness
        if self.config.preferred_handedness == "Right" and right_hands:
            primary = max(right_hands, key=lambda h: h.confidence)
            if self.config.enable_two_hand and left_hands:
                secondary = max(left_hands, key=lambda h: h.confidence)
        elif self.config.preferred_handedness == "Left" and left_hands:
            primary = max(left_hands, key=lambda h: h.confidence)
            if self.config.enable_two_hand and right_hands:
                secondary = max(right_hands, key=lambda h: h.confidence)
        else:
            # "Any" or preferred not available - pick highest confidence as primary
            primary = max(valid_hands, key=lambda h: h.confidence)
            if self.config.enable_two_hand:
                # Secondary is the other hand (different handedness) with highest confidence
                other_hands = [h for h in valid_hands if h.handedness != primary.handedness]
                if other_hands:
                    secondary = max(other_hands, key=lambda h: h.confidence)

        return primary, secondary

    def _update_head_tracking(self, face: Face):
        """Update head coordinate system, virtual plane, and projector from face."""
        # Create head coordinate system from face
        new_head_coords = HeadCoordinateSystem.from_face(face)

        # Check if head coords changed significantly (to avoid re-creating projector unnecessarily)
        if (self._head_coords is None or not self._head_coords.is_valid() or
                not new_head_coords.is_valid() or
                self._head_coords_changed(self._head_coords, new_head_coords)):
            self._head_coords = new_head_coords

            # Create virtual plane
            self._virtual_plane = VirtualDisplayPlane(
                distance=self.config.virtual_plane_distance,
                width=self.config.virtual_plane_width,
                height=self.config.virtual_plane_height,
                head_coords=self._head_coords
            )

            # Create projector
            self._projector = HandProjector(
                virtual_plane=self._virtual_plane,
                head_coords=self._head_coords,
                use_head_coords_for_ray=self.config.use_head_coords_for_ray
            )

            logger.debug("Updated head coordinate system and virtual plane")

    def _head_coords_changed(self, old: HeadCoordinateSystem, new: HeadCoordinateSystem, threshold: float = 0.02) -> bool:
        """Check if head coordinate system changed significantly."""
        # Compare origins
        origin_diff = np.linalg.norm(old.origin - new.origin)
        if origin_diff > threshold:
            return True

        # Compare forward vectors
        forward_diff = np.arccos(np.clip(np.dot(old.forward, new.forward), -1.0, 1.0))
        if forward_diff > 0.1:  # ~5.7 degrees
            return True

        return False

    def _select_best_hand(self, hands: List[Hand]) -> Optional[Hand]:
        """Select the best hand based on confidence and handedness preference."""
        if not hands:
            return None

        # Filter by minimum confidence
        valid_hands = [h for h in hands if h.confidence >= self.config.min_hand_confidence]
        if not valid_hands:
            return None

        # Prefer preferred handedness
        if self.config.preferred_handedness != "Any":
            preferred = [h for h in valid_hands if h.handedness == self.config.preferred_handedness]
            if preferred:
                return max(preferred, key=lambda h: h.confidence)

        # Return highest confidence hand
        return max(valid_hands, key=lambda h: h.confidence)

    def get_cursor_movement(self) -> Optional[Tuple[float, float]]:
        """
        Get smoothed cursor movement (dx, dy) in normalized coordinates.

        For head-relative mode, uses the projection result from the virtual plane.
        For legacy mode, uses the 2D hand landmark position.

        Returns:
            (dx, dy) normalized movement, or None if no valid tracking
        """
        if not self._primary_hand or self._primary_hand.is_lost(self.config):
            return None

        # For head-relative mode, use projection result
        if self.config.use_head_relative and self._last_projection and self._last_projection.valid:
            current_pos = (self._last_projection.u, self._last_projection.v)
        else:
            # Legacy 2D mode
            current_pos = self._primary_hand.get_cursor_position(self.config)
            if current_pos is None:
                return None

        # Initialize reference point
        if self._reference_point is None:
            self._reference_point = current_pos
            self._last_cursor_pos = current_pos
            return (0.0, 0.0)

        # Calculate raw movement relative to reference point
        dx = current_pos[0] - self._reference_point[0]
        dy = current_pos[1] - self._reference_point[1]

        # Apply dead zone
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < self.config.dead_zone_radius:
            # In dead zone - don't move cursor, but update reference point slightly
            # to prevent drift when leaving dead zone
            self._dead_zone_active = True
            return (0.0, 0.0)

        self._dead_zone_active = False

        # Apply velocity limiting to cursor movement - reuse pre-created limiters
        current_time = time.time()
        limited_dx = self._cursor_vel_limiter.limit(dx, current_time)
        limited_dy = self._cursor_vel_limiter_y.limit(dy, current_time)

        # Update reference point to current position (relative tracking)
        # This makes movement relative to hand's current position
        self._reference_point = current_pos

        return (limited_dx, limited_dy)

    def get_absolute_cursor_position(self, screen_width: int, screen_height: int) -> Optional[Tuple[int, int]]:
        """Get absolute cursor position in screen coordinates."""
        if not self._primary_hand or self._primary_hand.is_lost(self.config):
            return None

        # For head-relative mode, use projection result
        if self.config.use_head_relative and self._last_projection and self._last_projection.valid:
            pos = (self._last_projection.u, self._last_projection.v)
        else:
            # Legacy 2D mode
            pos = self._primary_hand.get_cursor_position(self.config)
            if pos is None:
                return None

        # Map to screen
        x = int(pos[0] * screen_width)
        y = int(pos[1] * screen_height)

        # Clamp
        x = max(0, min(screen_width - 1, x))
        y = max(0, min(screen_height - 1, y))

        return (x, y)

    def is_tracking(self) -> bool:
        """Check if currently tracking a hand."""
        return self._primary_hand is not None and not self._primary_hand.is_lost(self.config)

    def is_two_hand_tracking(self) -> bool:
        """Check if currently tracking two hands."""
        return (self._primary_hand is not None and not self._primary_hand.is_lost(self.config) and
                self._secondary_hand is not None and not self._secondary_hand.is_lost(self.config))

    def get_tracking_info(self) -> Dict:
        """Get debugging info about tracking state."""
        if not self._primary_hand:
            return {"tracking": False, "two_hand": False}

        info = {
            "tracking": True,
            "two_hand": self.is_two_hand_tracking(),
            "primary": {
                "handedness": self._primary_hand.hand.handedness,
                "confidence": self._primary_hand.hand.confidence,
                "frame_count": self._primary_hand.frame_count,
                "lost_frames": self._primary_hand.lost_frames,
            },
            "dead_zone_active": self._dead_zone_active,
            "reference_point": self._reference_point,
        }

        if self._secondary_hand and not self._secondary_hand.is_lost(self.config):
            info["secondary"] = {
                "handedness": self._secondary_hand.hand.handedness,
                "confidence": self._secondary_hand.hand.confidence,
                "frame_count": self._secondary_hand.frame_count,
                "lost_frames": self._secondary_hand.lost_frames,
            }

        return info

    def reset(self):
        """Reset all tracking state."""
        self._primary_hand = None
        self._secondary_hand = None
        self._last_cursor_pos = None
        self._reference_point = None
        self._cursor_vel_limiter.reset()
        if hasattr(self, '_cursor_vel_limiter_y'):
            self._cursor_vel_limiter_y.reset()
        self._dead_zone_active = False


if __name__ == "__main__":
    # Test the tracking processor
    import sys
    import cv2
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from camera.manager import CameraManager, CameraSettings
    from vision.hand_tracker import HandTracker, HandTrackerSettings

    logging.basicConfig(level=logging.DEBUG)

    print("Testing TrackingProcessor...")
    print("Press 'q' to quit")

    camera = CameraManager()
    if not camera.open_camera(CameraSettings(device_index=0, width=640, height=480)):
        print("Failed to open camera")
        sys.exit(1)

    tracker = HandTracker(HandTrackerSettings(max_hands=1))
    processor = TrackingProcessor(TrackingConfig(
        min_hand_confidence=0.65,
        preferred_handedness="Right",
        dead_zone_radius=0.015,
    ))

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                break

            hands = tracker.process(frame)
            tracked_hand = processor.process(hands)

            movement = processor.get_cursor_movement()
            info = processor.get_tracking_info()

            if tracked_hand:
                print(f"\rHand: {tracked_hand.handedness}, "
                      f"Conf: {tracked_hand.confidence:.2f}, "
                      f"Movement: {movement}, "
                      f"Frames: {info.get('frame_count', 0)}, "
                      f"Lost: {info.get('lost_frames', 0)}", end="")
                annotated = tracker.draw_landmarks(frame, [tracked_hand])
            else:
                print(f"\rNo hand tracked - {info}", end="")
                annotated = frame

            cv2.imshow("Tracking Processor Test", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.close_camera()
        tracker.close()
        cv2.destroyAllWindows()