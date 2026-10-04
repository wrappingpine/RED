"""
3D Projection Module for Air Mouse

Projects hand landmarks (index fingertip) through eye midpoint to virtual display plane.
Provides normalized coordinates for cursor control.
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from enum import Enum
from .hand_tracker import Hand, Landmark
from .face_tracker import Face, FaceLandmark
from .head_coords import HeadCoordinateSystem
from .virtual_plane import VirtualDisplayPlane

logger = logging.getLogger(__name__)


class ProjectionLogLevel(Enum):
    """Configurable log levels for projection diagnostics."""
    OFF = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4
    TRACE = 5


class ProjectionDiagnostics:
    """
    Structured diagnostics for projection pipeline.

    Collects structured events with configurable log levels so that
    production users can tune verbosity without changing code.
    """

    def __init__(self, log_level: ProjectionLogLevel = ProjectionLogLevel.WARN):
        self._log_level = log_level
        self._events: List[Dict] = []
        self._max_events = 1000
        self._last_sample: Dict[str, float] = {}
        self._sample_interval = 1.0  # seconds between samples of same event

    @property
    def log_level(self) -> ProjectionLogLevel:
        return self._log_level

    @log_level.setter
    def log_level(self, level: ProjectionLogLevel):
        self._log_level = level

    def _should_log(self, level: ProjectionLogLevel) -> bool:
        return level.value <= self._log_level.value

    def _sample_key(self, event_type: str) -> bool:
        """Rate-limit repeated events of the same type."""
        now = time.time()
        last = self._last_sample.get(event_type, 0.0)
        if now - last >= self._sample_interval:
            self._last_sample[event_type] = now
            return True
        return False

    def log_event(self, level: ProjectionLogLevel, event_type: str, message: str,
                  details: Optional[Dict] = None):
        """Log a structured event if the level is enabled."""
        if not self._should_log(level):
            return
        if not self._sample_key(event_type):
            return

        event = {
            "timestamp": time.time(),
            "level": level.name,
            "event": event_type,
            "message": message,
        }
        if details:
            event["details"] = details

        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events.pop(0)

        if level == ProjectionLogLevel.ERROR:
            logger.error(f"[PROJECTION] {event_type}: {message}")
        elif level == ProjectionLogLevel.WARN:
            logger.warning(f"[PROJECTION] {event_type}: {message}")
        elif level == ProjectionLogLevel.INFO:
            logger.info(f"[PROJECTION] {event_type}: {message}")
        elif level == ProjectionLogLevel.DEBUG:
            logger.debug(f"[PROJECTION] {event_type}: {message}")

    def get_events(self, limit: int = 100) -> List[Dict]:
        """Get recent diagnostic events."""
        return self._events[-limit:] if self._events else []

    def clear(self):
        """Clear stored events."""
        self._events.clear()
        self._last_sample.clear()

    def get_summary(self) -> Dict:
        """Get a summary of collected diagnostics."""
        event_counts: Dict[str, int] = {}
        for event in self._events:
            etype = event.get("event", "unknown")
            event_counts[etype] = event_counts.get(etype, 0) + 1
        return {
            "total_events": len(self._events),
            "event_counts": event_counts,
            "log_level": self._log_level.name,
        }


@dataclass
class ProjectionResult:
    """Result of hand-to-plane projection."""
    # Normalized coordinates on plane [0, 1] x [0, 1]
    u: float = 0.5
    v: float = 0.5

    # 3D intersection point in camera coordinates
    intersection_camera: Optional[np.ndarray] = None

    # 3D intersection point in head coordinates
    intersection_head: Optional[np.ndarray] = None

    # Ray information
    ray_origin_camera: Optional[np.ndarray] = None  # Eye midpoint
    ray_direction_camera: Optional[np.ndarray] = None

    # Validity
    valid: bool = False
    error_message: str = ""

    # Confidence at time of projection (for gating)
    hand_confidence: float = 0.0
    face_confidence: float = 0.0

    def get_normalized(self) -> Tuple[float, float]:
        """Get normalized (u, v) coordinates."""
        return (self.u, self.v)

    def is_valid(self) -> bool:
        """Check if projection is valid."""
        return self.valid


class HandProjector:
    """
    Projects hand landmarks to virtual display plane via ray-plane intersection.

    Pipeline:
    1. Get index fingertip 3D position from MediaPipe hand landmarks
    2. Get eye midpoint from face landmarks
    3. Create ray from eye midpoint through fingertip
    4. Intersect ray with virtual display plane
    5. Convert intersection to normalized [0, 1] coordinates

    Confidence gating:
    - Hand and face confidence thresholds are configurable.
    - Low-confidence projections are rejected before ray computation
      to prevent jitter from noisy landmark estimates.
    """

    def __init__(
        self,
        virtual_plane: VirtualDisplayPlane,
        head_coords: HeadCoordinateSystem,
        use_head_coords_for_ray: bool = True,
        min_hand_confidence: float = 0.5,
        min_face_confidence: float = 0.5,
        diagnostics: Optional[ProjectionDiagnostics] = None,
        log_level: ProjectionLogLevel = ProjectionLogLevel.WARN,
    ):
        """
        Initialize HandProjector.

        Args:
            virtual_plane: Virtual display plane
            head_coords: Head coordinate system
            use_head_coords_for_ray: If True, compute ray in head coordinates for accuracy
            min_hand_confidence: Minimum hand confidence to accept projection
            min_face_confidence: Minimum face confidence to accept projection
            diagnostics: Optional diagnostics instance (created if None)
            log_level: Log level for diagnostics (used if diagnostics is None)
        """
        self.virtual_plane = virtual_plane
        self.head_coords = head_coords
        self.use_head_coords_for_ray = use_head_coords_for_ray
        self.min_hand_confidence = min_hand_confidence
        self.min_face_confidence = min_face_confidence

        if diagnostics is not None:
            self.diagnostics = diagnostics
        else:
            self.diagnostics = ProjectionDiagnostics(log_level=log_level)

        # MediaPipe hand landmark indices for index fingertip
        self.INDEX_TIP_IDX = 8  # MediaPipe HandLandmark.INDEX_TIP

        # Statistics
        self._last_result = ProjectionResult()
        self._projection_count = 0
        self._failed_count = 0
        self._gated_count = 0  # Count of confidence-gated rejections

    def set_debug(self, enabled: bool, log_level: ProjectionLogLevel = ProjectionLogLevel.TRACE):
        """
        Enable/disable projection debug mode.

        When enabled, each successful projection logs raw fingertip position,
        ray origin/direction, plane basis, intersection point, parameter t,
        normalized u/v, validity, and confidence.

        Args:
            enabled: Whether to enable debug logging
            log_level: Minimum log level for debug output (default: TRACE)
        """
        if enabled:
            self.diagnostics.log_level = log_level
        else:
            self.diagnostics.log_level = ProjectionLogLevel.OFF

    def _debug_projection(self, label: str, **kwargs):
        """Emit a structured TRACE-level debug event for projection pipeline."""
        if not self.diagnostics._should_log(ProjectionLogLevel.TRACE):
            return
        details = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in kwargs.items()}
        self.diagnostics.log_event(
            ProjectionLogLevel.TRACE, "projection_debug",
            label, details
        )

    def project(
        self,
        hand: Hand,
        face: Face
    ) -> ProjectionResult:
        """
        Project hand index fingertip to virtual plane.

        Args:
            hand: Detected hand with landmarks
            face: Detected face with eye midpoint

        Returns:
            ProjectionResult with normalized coordinates
        """
        self._projection_count += 1

        # Validate inputs
        if not hand or not hand.landmarks or len(hand.landmarks) < 21:
            self._failed_count += 1
            self.diagnostics.log_event(
                ProjectionLogLevel.ERROR, "invalid_hand",
                "Invalid hand landmarks"
            )
            return ProjectionResult(
                valid=False,
                error_message="Invalid hand landmarks"
            )

        if not face or not face.eye_midpoint:
            self._failed_count += 1
            self.diagnostics.log_event(
                ProjectionLogLevel.ERROR, "invalid_face",
                "Invalid face or missing eye midpoint"
            )
            return ProjectionResult(
                valid=False,
                error_message="Invalid face or missing eye midpoint"
            )

        # Confidence gating — reject low-confidence inputs BEFORE ray
        # computation to prevent jitter from noisy landmark estimates.
        hand_conf = getattr(hand, 'confidence', 1.0)
        face_conf = getattr(face, 'confidence', 1.0)

        if hand_conf < self.min_hand_confidence:
            self._gated_count += 1
            self.diagnostics.log_event(
                ProjectionLogLevel.WARN, "confidence_gated_hand",
                f"Hand confidence {hand_conf:.2f} below threshold {self.min_hand_confidence:.2f}",
                {"hand_confidence": hand_conf, "threshold": self.min_hand_confidence}
            )
            return ProjectionResult(
                valid=False,
                error_message=f"Low hand confidence: {hand_conf:.2f}",
                hand_confidence=hand_conf,
                face_confidence=face_conf,
            )

        if face_conf < self.min_face_confidence:
            self._gated_count += 1
            self.diagnostics.log_event(
                ProjectionLogLevel.WARN, "confidence_gated_face",
                f"Face confidence {face_conf:.2f} below threshold {self.min_face_confidence:.2f}",
                {"face_confidence": face_conf, "threshold": self.min_face_confidence}
            )
            return ProjectionResult(
                valid=False,
                error_message=f"Low face confidence: {face_conf:.2f}",
                hand_confidence=hand_conf,
                face_confidence=face_conf,
            )

        if not self.head_coords.is_valid():
            self._failed_count += 1
            self.diagnostics.log_event(
                ProjectionLogLevel.ERROR, "invalid_head_coords",
                "Invalid head coordinate system"
            )
            return ProjectionResult(
                valid=False,
                error_message="Invalid head coordinate system",
                hand_confidence=hand_conf,
                face_confidence=face_conf,
            )

        if not self.virtual_plane.head_coords or not self.virtual_plane.head_coords.is_valid():
            self._failed_count += 1
            self.diagnostics.log_event(
                ProjectionLogLevel.ERROR, "invalid_virtual_plane",
                "Invalid virtual plane"
            )
            return ProjectionResult(
                valid=False,
                error_message="Invalid virtual plane",
                hand_confidence=hand_conf,
                face_confidence=face_conf,
            )

        try:
            # Get index fingertip in camera coordinates
            index_tip = hand.landmarks[self.INDEX_TIP_IDX]
            fingertip_cam = index_tip.to_numpy()

            # Get eye midpoint (ray origin) in camera coordinates
            eye_midpoint_cam = face.eye_midpoint.to_numpy()

            # Compute ray direction (from eye to fingertip)
            ray_direction_cam = fingertip_cam - eye_midpoint_cam
            ray_norm = np.linalg.norm(ray_direction_cam)

            self._debug_projection("project_start",
                fingertip_x=float(fingertip_cam[0]), fingertip_y=float(fingertip_cam[1]),
                fingertip_z=float(fingertip_cam[2]),
                eye_x=float(eye_midpoint_cam[0]), eye_y=float(eye_midpoint_cam[1]),
                eye_z=float(eye_midpoint_cam[2]),
                hand_conf=float(hand_conf), face_conf=float(face_conf))

            if ray_norm < 1e-6:
                self._failed_count += 1
                self.diagnostics.log_event(
                    ProjectionLogLevel.WARN, "ray_too_small",
                    "Ray direction too small (fingertip at eye midpoint)"
                )
                return ProjectionResult(
                    valid=False,
                    error_message="Ray direction too small (fingertip at eye midpoint)",
                    hand_confidence=hand_conf,
                    face_confidence=face_conf,
                )

            ray_direction_cam = ray_direction_cam / ray_norm

            if self.use_head_coords_for_ray:
                # Transform to head coordinates for more accurate intersection
                result = self._project_in_head_coords(
                    eye_midpoint_cam, fingertip_cam, ray_direction_cam
                )
                # Graceful fallback: if the ray missed the plane (parallel or
                # pointing away — common when hand is at eye depth or between
                # eye and plane), fall back to projecting the fingertip's
                # head-space x/y onto the plane surface.  This prevents the
                # tracking loop from hard-freezing on every such frame.
                if not result.valid and result.error_message.startswith("Ray-plane intersection failed"):
                    result = self._fallback_projection(eye_midpoint_cam, fingertip_cam)
                result.hand_confidence = hand_conf
                result.face_confidence = face_conf
                return result
            else:
                # Project directly in camera coordinates
                result = self._project_in_camera_coords(
                    eye_midpoint_cam, ray_direction_cam
                )
                result.hand_confidence = hand_conf
                result.face_confidence = face_conf
                return result

        except Exception as e:
            self._failed_count += 1
            self.diagnostics.log_event(
                ProjectionLogLevel.ERROR, "projection_exception",
                f"Projection error: {e}"
            )
            logger.exception("Projection failed")
            return ProjectionResult(
                valid=False,
                error_message=f"Projection error: {e}"
            )

    def _project_in_head_coords(
        self,
        eye_midpoint_cam: np.ndarray,
        fingertip_cam: np.ndarray,
        ray_direction_cam: np.ndarray
    ) -> ProjectionResult:
        """Project using head coordinate system for accuracy.

        Ray direction is derived from the image-plane (X/Y) angular offset
        between the eye midpoint and the fingertip, with a forced forward
        Z component.  This is necessary because MediaPipe hand-landmark
        Z values are *relative* depth estimates, not absolute distances:
        using them directly in the ray direction places the fingertip
        behind the eye in head space, producing a backward-pointing ray
        that can never reach the virtual plane at z=+distance.

        The image-plane X/Y encodes the angular direction of the fingertip
        relative to the eye midpoint, which is exactly the information we
        need.  We force the Z component to +1 (forward in head coords) so
        the ray always points toward the virtual plane, then solve for the
        intersection distance t.
        """
        # Transform eye midpoint and fingertip to head coordinates
        eye_midpoint_head = self.head_coords.camera_to_head(eye_midpoint_cam)
        fingertip_head = self.head_coords.camera_to_head(fingertip_cam)

        # Compute angular offset in head-space X/Y (image-plane direction).
        # These two components encode *where on the image plane* the
        # fingertip sits relative to the eye midpoint — the true direction
        # we want for the ray.  The Z component is discarded because it is
        # an unreliable relative-depth estimate from MediaPipe.
        dx = fingertip_head[0] - eye_midpoint_head[0]
        dy = fingertip_head[1] - eye_midpoint_head[1]

        # Build the ray direction using the plane distance as the forward
        # (Z) component.  This places the ray-plane intersection exactly at
        # the fingertip's head-space (X, Y) on the plane surface at
        # z=+distance, giving a clean 1:1 mapping between the image-plane
        # position of the fingertip and the normalized cursor coordinates.
        #
        # Using plane_distance (rather than a unit Z=1) means the
        # intersection point is (dx, dy, distance) in head coords — the
        # fingertip's angular position projected straight forward onto the
        # plane.  This is the standard HMD pointing model and preserves
        # head-movement invariance: when the head rotates but the hand stays
        # fixed relative to the head, (dx, dy) is unchanged, so the
        # intersection and the normalized (u, v) are unchanged too.
        ray_direction_head = np.array(
            [dx, dy, self.virtual_plane.distance], dtype=np.float32
        )
        ray_norm = np.linalg.norm(ray_direction_head)

        if ray_norm < 1e-6:
            # Fingertip is exactly at the eye midpoint in image-plane
            # projection — use a straight-forward ray.
            ray_direction_head = np.array(
                [0.0, 0.0, self.virtual_plane.distance], dtype=np.float32
            )
            ray_norm = self.virtual_plane.distance

        ray_direction_head = ray_direction_head / ray_norm

        # Intersect with plane in head coordinates
        intersection_head = self.virtual_plane.ray_plane_intersection_head(
            eye_midpoint_head, ray_direction_head
        )

        if intersection_head is None:
            self._failed_count += 1
            return ProjectionResult(
                valid=False,
                error_message="Ray-plane intersection failed in head coordinates"
            )

        # Convert to normalized coordinates — returns raw u/v WITHOUT
        # clamping.  Out-of-bounds values are valid; the caller (cursor
        # controller) maps them to screen coordinates.
        normalized = self.virtual_plane.point_to_normalized(
            self.head_coords.head_to_camera(intersection_head)
        )

        if normalized is None:
            self._failed_count += 1
            return ProjectionResult(
                valid=False,
                error_message="Failed to convert to normalized coordinates"
            )

        u, v = normalized

        # Also get intersection in camera coordinates for debugging
        intersection_cam = self.head_coords.head_to_camera(intersection_head)

        u, v = normalized

        self._debug_projection("project_head_coords_success",
            eye_head_x=float(eye_midpoint_head[0]), eye_head_y=float(eye_midpoint_head[1]),
            eye_head_z=float(eye_midpoint_head[2]),
            fingertip_head_x=float(fingertip_head[0]), fingertip_head_y=float(fingertip_head[1]),
            fingertip_head_z=float(fingertip_head[2]),
            ray_dir_x=float(ray_direction_head[0]), ray_dir_y=float(ray_direction_head[1]),
            ray_dir_z=float(ray_direction_head[2]),
            plane_normal_x=float(self.virtual_plane._plane_normal_cam[0]) if self.virtual_plane._plane_normal_cam is not None else 0.0,
            plane_normal_y=float(self.virtual_plane._plane_normal_cam[1]) if self.virtual_plane._plane_normal_cam is not None else 0.0,
            plane_normal_z=float(self.virtual_plane._plane_normal_cam[2]) if self.virtual_plane._plane_normal_cam is not None else 0.0,
            intersection_head_x=float(intersection_head[0]), intersection_head_y=float(intersection_head[1]),
            intersection_head_z=float(intersection_head[2]),
            plane_width=self.virtual_plane.width, plane_height=self.virtual_plane.height,
            plane_distance=self.virtual_plane.distance,
            t=float(np.dot(intersection_head - eye_midpoint_head, ray_direction_head) / max(np.dot(ray_direction_head, ray_direction_head), 1e-10)),
            u=float(u), v=float(v),
            valid=True)

        result = ProjectionResult(
            u=u,
            v=v,
            intersection_camera=intersection_cam,
            intersection_head=intersection_head,
            ray_origin_camera=eye_midpoint_cam,
            ray_direction_camera=ray_direction_cam,
            valid=True
        )

        self._last_result = result
        return result

    def _fallback_projection(
        self,
        eye_midpoint_cam: np.ndarray,
        fingertip_cam: np.ndarray
    ) -> ProjectionResult:
        """
        Fallback when the ray misses the plane (parallel or pointing away).

        Projects the fingertip's head-space X/Y onto the plane surface at
        z=distance.  This keeps the cursor responsive when the hand is at
        eye depth or between the eye and the virtual plane, instead of
        freezing.

        The head-space X/Y is clamped to the plane's physical bounds
        (width/2 × height/2) before converting to camera coords, so the
        resulting normalized (u, v) is always in [0, 1].  This avoids the
        situation where the fallback itself produces out-of-bounds values
        that then get clamped by point_to_normalized — instead the clamp
        happens here at the geometry level, which is more correct.
        """
        fingertip_head = self.head_coords.camera_to_head(fingertip_cam)

        # Clamp head-space X/Y to the plane's physical bounds so the
        # resulting normalized coordinates are always in [0, 1].
        half_w = self.virtual_plane.width / 2.0
        half_h = self.virtual_plane.height / 2.0
        clamped_x = float(np.clip(fingertip_head[0], -half_w, half_w))
        clamped_y = float(np.clip(fingertip_head[1], -half_h, half_h))

        # Build the head-space point on the plane surface at z=distance
        point_head = np.array([
            clamped_x,
            clamped_y,
            self.virtual_plane.distance,
        ], dtype=np.float32)
        point_cam = self.head_coords.head_to_camera(point_head)

        normalized = self.virtual_plane.point_to_normalized(point_cam)
        if normalized is None:
            return ProjectionResult(
                valid=False,
                error_message="Fallback projection failed"
            )
        u, v = normalized
        logger.debug(
            f"Fallback projection (ray missed plane): u={u:.3f} v={v:.3f}"
        )
        return ProjectionResult(
            u=u,
            v=v,
            intersection_camera=point_cam,
            intersection_head=point_head,
            ray_origin_camera=eye_midpoint_cam,
            ray_direction_camera=fingertip_cam - eye_midpoint_cam,
            valid=True,
        )

    def _project_in_camera_coords(
        self,
        eye_midpoint_cam: np.ndarray,
        ray_direction_cam: np.ndarray
    ) -> ProjectionResult:
        """Project directly in camera coordinates."""
        # Intersect with plane in camera coordinates
        intersection_cam = self.virtual_plane.ray_plane_intersection(
            eye_midpoint_cam, ray_direction_cam
        )

        if intersection_cam is None:
            self._failed_count += 1
            return ProjectionResult(
                valid=False,
                error_message="Ray-plane intersection failed in camera coordinates"
            )

        # Convert to normalized coordinates — returns raw u/v WITHOUT
        # clamping.  Out-of-bounds values are valid; the caller maps
        # them to screen coordinates.
        normalized = self.virtual_plane.point_to_normalized(intersection_cam)

        if normalized is None:
            self._failed_count += 1
            return ProjectionResult(
                valid=False,
                error_message="Failed to convert to normalized coordinates"
            )

        u, v = normalized

        self._debug_projection("project_cam_coords_success",
            intersection_cam_x=float(intersection_cam[0]), intersection_cam_y=float(intersection_cam[1]),
            intersection_cam_z=float(intersection_cam[2]),
            u=float(u), v=float(v), valid=True)

        result = ProjectionResult(
            u=u,
            v=v,
            intersection_camera=intersection_cam,
            intersection_head=None,
            ray_origin_camera=eye_midpoint_cam,
            ray_direction_camera=ray_direction_cam,
            valid=True
        )

        self._last_result = result
        return result

    def project_from_landmarks(
        self,
        index_tip: Landmark,
        eye_midpoint: FaceLandmark,
        head_coords: Optional[HeadCoordinateSystem] = None
    ) -> ProjectionResult:
        """
        Project from individual landmarks (for testing or alternative input).

        Returns raw u/v values WITHOUT clamping.  Out-of-bounds values
        are valid; the caller maps them to screen coordinates.

        Args:
            index_tip: Index fingertip landmark
            eye_midpoint: Eye midpoint landmark
            head_coords: Optional head coordinate system (uses self.head_coords if None)

        Returns:
            ProjectionResult
        """
        hc = head_coords or self.head_coords

        if not hc or not hc.is_valid():
            return ProjectionResult(valid=False, error_message="Invalid head coordinates")

        fingertip_cam = index_tip.to_numpy()
        eye_cam = eye_midpoint.to_numpy()

        ray_direction = fingertip_cam - eye_cam
        ray_norm = np.linalg.norm(ray_direction)

        if ray_norm < 1e-6:
            return ProjectionResult(valid=False, error_message="Ray too small")

        ray_direction = ray_direction / ray_norm

        if self.use_head_coords_for_ray:
            eye_head = hc.camera_to_head(eye_cam)
            fingertip_head = hc.camera_to_head(fingertip_cam)
            # Use image-plane X/Y for the ray direction with the plane
            # distance as the forward Z (see _project_in_head_coords).
            dx = fingertip_head[0] - eye_head[0]
            dy = fingertip_head[1] - eye_head[1]
            ray_dir_head = np.array(
                [dx, dy, self.virtual_plane.distance], dtype=np.float32
            )
            ray_norm = np.linalg.norm(ray_dir_head)
            if ray_norm < 1e-6:
                ray_dir_head = np.array(
                    [0.0, 0.0, self.virtual_plane.distance], dtype=np.float32
                )
                ray_norm = self.virtual_plane.distance
            ray_dir_head = ray_dir_head / ray_norm

            intersection_head = self.virtual_plane.ray_plane_intersection_head(eye_head, ray_dir_head)

            if intersection_head is None:
                # Graceful fallback: ray missed the plane, project fingertip
                # x/y onto the plane surface at z=distance.  Clamp to the
                # plane's physical bounds so normalized coords are in [0,1].
                half_w = self.virtual_plane.width / 2.0
                half_h = self.virtual_plane.height / 2.0
                clamped_x = float(np.clip(fingertip_head[0], -half_w, half_w))
                clamped_y = float(np.clip(fingertip_head[1], -half_h, half_h))
                point_head = np.array([
                    clamped_x, clamped_y, self.virtual_plane.distance,
                ], dtype=np.float32)
                normalized = self.virtual_plane.point_to_normalized(hc.head_to_camera(point_head))
                if normalized is None:
                    return ProjectionResult(valid=False, error_message="Normalized conversion failed")
                u, v = normalized
                logger.debug("Fallback projection (ray missed plane) in project_from_landmarks")
                return ProjectionResult(
                    u=u, v=v,
                    intersection_camera=hc.head_to_camera(point_head),
                    intersection_head=point_head,
                    ray_origin_camera=eye_cam,
                    ray_direction_camera=ray_direction,
                    valid=True
                )

            normalized = self.virtual_plane.point_to_normalized(hc.head_to_camera(intersection_head))
            if normalized is None:
                return ProjectionResult(valid=False, error_message="Normalized conversion failed")

            u, v = normalized

            # Return raw u/v — out-of-bounds values are valid and handled
            # by the caller (cursor controller maps them to screen coords).
            intersection_cam = hc.head_to_camera(intersection_head)

            return ProjectionResult(
                u=u, v=v,
                intersection_camera=intersection_cam,
                intersection_head=intersection_head,
                ray_origin_camera=eye_cam,
                ray_direction_camera=ray_direction,
                valid=True
            )
        else:
            intersection_cam = self.virtual_plane.ray_plane_intersection(eye_cam, ray_direction)
            if intersection_cam is None:
                return ProjectionResult(valid=False, error_message="No intersection in cam coords")

            normalized = self.virtual_plane.point_to_normalized(intersection_cam)
            if normalized is None:
                return ProjectionResult(valid=False, error_message="Normalized conversion failed")

            u, v = normalized

            # Return raw u/v — out-of-bounds values are valid.
            return ProjectionResult(
                u=u, v=v,
                intersection_camera=intersection_cam,
                ray_origin_camera=eye_cam,
                ray_direction_camera=ray_direction,
                valid=True
            )

    def get_last_result(self) -> ProjectionResult:
        """Get the last projection result."""
        return self._last_result

    def get_stats(self) -> dict:
        """Get projection statistics."""
        return {
            "total_projections": self._projection_count,
            "failed_projections": self._failed_count,
            "gated_projections": self._gated_count,
            "success_rate": (
                (self._projection_count - self._failed_count) / self._projection_count
                if self._projection_count > 0 else 0.0
            ),
            "diagnostics": self.diagnostics.get_summary(),
        }

    def reset_stats(self):
        """Reset projection statistics."""
        self._projection_count = 0
        self._failed_count = 0
        self._gated_count = 0
        self.diagnostics.clear()


def create_projector(
    face: Face,
    virtual_plane_distance: float = 0.30,
    virtual_plane_width: float = 0.70,
    virtual_plane_height: float = 0.50
) -> Optional[HandProjector]:
    """
    Convenience function to create a HandProjector from a face.

    Args:
        face: Detected face with landmarks
        virtual_plane_distance: Distance to virtual plane (meters)
        virtual_plane_width: Plane width (meters)
        virtual_plane_height: Plane height (meters)

    Returns:
        HandProjector instance or None if face invalid
    """
    if not face or not face.eye_midpoint or not face.nose_tip or not face.forehead:
        logger.warning("Face missing required landmarks for projector creation")
        return None

    # Create head coordinate system
    head_coords = HeadCoordinateSystem.from_face(face)
    if not head_coords.is_valid():
        logger.warning("Failed to create valid head coordinate system")
        return None

    # Create virtual plane
    virtual_plane = VirtualDisplayPlane(
        distance=virtual_plane_distance,
        width=virtual_plane_width,
        height=virtual_plane_height,
        head_coords=head_coords
    )

    # Create projector
    return HandProjector(virtual_plane, head_coords)