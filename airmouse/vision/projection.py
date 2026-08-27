"""
3D Projection Module for Air Mouse

Projects hand landmarks (index fingertip) through eye midpoint to virtual display plane.
Provides normalized coordinates for cursor control.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple
from .hand_tracker import Hand, Landmark
from .face_tracker import Face, FaceLandmark
from .head_coords import HeadCoordinateSystem
from .virtual_plane import VirtualDisplayPlane

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        virtual_plane: VirtualDisplayPlane,
        head_coords: HeadCoordinateSystem,
        use_head_coords_for_ray: bool = True
    ):
        """
        Initialize HandProjector.

        Args:
            virtual_plane: Virtual display plane
            head_coords: Head coordinate system
            use_head_coords_for_ray: If True, compute ray in head coordinates for accuracy
        """
        self.virtual_plane = virtual_plane
        self.head_coords = head_coords
        self.use_head_coords_for_ray = use_head_coords_for_ray

        # MediaPipe hand landmark indices for index fingertip
        self.INDEX_TIP_IDX = 8  # MediaPipe HandLandmark.INDEX_TIP

        # Statistics
        self._last_result = ProjectionResult()
        self._projection_count = 0
        self._failed_count = 0

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
            return ProjectionResult(
                valid=False,
                error_message="Invalid hand landmarks"
            )

        if not face or not face.eye_midpoint:
            return ProjectionResult(
                valid=False,
                error_message="Invalid face or missing eye midpoint"
            )

        if not self.head_coords.is_valid():
            return ProjectionResult(
                valid=False,
                error_message="Invalid head coordinate system"
            )

        if not self.virtual_plane.head_coords or not self.virtual_plane.head_coords.is_valid():
            return ProjectionResult(
                valid=False,
                error_message="Invalid virtual plane"
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

            if ray_norm < 1e-6:
                self._failed_count += 1
                return ProjectionResult(
                    valid=False,
                    error_message="Ray direction too small (fingertip at eye midpoint)"
                )

            ray_direction_cam = ray_direction_cam / ray_norm

            if self.use_head_coords_for_ray:
                # Transform to head coordinates for more accurate intersection
                return self._project_in_head_coords(
                    eye_midpoint_cam, fingertip_cam, ray_direction_cam
                )
            else:
                # Project directly in camera coordinates
                return self._project_in_camera_coords(
                    eye_midpoint_cam, ray_direction_cam
                )

        except Exception as e:
            self._failed_count += 1
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
        """Project using head coordinate system for accuracy."""
        # Transform eye midpoint and fingertip to head coordinates
        eye_midpoint_head = self.head_coords.camera_to_head(eye_midpoint_cam)
        fingertip_head = self.head_coords.camera_to_head(fingertip_cam)

        # Ray in head coordinates: from eye midpoint through fingertip
        ray_direction_head = fingertip_head - eye_midpoint_head
        ray_norm = np.linalg.norm(ray_direction_head)

        if ray_norm < 1e-6:
            self._failed_count += 1
            return ProjectionResult(
                valid=False,
                error_message="Ray direction too small in head coordinates"
            )

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

        # Convert to normalized coordinates
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

        # Convert to normalized coordinates
        normalized = self.virtual_plane.point_to_normalized(intersection_cam)

        if normalized is None:
            self._failed_count += 1
            return ProjectionResult(
                valid=False,
                error_message="Failed to convert to normalized coordinates"
            )

        u, v = normalized

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
            ray_dir_head = fingertip_head - eye_head
            ray_dir_head = ray_dir_head / np.linalg.norm(ray_dir_head)

            intersection_head = self.virtual_plane.ray_plane_intersection_head(eye_head, ray_dir_head)

            if intersection_head is None:
                return ProjectionResult(valid=False, error_message="No intersection in head coords")

            normalized = self.virtual_plane.point_to_normalized(hc.head_to_camera(intersection_head))
            if normalized is None:
                return ProjectionResult(valid=False, error_message="Normalized conversion failed")

            u, v = normalized
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
            "success_rate": (
                (self._projection_count - self._failed_count) / self._projection_count
                if self._projection_count > 0 else 0.0
            )
        }

    def reset_stats(self):
        """Reset projection statistics."""
        self._projection_count = 0
        self._failed_count = 0


def create_projector(
    face: Face,
    virtual_plane_distance: float = 0.30,
    virtual_plane_width: float = 0.40,
    virtual_plane_height: float = 0.25
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