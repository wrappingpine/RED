"""
Head Coordinate System Module for Air Mouse

Provides coordinate transformations between camera space and head-relative space.
Head coordinate system:
- Origin: midpoint between left and right eye centers
- X axis (right): cross(forward, up) - pointing to user's right
- Y axis (up): cross(right, forward) - pointing up
- Z axis (forward): nose_tip - eye_midpoint - pointing forward from face
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple
from .face_tracker import Face, FaceLandmark

logger = logging.getLogger(__name__)


@dataclass
class HeadCoordinateSystem:
    """
    Head-centered coordinate system for mapping camera-space points to head-relative coordinates.

    Coordinate System Definition:
    - Origin: Eye midpoint (between left and right eye centers)
    - Forward (Z): Normalized vector from eye midpoint to nose tip
    - Right (X): Normalized cross(forward, up)
    - Up (Y): Normalized cross(right, forward)

    This creates a right-handed coordinate system where:
    - +X points to user's right
    - +Y points up
    - +Z points forward from the face
    """

    # Basis vectors in camera coordinates
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    forward: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=np.float32))
    right: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0], dtype=np.float32))

    # Cached 4x4 transformation matrix (camera -> head)
    _transform_matrix: Optional[np.ndarray] = None
    _inverse_matrix: Optional[np.ndarray] = None

    # Validity flag
    _valid: bool = False

    # Temporal smoothing state
    _smoothed_origin: Optional[np.ndarray] = None
    _smoothed_forward: Optional[np.ndarray] = None
    _smoothed_right: Optional[np.ndarray] = None
    _smoothed_up: Optional[np.ndarray] = None
    _smoothing_alpha: float = 0.3

    @classmethod
    def from_face(cls, face: Face, smoothing_alpha: float = 0.3,
                  prev_coords: Optional["HeadCoordinateSystem"] = None) -> "HeadCoordinateSystem":
        """
        Create HeadCoordinateSystem from a detected Face with optional temporal smoothing.

        Args:
            face: Face object with computed eye_midpoint, nose_tip, forehead
            smoothing_alpha: Smoothing factor (0=no smoothing, 1=max). Default 0.3
            prev_coords: Previous HeadCoordinateSystem for temporal smoothing

        Returns:
            HeadCoordinateSystem instance
        """
        if not (face.eye_midpoint and face.nose_tip and face.forehead):
            logger.warning("Face missing required landmarks for head coordinate system")
            return cls(_valid=False)

        # Origin = eye midpoint in camera coordinates
        origin = face.eye_midpoint.to_numpy()

        # Forward vector = normalized(nose_tip - eye_midpoint)
        nose = face.nose_tip.to_numpy()
        forward = nose - origin
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            logger.warning("Forward vector too small, using default")
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward = forward / forward_norm

        # Up vector (rough) = normalized(forehead - eye_midpoint)
        forehead = face.forehead.to_numpy()
        up_rough = forehead - origin
        up_norm = np.linalg.norm(up_rough)
        if up_norm < 1e-6:
            logger.warning("Up vector too small, using default")
            up_rough = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        else:
            up_rough = up_rough / up_norm

        # Right vector = cross(up_rough, forward) for right-handed system
        right = np.cross(up_rough, forward)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            logger.warning("Right vector too small, using default")
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = right / right_norm

        # Recompute up to be perfectly orthogonal: up = cross(forward, right)
        # (forward, right in camera coords gives up in camera coords for right-handed system)
        up = np.cross(forward, right)

        # Verify orthonormality
        cls._verify_basis(forward, right, up)

        # Apply temporal smoothing if previous coordinates available
        if prev_coords is not None and prev_coords.is_valid():
            origin = cls._lerp_vector(origin, prev_coords.origin, smoothing_alpha)
            forward = cls._lerp_vector(forward, prev_coords.forward, smoothing_alpha)
            right = cls._lerp_vector(right, prev_coords.right, smoothing_alpha)
            up = cls._lerp_vector(up, prev_coords.up, smoothing_alpha)

            # Re-verify after smoothing
            cls._verify_basis(forward, right, up)

        coords = cls(
            origin=origin,
            forward=forward,
            right=right,
            up=up,
            _valid=True,
            _smoothing_alpha=smoothing_alpha
        )

        # Cache smoothed values for next frame
        coords._smoothed_origin = origin.copy()
        coords._smoothed_forward = forward.copy()
        coords._smoothed_right = right.copy()
        coords._smoothed_up = up.copy()

        return coords

    @staticmethod
    def _lerp_vector(current: np.ndarray, previous: np.ndarray, alpha: float) -> np.ndarray:
        """Linear interpolation between current and previous vectors."""
        return alpha * current + (1.0 - alpha) * previous

    @staticmethod
    def _verify_basis(forward: np.ndarray, right: np.ndarray, up: np.ndarray):
        """Verify basis vectors form a valid right-handed orthonormal basis."""
        # Check unit length
        for name, vec in [("forward", forward), ("right", right), ("up", up)]:
            norm = np.linalg.norm(vec)
            if abs(norm - 1.0) > 1e-3:
                logger.warning(f"{name} vector not unit length: {norm:.6f}")

        # Check orthogonality
        dot_fr = np.dot(forward, right)
        dot_fu = np.dot(forward, up)
        dot_ru = np.dot(right, up)
        if abs(dot_fr) > 1e-3 or abs(dot_fu) > 1e-3 or abs(dot_ru) > 1e-3:
            logger.warning(f"Basis not orthogonal: f·r={dot_fr:.6f}, f·u={dot_fu:.6f}, r·u={dot_ru:.6f}")

        # Check right-handed (det > 0)
        det = np.linalg.det(np.column_stack([right, up, forward]))
        if det < 0:
            logger.warning(f"Basis not right-handed: det={det:.6f}")

    def is_valid(self) -> bool:
        """Check if coordinate system is valid."""
        return self._valid

    def get_transform_matrix(self) -> np.ndarray:
        """
        Get 4x4 transformation matrix from camera coordinates to head coordinates.

        Transforms a point from camera space to head space:
        p_head = T @ p_camera_homogeneous

        Returns:
            4x4 transformation matrix
        """
        if self._transform_matrix is not None:
            return self._transform_matrix

        if not self._valid:
            return np.eye(4, dtype=np.float32)

        # Rotation matrix: columns are basis vectors (right, up, forward) in camera coords
        # For camera->head, we need R.T because we're projecting onto basis vectors
        R = np.column_stack([self.right, self.up, self.forward]).astype(np.float32)

        # Translation: R^T @ (-origin) to transform origin to zero
        t = -R.T @ self.origin

        # 4x4 transform matrix
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.T
        T[:3, 3] = t

        self._transform_matrix = T
        return T

    def get_inverse_transform_matrix(self) -> np.ndarray:
        """
        Get 4x4 transformation matrix from head coordinates to camera coordinates.

        Transforms a point from head space to camera space:
        p_camera = T_inv @ p_head_homogeneous

        Returns:
            4x4 inverse transformation matrix
        """
        if self._inverse_matrix is not None:
            return self._inverse_matrix

        if not self._valid:
            return np.eye(4, dtype=np.float32)

        # Rotation matrix: columns are basis vectors (right, up, forward) in camera coords
        R = np.column_stack([self.right, self.up, self.forward]).astype(np.float32)

        # For head->camera: p_cam = R @ p_head + origin
        T_inv = np.eye(4, dtype=np.float32)
        T_inv[:3, :3] = R
        T_inv[:3, 3] = self.origin

        self._inverse_matrix = T_inv
        return T_inv

    def camera_to_head(self, point: np.ndarray) -> np.ndarray:
        """
        Transform a 3D point from camera coordinates to head coordinates.

        Args:
            point: 3D point in camera coordinates (x, y, z)

        Returns:
            3D point in head coordinates
        """
        if not self._valid:
            return point.copy()

        T = self.get_transform_matrix()

        # Homogeneous coordinates
        p_homo = np.append(point.astype(np.float32), 1.0)
        p_head = T @ p_homo

        return p_head[:3]

    def head_to_camera(self, point: np.ndarray) -> np.ndarray:
        """
        Transform a 3D point from head coordinates to camera coordinates.

        Args:
            point: 3D point in head coordinates (x, y, z)

        Returns:
            3D point in camera coordinates
        """
        if not self._valid:
            return point.copy()

        T_inv = self.get_inverse_transform_matrix()

        # Homogeneous coordinates
        p_homo = np.append(point.astype(np.float32), 1.0)
        p_camera = T_inv @ p_homo

        return p_camera[:3]

    def camera_to_head_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Transform multiple 3D points from camera to head coordinates.

        Args:
            points: Nx3 array of points in camera coordinates

        Returns:
            Nx3 array of points in head coordinates
        """
        if not self._valid:
            return points.copy()

        if points.shape[1] != 3:
            raise ValueError("Points must be Nx3")

        T = self.get_transform_matrix()

        # Add homogeneous coordinate
        n = points.shape[0]
        points_homo = np.hstack([points.astype(np.float32), np.ones((n, 1), dtype=np.float32)])

        # Transform
        points_head = (T @ points_homo.T).T

        return points_head[:, :3]

    def get_forward_vector(self) -> np.ndarray:
        """Get forward vector (Z axis) in camera coordinates."""
        return self.forward.copy()

    def get_up_vector(self) -> np.ndarray:
        """Get up vector (Y axis) in camera coordinates."""
        return self.up.copy()

    def get_right_vector(self) -> np.ndarray:
        """Get right vector (X axis) in camera coordinates."""
        return self.right.copy()

    def get_origin(self) -> np.ndarray:
        """Get origin (eye midpoint) in camera coordinates."""
        return self.origin.copy()

    def get_basis_matrix(self) -> np.ndarray:
        """
        Get 3x3 rotation matrix (basis vectors as columns).

        Returns:
            3x3 matrix with columns [right, up, forward]
        """
        return np.column_stack([self.right, self.up, self.forward]).astype(np.float32)

    def __repr__(self) -> str:
        if not self._valid:
            return "HeadCoordinateSystem(invalid)"
        return (f"HeadCoordinateSystem(origin={self.origin}, "
                f"forward={self.forward}, right={self.right}, up={self.up})")


def create_head_coords_from_landmarks(
    eye_midpoint: FaceLandmark,
    nose_tip: FaceLandmark,
    forehead: FaceLandmark
) -> HeadCoordinateSystem:
    """
    Convenience function to create HeadCoordinateSystem from individual landmarks.

    Args:
        eye_midpoint: Eye midpoint landmark
        nose_tip: Nose tip landmark
        forehead: Forehead landmark

    Returns:
        HeadCoordinateSystem instance
    """
    # Create a minimal Face object
    face = Face(landmarks=[], confidence=1.0)
    face._eye_midpoint = eye_midpoint
    face._nose_tip = nose_tip
    face._forehead = forehead

    return HeadCoordinateSystem.from_face(face)