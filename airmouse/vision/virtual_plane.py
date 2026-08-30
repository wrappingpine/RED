"""
Virtual Display Plane Module for Air Mouse

Defines a virtual display plane anchored to the user's head at a fixed distance.
The plane serves as the interaction surface - hand rays intersect this plane
to produce normalized cursor coordinates.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from .head_coords import HeadCoordinateSystem

logger = logging.getLogger(__name__)


@dataclass
class VirtualDisplayPlane:
    """
    Virtual display plane anchored to head coordinate system.

    Plane Properties:
    - Distance: 30cm in front of face (along -forward direction in head coords)
    - Size: 40cm wide x 25cm high (~16:10 aspect ratio)
    - Normal: Faces the user (-forward in head coords, i.e., -Z in head space)
    - Center: At origin + forward * distance in head coordinates

    In head coordinate system:
    - Plane center: (0, 0, distance)
    - Plane normal: (0, 0, -1)
    - X range: [-width/2, width/2]
    - Y range: [-height/2, height/2]
    - Z = distance (constant)

    Normalized coordinates (0,0) to (1,1) map to plane:
    - u = (x + width/2) / width
    - v = (y + height/2) / height
    """

    # Plane dimensions (meters)
    distance: float = 0.30  # 30cm in front of face
    width: float = 0.40     # 40cm wide
    height: float = 0.25    # 25cm high (~16:10 aspect)

    # Head coordinate system reference
    head_coords: Optional[HeadCoordinateSystem] = None

    # Cached plane properties in camera coordinates
    _plane_center_cam: Optional[np.ndarray] = None
    _plane_normal_cam: Optional[np.ndarray] = None
    _plane_x_axis_cam: Optional[np.ndarray] = None
    _plane_y_axis_cam: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate plane parameters."""
        if self.distance <= 0:
            raise ValueError("Plane distance must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Plane dimensions must be positive")
        if self.head_coords is not None:
            self._recompute_plane_camera()

    def update_head_coords(self, head_coords: HeadCoordinateSystem):
        """
        Update the head coordinate system and recompute plane in camera coordinates.

        Args:
            head_coords: Current head coordinate system
        """
        self.head_coords = head_coords
        self._recompute_plane_camera()

    def _recompute_plane_camera(self):
        """Recompute plane center and axes in camera coordinates."""
        if self.head_coords is None or not self.head_coords.is_valid():
            self._plane_center_cam = None
            self._plane_normal_cam = None
            self._plane_x_axis_cam = None
            self._plane_y_axis_cam = None
            return

        # In head coordinates:
        # - Plane center: (0, 0, distance)
        # - Plane normal: (0, 0, -1) (facing user)
        # - X axis: (1, 0, 0) (right)
        # - Y axis: (0, 1, 0) (up)

        # Transform to camera coordinates
        center_head = np.array([0.0, 0.0, self.distance], dtype=np.float32)
        normal_head = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        x_axis_head = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_axis_head = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self._plane_center_cam = self.head_coords.head_to_camera(center_head)
        self._plane_normal_cam = self.head_coords.head_to_camera(normal_head) - self.head_coords.head_to_camera(np.zeros(3))
        self._plane_x_axis_cam = self.head_coords.head_to_camera(x_axis_head) - self.head_coords.head_to_camera(np.zeros(3))
        self._plane_y_axis_cam = self.head_coords.head_to_camera(y_axis_head) - self.head_coords.head_to_camera(np.zeros(3))

        # Normalize
        for attr in ['_plane_normal_cam', '_plane_x_axis_cam', '_plane_y_axis_cam']:
            vec = getattr(self, attr)
            if vec is not None:
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    setattr(self, attr, vec / norm)

    def get_plane_center_camera(self) -> Optional[np.ndarray]:
        """Get plane center in camera coordinates."""
        return self._plane_center_cam.copy() if self._plane_center_cam is not None else None

    def get_plane_normal_camera(self) -> Optional[np.ndarray]:
        """Get plane normal in camera coordinates (points toward user)."""
        return self._plane_normal_cam.copy() if self._plane_normal_cam is not None else None

    def get_plane_axes_camera(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get plane X and Y axes in camera coordinates."""
        return (
            self._plane_x_axis_cam.copy() if self._plane_x_axis_cam is not None else None,
            self._plane_y_axis_cam.copy() if self._plane_y_axis_cam is not None else None
        )

    def ray_plane_intersection(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute intersection of ray with virtual plane.

        Args:
            ray_origin: 3D ray origin in camera coordinates
            ray_direction: 3D ray direction (normalized) in camera coordinates

        Returns:
            3D intersection point in camera coordinates, or None if no intersection
        """
        if (self._plane_center_cam is None or self._plane_normal_cam is None or
                ray_origin is None or ray_direction is None):
            return None

        # Ray: P = ray_origin + t * ray_direction
        # Plane: (P - plane_center) · plane_normal = 0
        # Solve for t: (ray_origin + t*ray_direction - plane_center) · plane_normal = 0
        # t = (plane_center - ray_origin) · plane_normal / (ray_direction · plane_normal)

        denom = np.dot(ray_direction, self._plane_normal_cam)

        # Ray parallel to plane (or pointing away)
        if abs(denom) < 1e-6:
            return None

        t = np.dot(self._plane_center_cam - ray_origin, self._plane_normal_cam) / denom

        # Intersection behind ray origin (t < 0) - ray points away from plane
        if t < 0:
            return None

        intersection = ray_origin + t * ray_direction
        return intersection

    def ray_plane_intersection_head(
        self,
        ray_origin_head: np.ndarray,
        ray_direction_head: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute intersection of ray with virtual plane in head coordinates.

        Simpler computation since plane is axis-aligned in head coordinates.

        Args:
            ray_origin: 3D ray origin in head coordinates
            ray_direction: 3D ray direction (normalized) in head coordinates

        Returns:
            3D intersection point in head coordinates, or None if no intersection
        """
        # Plane in head coords: z = distance, normal = (0, 0, -1) (facing user)
        # Ray: P = ray_origin + t * ray_direction
        # Intersection when z = distance:
        # ray_origin[2] + t * ray_direction[2] = distance
        # t = (distance - ray_origin[2]) / ray_direction[2]
        #
        # Ray must point toward positive Z to hit the plane (which is at z=+distance)

        if ray_direction_head[2] <= 1e-6:  # Ray not pointing toward plane (positive Z)
            return None

        t = (self.distance - ray_origin_head[2]) / ray_direction_head[2]

        if t < 0:
            return None

        intersection = ray_origin_head + t * ray_direction_head
        return intersection

    def point_to_normalized(self, point_camera: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Convert 3D point on plane to normalized (u, v) coordinates [0, 1] x [0, 1].

        Args:
            point_camera: 3D point on plane in camera coordinates

        Returns:
            (u, v) normalized coordinates, or None if invalid
        """
        if self.head_coords is None or not self.head_coords.is_valid():
            return None

        # Transform to head coordinates
        point_head = self.head_coords.camera_to_head(point_camera)

        # In head coords, plane is at z = distance
        # Check point is on plane (within tolerance)
        if abs(point_head[2] - self.distance) > 0.01:  # 1cm tolerance
            logger.debug(f"Point not on plane: z={point_head[2]:.4f}, expected={self.distance}")
            # Still project

        # Normalized coordinates
        # X in head coords maps to u: [-width/2, width/2] -> [0, 1]
        # Y in head coords maps to v: [-height/2, height/2] -> [0, 1]
        u = (point_head[0] + self.width / 2) / self.width
        v = (point_head[1] + self.height / 2) / self.height

        # Clamp to [0, 1]
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        return (float(u), float(v))

    def normalized_to_point_camera(self, u: float, v: float) -> Optional[np.ndarray]:
        """
        Convert normalized (u, v) coordinates to 3D point on plane in camera coordinates.

        Args:
            u: Normalized X coordinate [0, 1]
            v: Normalized Y coordinate [0, 1]

        Returns:
            3D point on plane in camera coordinates, or None if invalid
        """
        if self.head_coords is None or not self.head_coords.is_valid():
            return None

        # Clamp
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        # Head coordinates
        x = (u - 0.5) * self.width
        y = (v - 0.5) * self.height
        z = self.distance

        point_head = np.array([x, y, z], dtype=np.float32)

        # Transform to camera coordinates
        return self.head_coords.head_to_camera(point_head)

    def is_point_on_plane(self, point_camera: np.ndarray, tolerance: float = 0.01) -> bool:
        """
        Check if a point lies on the virtual plane (within tolerance).

        Args:
            point_camera: 3D point in camera coordinates
            tolerance: Distance tolerance in meters

        Returns:
            True if point is on plane
        """
        if self.head_coords is None or not self.head_coords.is_valid():
            return False

        point_head = self.head_coords.camera_to_head(point_camera)
        return abs(point_head[2] - self.distance) <= tolerance

    def get_plane_bounds_head(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get plane bounds in head coordinates.

        Returns:
            (min_corner, max_corner) in head coordinates
        """
        min_corner = np.array([-self.width/2, -self.height/2, self.distance], dtype=np.float32)
        max_corner = np.array([self.width/2, self.height/2, self.distance], dtype=np.float32)
        return min_corner, max_corner

    def get_plane_corners_camera(self) -> Optional[np.ndarray]:
        """
        Get 4 corners of the plane in camera coordinates.

        Returns:
            4x3 array of corner points, or None if invalid
        """
        if self.head_coords is None or not self.head_coords.is_valid():
            return None

        corners_head = np.array([
            [-self.width/2, -self.height/2, self.distance],
            [self.width/2, -self.height/2, self.distance],
            [self.width/2, self.height/2, self.distance],
            [-self.width/2, self.height/2, self.distance],
        ], dtype=np.float32)

        return self.head_coords.camera_to_head_batch(corners_head)

    def draw_debug(self, frame, camera_matrix=None, dist_coeffs=None,
                  ray_origin=None, ray_direction=None, intersection=None,
                  normalized_coords=None):
        """
        Draw comprehensive debug overlay on frame.

        Args:
            frame: BGR image to draw on
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
            ray_origin: 3D ray origin in camera coordinates (e.g., eye midpoint)
            ray_direction: 3D ray direction in camera coordinates (normalized)
            intersection: 3D intersection point in camera coordinates
            normalized_coords: (u, v) normalized coordinates on plane [0,1]x[0,1]

        Returns:
            Annotated frame
        """
        if camera_matrix is None or dist_coeffs is None:
            return frame

        if self.head_coords is None or not self.head_coords.is_valid():
            return frame

        try:
            import cv2

            # 1. Draw virtual plane corners (projected to 2D)
            corners = self.get_plane_corners_camera()
            if corners is not None:
                corners_2d, _ = cv2.projectPoints(
                    corners, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
                )
                corners_2d = corners_2d.reshape(-1, 2).astype(int)
                cv2.polylines(frame, [corners_2d], True, (0, 255, 255), 2)

                # Draw center
                center = self.get_plane_center_camera()
                if center is not None:
                    center_2d, _ = cv2.projectPoints(
                        center.reshape(1, 3), np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
                    )
                    cx, cy = center_2d[0, 0].astype(int)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

            # 2. Draw head coordinate axes (forward, right, up) at eye midpoint
            origin = self.head_coords.get_origin()
            if origin is not None:
                # Axis length in meters
                axis_len = 0.1  # 10cm

                # Forward axis (blue - Z)
                forward_end = origin + self.head_coords.get_forward_vector() * axis_len
                fwd_2d, _ = cv2.projectPoints(
                    np.array([origin, forward_end], dtype=np.float32),
                    np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
                )
                fwd_2d = fwd_2d.reshape(-1, 2).astype(int)
                cv2.arrowedLine(frame, tuple(fwd_2d[0]), tuple(fwd_2d[1]), (255, 0, 0), 2, tipLength=0.3)
                cv2.putText(frame, "F", tuple(fwd_2d[1] + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Right axis (red - X)
                right_end = origin + self.head_coords.get_right_vector() * axis_len
                right_2d, _ = cv2.projectPoints(
                    np.array([origin, right_end], dtype=np.float32),
                    np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
                )
                right_2d = right_2d.reshape(-1, 2).astype(int)
                cv2.arrowedLine(frame, tuple(right_2d[0]), tuple(right_2d[1]), (0, 0, 255), 2, tipLength=0.3)
                cv2.putText(frame, "R", tuple(right_2d[1] + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Up axis (green - Y)
                up_end = origin + self.head_coords.get_up_vector() * axis_len
                up_2d, _ = cv2.projectPoints(
                    np.array([origin, up_end], dtype=np.float32),
                    np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
                )
                up_2d = up_2d.reshape(-1, 2).astype(int)
                cv2.arrowedLine(frame, tuple(up_2d[0]), tuple(up_2d[1]), (0, 255, 0), 2, tipLength=0.3)
                cv2.putText(frame, "U", tuple(up_2d[1] + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 3. Draw ray from eye midpoint through fingertip
            if ray_origin is not None and ray_direction is not None:
                # Draw ray line (extend to plane distance or a bit further)
                ray_end = ray_origin + ray_direction * (self.distance + 0.1)
                ray_pts_2d, _ = cv2.projectPoints(
                    np.array([ray_origin, ray_end], dtype=np.float32),
                    np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
                )
                ray_pts_2d = ray_pts_2d.reshape(-1, 2).astype(int)
                cv2.line(frame, tuple(ray_pts_2d[0]), tuple(ray_pts_2d[1]), (255, 255, 0), 2)
                cv2.circle(frame, tuple(ray_pts_2d[0]), 4, (255, 255, 0), -1)  # Ray origin

            # 4. Draw ray-plane intersection point
            if intersection is not None:
                inter_2d, _ = cv2.projectPoints(
                    intersection.reshape(1, 3), np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
                )
                inter_2d = inter_2d.reshape(-1, 2).astype(int)
                cv2.circle(frame, tuple(inter_2d[0]), 8, (0, 255, 0), 2)
                cv2.circle(frame, tuple(inter_2d[0]), 4, (0, 255, 0), -1)

            # 5. Draw normalized coordinates text
            if normalized_coords is not None:
                u, v = normalized_coords
                # Project plane center for text position
                center = self.get_plane_center_camera()
                if center is not None:
                    center_2d, _ = cv2.projectPoints(
                        center.reshape(1, 3), np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
                    )
                    cx, cy = center_2d[0, 0].astype(int)
                    text = f"u={u:.3f}, v={v:.3f}"
                    cv2.putText(frame, text, (cx + 10, cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, text, (cx + 10, cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        except Exception as e:
            logger.debug(f"Failed to draw plane debug: {e}")

        return frame

    def __repr__(self) -> str:
        return (f"VirtualDisplayPlane(distance={self.distance:.2f}m, "
                f"size={self.width:.2f}x{self.height:.2f}m, "
                f"valid={self.head_coords is not None and self.head_coords.is_valid()})")