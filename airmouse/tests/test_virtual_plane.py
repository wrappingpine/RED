"""
Unit tests for VirtualDisplayPlane module.

Tests ray-plane intersection accuracy and coordinate mapping.
"""
import sys
sys.path.insert(0, '/home/shubham/airmouse')

import pytest
import numpy as np
from airmouse.vision.virtual_plane import VirtualDisplayPlane
from airmouse.vision.head_coords import HeadCoordinateSystem
from airmouse.vision.face_tracker import Face, FaceLandmark


def create_mock_face() -> Face:
    """Create a mock face with valid landmarks for head coordinate system."""
    # Create face with proper landmarks
    landmarks = [FaceLandmark(0.0, 0.0, 0.0, 1.0) for _ in range(468)]
    face = Face(landmarks=landmarks, confidence=1.0)

    # Eye midpoint at origin
    face._eye_midpoint = FaceLandmark(0.0, 0.0, 0.0, 1.0)
    # Nose tip forward along -Z (camera looks along -Z)
    face._nose_tip = FaceLandmark(0.0, 0.0, -0.1, 1.0)
    # Forehead up along -Y
    face._forehead = FaceLandmark(0.0, -0.1, 0.0, 1.0)

    return face


def create_valid_head_coords() -> HeadCoordinateSystem:
    """Create a valid HeadCoordinateSystem from a mock face."""
    face = create_mock_face()
    return HeadCoordinateSystem.from_face(face)


class TestVirtualDisplayPlane:
    """Tests for VirtualDisplayPlane class."""

    def test_plane_creation_defaults(self):
        """Test plane creation with default parameters."""
        plane = VirtualDisplayPlane()
        assert plane.distance == 0.30
        assert plane.width == 0.40
        assert plane.height == 0.25

    def test_plane_creation_custom(self):
        """Test plane creation with custom parameters."""
        plane = VirtualDisplayPlane(distance=0.50, width=0.60, height=0.40)
        assert plane.distance == 0.50
        assert plane.width == 0.60
        assert plane.height == 0.40

    def test_ray_plane_intersection_head_center(self):
        """Test ray-plane intersection at center of plane in head coordinates."""
        plane = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25)

        # Ray from origin (0,0,0) through (0,0,-1) - straight forward in head coords
        ray_origin = np.array([0.0, 0.0, 0.0])
        ray_dir = np.array([0.0, 0.0, 1.0])

        intersection = plane.ray_plane_intersection_head(ray_origin, ray_dir)

        assert intersection is not None
        # At center: x=0, y=0, z=distance
        assert abs(intersection[0]) < 1e-6
        assert abs(intersection[1]) < 1e-6
        assert abs(intersection[2] - 0.30) < 1e-6

    def test_ray_plane_intersection_head_edges(self):
        """Test ray-plane intersection at edges of plane in head coordinates."""
        plane = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25)

        # Top-left corner: x=-0.20, y=0.125 at z=0.30
        ray_origin = np.array([0.0, 0.0, 0.0])
        ray_dir = np.array([-0.20, 0.125, 0.30])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        intersection = plane.ray_plane_intersection_head(ray_origin, ray_dir)

        assert intersection is not None
        # Check x, y coordinates match
        assert abs(intersection[0] - (-0.20)) < 1e-3
        assert abs(intersection[1] - 0.125) < 1e-3
        assert abs(intersection[2] - 0.30) < 1e-6

        # Bottom-right corner
        ray_dir = np.array([0.20, -0.125, 0.30])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        intersection = plane.ray_plane_intersection_head(ray_origin, ray_dir)

        assert intersection is not None
        assert abs(intersection[0] - 0.20) < 1e-3
        assert abs(intersection[1] - (-0.125)) < 1e-3

    def test_ray_plane_intersection_behind(self):
        """Test ray that doesn't intersect plane (pointing away in head coords)."""
        plane = VirtualDisplayPlane(distance=0.30)

        # Ray pointing backward (negative Z in head coords)
        ray_origin = np.array([0.0, 0.0, 0.0])
        ray_dir = np.array([0.0, 0.0, -1.0])

        intersection = plane.ray_plane_intersection_head(ray_origin, ray_dir)

        # Should return None for no intersection
        assert intersection is None

    def test_ray_plane_parallel(self):
        """Test ray parallel to plane (no Z component in head coords)."""
        plane = VirtualDisplayPlane(distance=0.30)

        # Ray parallel to plane (no Z component)
        ray_origin = np.array([0.0, 0.0, 0.0])
        ray_dir = np.array([1.0, 0.0, 0.0])

        intersection = plane.ray_plane_intersection_head(ray_origin, ray_dir)

        # Should return None for no intersection (ray_direction[2] >= 0)
        assert intersection is None

    def test_point_to_normalized(self):
        """Test conversion from 3D point on plane to normalized (u, v) coordinates."""
        # Need head_coords for this test
        head_coords = create_valid_head_coords()
        plane = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25, head_coords=head_coords)

        # Center of plane in head coords
        point_head = np.array([0.0, 0.0, 0.30])
        # Transform to camera coords
        point_camera = head_coords.head_to_camera(point_head)

        u, v = plane.point_to_normalized(point_camera)

        # Center should be (0.5, 0.5)
        assert abs(u - 0.5) < 1e-6
        assert abs(v - 0.5) < 1e-6

    def test_normalized_to_point_camera(self):
        """Test conversion from normalized (u, v) to 3D point on plane."""
        head_coords = create_valid_head_coords()
        plane = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25, head_coords=head_coords)

        # Center of plane
        point = plane.normalized_to_point_camera(0.5, 0.5)

        assert point is not None
        # Transform back to head coords to verify
        point_head = head_coords.camera_to_head(point)
        assert abs(point_head[0]) < 1e-6
        assert abs(point_head[1]) < 1e-6
        assert abs(point_head[2] - 0.30) < 1e-6

        # Top-left
        point = plane.normalized_to_point_camera(0.0, 1.0)
        point_head = head_coords.camera_to_head(point)
        assert abs(point_head[0] - (-0.20)) < 1e-6
        assert abs(point_head[1] - 0.125) < 1e-6

        # Bottom-right
        point = plane.normalized_to_point_camera(1.0, 0.0)
        point_head = head_coords.camera_to_head(point)
        assert abs(point_head[0] - 0.20) < 1e-6
        assert abs(point_head[1] - (-0.125)) < 1e-6

    def test_is_point_on_plane(self):
        """Test checking if 3D point lies on plane."""
        head_coords = create_valid_head_coords()
        plane = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25, head_coords=head_coords)

        # Point on plane
        point_head = np.array([0.0, 0.0, 0.30])
        point_camera = head_coords.head_to_camera(point_head)
        assert plane.is_point_on_plane(point_camera) == True

        # Point off plane (different Z)
        point_head = np.array([0.0, 0.0, 0.50])
        point_camera = head_coords.head_to_camera(point_head)
        assert plane.is_point_on_plane(point_camera) == False

    def test_get_plane_bounds_head(self):
        """Test getting plane bounds in head coordinates."""
        plane = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25)

        min_corner, max_corner = plane.get_plane_bounds_head()

        assert min_corner[0] == -0.20
        assert min_corner[1] == -0.125
        assert min_corner[2] == 0.30
        assert max_corner[0] == 0.20
        assert max_corner[1] == 0.125
        assert max_corner[2] == 0.30

    def test_get_plane_corners_camera(self):
        """Test getting plane corners in camera coordinates."""
        head_coords = create_valid_head_coords()
        plane = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25, head_coords=head_coords)

        corners = plane.get_plane_corners_camera()

        assert corners is not None
        assert corners.shape == (4, 3)

    def test_repr(self):
        """Test string representation."""
        plane = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25)
        repr_str = repr(plane)
        assert "0.30" in repr_str
        assert "0.40x0.25" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])