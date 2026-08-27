"""
Unit tests for HeadCoordinateSystem module.

Tests orthonormality, right-handedness, and transform round-trip.
"""
import sys
sys.path.insert(0, '/home/shubham/airmouse')

import pytest
import numpy as np
from airmouse.vision.head_coords import HeadCoordinateSystem
from airmouse.vision.face_tracker import Face, FaceLandmark


def create_mock_face() -> Face:
    """Create a mock face with valid landmarks for head coordinate system."""
    landmarks = [FaceLandmark(0.0, 0.0, 0.0, 1.0) for _ in range(468)]
    face = Face(landmarks=landmarks, confidence=1.0)
    face._eye_midpoint = FaceLandmark(0.0, 0.0, 0.0, 1.0)
    face._nose_tip = FaceLandmark(0.0, 0.0, -0.1, 1.0)
    face._forehead = FaceLandmark(0.0, -0.1, 0.0, 1.0)
    return face


class TestHeadCoordinateSystem:
    """Tests for HeadCoordinateSystem class."""

    def test_identity_creation(self):
        """Test creating identity coordinate system."""
        coords = HeadCoordinateSystem()

        # Check origin at zero
        np.testing.assert_array_almost_equal(coords.origin, [0.0, 0.0, 0.0])

        # Check basis vectors (default values)
        np.testing.assert_array_almost_equal(coords.right, [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(coords.up, [0.0, -1.0, 0.0])
        np.testing.assert_array_almost_equal(coords.forward, [0.0, 0.0, -1.0])

    def test_orthonormality(self):
        """Test that basis vectors are orthonormal."""
        coords = HeadCoordinateSystem()

        # Check unit length
        assert abs(np.linalg.norm(coords.right) - 1.0) < 1e-6
        assert abs(np.linalg.norm(coords.up) - 1.0) < 1e-6
        assert abs(np.linalg.norm(coords.forward) - 1.0) < 1e-6

        # Check orthogonality
        assert abs(np.dot(coords.right, coords.up)) < 1e-6
        assert abs(np.dot(coords.right, coords.forward)) < 1e-6
        assert abs(np.dot(coords.up, coords.forward)) < 1e-6

    def test_right_handed(self):
        """Test that coordinate system is right-handed."""
        coords = HeadCoordinateSystem()

        # Cross product right × up should = forward (right-handed)
        cross = np.cross(coords.right, coords.up)
        np.testing.assert_array_almost_equal(cross, coords.forward)

    def test_from_face_basic(self):
        """Test creating coordinate system from face landmarks."""
        face = create_mock_face()

        coords = HeadCoordinateSystem.from_face(face)

        # Origin should be at eye midpoint
        expected_origin = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(coords.origin, expected_origin, decimal=3)

        # Forward should point along -Z (toward nose)
        expected_forward = np.array([0.0, 0.0, -1.0])
        np.testing.assert_array_almost_equal(coords.forward, expected_forward)

        # Should be valid
        assert coords.is_valid() == True

    def test_transform_to_head(self):
        """Test transforming camera point to head coordinates."""
        face = create_mock_face()
        coords = HeadCoordinateSystem.from_face(face)

        # Point at camera origin (which is also head origin in this case)
        camera_point = np.array([0.0, 0.0, 0.0])
        head_point = coords.camera_to_head(camera_point)

        # Should be at head origin
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(head_point, expected)

    def test_transform_to_world(self):
        """Test transforming head point to camera coordinates."""
        face = create_mock_face()
        coords = HeadCoordinateSystem.from_face(face)

        # Point at head origin
        head_point = np.array([0.0, 0.0, 0.0])
        camera_point = coords.head_to_camera(head_point)

        # Should be camera origin (same as head origin in this setup)
        np.testing.assert_array_almost_equal(camera_point, coords.origin)

    def test_round_trip_transform(self):
        """Test that camera->head->camera round-trip returns original point."""
        face = create_mock_face()
        coords = HeadCoordinateSystem.from_face(face)

        # Test point
        camera_point = np.array([1.0, 2.0, 3.0])

        # Round trip
        head_point = coords.camera_to_head(camera_point)
        back_to_camera = coords.head_to_camera(head_point)

        np.testing.assert_array_almost_equal(back_to_camera, camera_point)

    def test_get_transform_matrix(self):
        """Test getting 4x4 transformation matrix."""
        coords = HeadCoordinateSystem()

        matrix = coords.get_transform_matrix()

        assert matrix.shape == (4, 4)

        # Last row should be [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(matrix[3], [0, 0, 0, 1])

    def test_inverse_transform_matrix(self):
        """Test that inverse matrix correctly inverts transform."""
        face = create_mock_face()
        coords = HeadCoordinateSystem.from_face(face)

        matrix = coords.get_transform_matrix()
        inv_matrix = coords.get_inverse_transform_matrix()

        # Matrix * inverse should be identity
        result = matrix @ inv_matrix
        np.testing.assert_array_almost_equal(result, np.eye(4))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])