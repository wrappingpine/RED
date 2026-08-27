"""
Unit tests for HandProjector module.

Tests 3D fingertip projection accuracy with synthetic landmarks.
"""
import sys
sys.path.insert(0, '/home/shubham/airmouse')

import pytest
import numpy as np
from airmouse.vision.projection import HandProjector
from airmouse.vision.head_coords import HeadCoordinateSystem
from airmouse.vision.virtual_plane import VirtualDisplayPlane
from airmouse.vision.hand_tracker import Hand, Landmark
from airmouse.vision.face_tracker import Face, FaceLandmark


def create_mock_hand() -> Hand:
    """Create a mock hand with index fingertip at specified position."""
    landmarks = []
    for i in range(21):
        if i == 8:  # Index fingertip
            landmarks.append(Landmark(0.0, 0.0, -0.5, 1.0))  # In front of face (negative Z in camera coords)
        else:
            landmarks.append(Landmark(0.0, 0.0, 0.0, 1.0))
    hand = Hand(landmarks=landmarks, confidence=1.0, handedness="Right")
    return hand


def create_mock_face() -> Face:
    """Create a mock face with valid landmarks for head coordinate system."""
    landmarks = [FaceLandmark(0.0, 0.0, 0.0, 1.0) for _ in range(468)]
    face = Face(landmarks=landmarks, confidence=1.0)
    face._eye_midpoint = FaceLandmark(0.0, 0.0, 0.0, 1.0)
    face._nose_tip = FaceLandmark(0.0, 0.0, -0.1, 1.0)
    face._forehead = FaceLandmark(0.0, -0.1, 0.0, 1.0)
    return face


class TestHandProjector:
    """Tests for HandProjector class."""

    def setup_method(self):
        """Set up common test fixtures."""
        face = create_mock_face()
        self.head_coords = HeadCoordinateSystem.from_face(face)
        self.plane = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25, head_coords=self.head_coords)
        self.projector = HandProjector(
            virtual_plane=self.plane,
            head_coords=self.head_coords,
            use_head_coords_for_ray=True
        )

    def test_projector_creation(self):
        """Test projector creation with default params."""
        head_coords = HeadCoordinateSystem()
        plane = VirtualDisplayPlane()
        projector = HandProjector(
            virtual_plane=plane,
            head_coords=head_coords,
            use_head_coords_for_ray=True
        )
        assert projector.use_head_coords_for_ray is True

    def test_projector_creation_custom(self):
        """Test projector creation with custom params."""
        head_coords = HeadCoordinateSystem()
        plane = VirtualDisplayPlane()
        projector = HandProjector(
            virtual_plane=plane,
            head_coords=head_coords,
            use_head_coords_for_ray=False
        )
        assert projector.use_head_coords_for_ray is False

    def test_project_fingertip_center(self):
        """Test projecting index fingertip at center of plane."""
        hand = create_mock_hand()
        face = create_mock_face()

        result = self.projector.project(hand, face)

        # Should hit center of plane
        assert result.valid is True
        assert abs(result.u - 0.5) < 0.01
        assert abs(result.v - 0.5) < 0.01

    def test_project_fingertip_left_side(self):
        """Test projecting fingertip to left side of plane."""
        hand = create_mock_hand()
        # Override index fingertip position to left side in camera coords
        # We need to transform from head coords to camera coords
        hand.landmarks[8] = Landmark(-0.2, 0.0, -0.5, 1.0)
        face = create_mock_face()

        result = self.projector.project(hand, face)

        # Should be on left side (u < 0.5)
        assert result.valid is True
        assert result.u < 0.5
        assert abs(result.v - 0.5) < 0.01

    def test_project_fingertip_right_side(self):
        """Test projecting fingertip to right side of plane."""
        hand = create_mock_hand()
        hand.landmarks[8] = Landmark(0.2, 0.0, -0.5, 1.0)
        face = create_mock_face()

        result = self.projector.project(hand, face)

        # Should be on right side (u > 0.5)
        assert result.valid is True
        assert result.u > 0.5
        assert abs(result.v - 0.5) < 0.01

    def test_project_fingertip_top(self):
        """Test projecting fingertip to top of plane."""
        hand = create_mock_hand()
        # In camera coords, +Y is down. Top of plane = -Y in camera coords
        hand.landmarks[8] = Landmark(0.0, -0.125, -0.5, 1.0)
        face = create_mock_face()

        result = self.projector.project(hand, face)

        # Should be at top (v > 0.5, closer to 1.0)
        assert result.valid is True
        assert abs(result.u - 0.5) < 0.01
        assert result.v > 0.5

    def test_project_fingertip_bottom(self):
        """Test projecting fingertip to bottom of plane."""
        hand = create_mock_hand()
        # In camera coords, +Y is down. Bottom of plane = +Y in camera coords
        hand.landmarks[8] = Landmark(0.0, 0.125, -0.5, 1.0)
        face = create_mock_face()

        result = self.projector.project(hand, face)

        # Should be at bottom (v < 0.5, closer to 0.0)
        assert result.valid is True
        assert abs(result.u - 0.5) < 0.01
        assert result.v < 0.5

    def test_project_from_landmarks(self):
        """Test projecting from individual landmarks."""
        from airmouse.vision.hand_tracker import Landmark
        from airmouse.vision.face_tracker import FaceLandmark

        index_tip = Landmark(0.0, 0.0, -0.5, 1.0)
        eye_midpoint = FaceLandmark(0.0, 0.0, 0.0, 1.0)

        result = self.projector.project_from_landmarks(index_tip, eye_midpoint)

        assert result.valid is True
        assert abs(result.u - 0.5) < 0.01
        assert abs(result.v - 0.5) < 0.01

    def test_no_head_coords_mode(self):
        """Test projector without head coordinate system (fallback mode)."""
        # Use a valid head coords from mock face
        face = create_mock_face()
        head_coords = HeadCoordinateSystem.from_face(face)
        plane = VirtualDisplayPlane(head_coords=head_coords)
        projector = HandProjector(
            virtual_plane=plane,
            head_coords=head_coords,
            use_head_coords_for_ray=False
        )

        # Use project_from_landmarks mode
        from airmouse.vision.hand_tracker import Landmark
        from airmouse.vision.face_tracker import FaceLandmark

        index_tip = Landmark(0.0, 0.0, -0.5, 1.0)
        eye_midpoint = FaceLandmark(0.0, 0.0, 0.0, 1.0)

        result = projector.project_from_landmarks(index_tip, eye_midpoint, head_coords=head_coords)

        # Should still work
        assert result.valid is True
        assert result.u is not None
        assert result.v is not None

    def test_invalid_hand(self):
        """Test error handling for invalid hand."""
        face = create_mock_face()

        # Invalid hand (no landmarks)
        invalid_hand = Hand(landmarks=[], confidence=0.0, handedness="Right")
        result = self.projector.project(invalid_hand, face)
        assert result.valid is False
        assert "Invalid hand landmarks" in result.error_message

    def test_invalid_face(self):
        """Test error handling for invalid face."""
        hand = create_mock_hand()

        # Invalid face (no eye midpoint)
        invalid_face = Face(landmarks=[], confidence=0.0)
        result = self.projector.project(hand, invalid_face)
        assert result.valid is False
        assert "Invalid face or missing eye midpoint" in result.error_message

    def test_get_stats(self):
        """Test getting projection statistics."""
        hand = create_mock_hand()
        face = create_mock_face()

        self.projector.project(hand, face)
        self.projector.project(hand, face)

        stats = self.projector.get_stats()
        assert stats["total_projections"] == 2
        assert stats["failed_projections"] == 0
        assert stats["success_rate"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])