"""
Synthetic Cursor Test Mode for Air Mouse.

Feeds synthetic cursor targets through the input backend without
requiring the camera.  This validates the projection-to-cursor
pipeline end-to-end: projection produces (u, v) → cursor controller
maps to screen coordinates → input backend delivers the event.
"""
import sys
sys.path.insert(0, '/home/shubham/airmouse')

import pytest
import numpy as np
import math
from unittest.mock import MagicMock, patch
from airmouse.vision.projection import HandProjector, ProjectionResult
from airmouse.vision.virtual_plane import VirtualDisplayPlane
from airmouse.vision.head_coords import HeadCoordinateSystem
from airmouse.vision.hand_tracker import Hand, Landmark
from airmouse.vision.face_tracker import Face, FaceLandmark


def create_mock_face() -> Face:
    """Create a mock face with valid landmarks for head coordinate system."""
    landmarks = [FaceLandmark(0.0, 0.0, 0.0, 1.0) for _ in range(468)]
    face = Face(landmarks=landmarks, confidence=1.0)
    face._eye_midpoint = FaceLandmark(0.0, 0.0, 0.0, 1.0)
    face._nose_tip = FaceLandmark(0.0, 0.0, -0.1, 1.0)
    face._forehead = FaceLandmark(0.0, -0.1, 0.0, 1.0)
    return face


def create_mock_hand_at_position(x: float, y: float, z: float) -> Hand:
    """Create a mock hand with index fingertip at given camera coordinates."""
    landmarks = [Landmark(0.0, 0.0, 0.0, 1.0) for _ in range(21)]
    landmarks[8] = Landmark(x, y, z, 1.0)  # INDEX_TIP
    return Hand(landmarks=landmarks, confidence=1.0, handedness="Right")


class TestSyntheticCursor:
    """Test synthetic cursor delivery through the input backend."""

    def setup_method(self):
        face = create_mock_face()
        self.head_coords = HeadCoordinateSystem.from_face(face)
        self.plane = VirtualDisplayPlane(
            distance=0.30, width=0.70, height=0.50,
            head_coords=self.head_coords
        )
        self.projector = HandProjector(
            virtual_plane=self.plane,
            head_coords=self.head_coords,
            use_head_coords_for_ray=True
        )

    def test_synthetic_cursor_center(self):
        """Test synthetic cursor at screen center via projection."""
        # Fingertip at center: (0, 0, -0.5) in camera coords
        # Eye midpoint at origin (0, 0, 0)
        # In head coords: fingertip = camera_to_head(0, 0, -0.5) = (-0, 0, 0.4)
        # Plane center is at (0, 0, 0.3) in head coords
        hand = create_mock_hand_at_position(0.0, 0.0, -0.5)
        face = create_mock_face()

        result = self.projector.project(hand, face)

        assert result.valid is True
        assert abs(result.u - 0.5) < 0.01
        assert abs(result.v - 0.5) < 0.01

    def test_synthetic_cursor_left(self):
        """Test synthetic cursor at left side of screen."""
        # Fingertip to the left: x=-0.15 in camera coords
        hand = create_mock_hand_at_position(-0.15, 0.0, -0.5)
        face = create_mock_face()

        result = self.projector.project(hand, face)

        assert result.valid is True
        assert result.u < 0.5

    def test_synthetic_cursor_right(self):
        """Test synthetic cursor at right side of screen."""
        hand = create_mock_hand_at_position(0.15, 0.0, -0.5)
        face = create_mock_face()

        result = self.projector.project(hand, face)

        assert result.valid is True
        assert result.u > 0.5

    def test_synthetic_cursor_top(self):
        """Test synthetic cursor at top of screen."""
        # In camera coords, +Y is down, so top = -Y
        hand = create_mock_hand_at_position(0.0, -0.15, -0.5)
        face = create_mock_face()

        result = self.projector.project(hand, face)

        assert result.valid is True
        assert result.v < 0.5  # Top of screen → v < 0.5

    def test_synthetic_cursor_bottom(self):
        """Test synthetic cursor at bottom of screen."""
        hand = create_mock_hand_at_position(0.0, 0.15, -0.5)
        face = create_mock_face()

        result = self.projector.project(hand, face)

        assert result.valid is True
        assert result.v > 0.5  # Bottom of screen → v > 0.5

    def test_input_backend_accepts_cursor_target(self):
        """
        Test that the input backend's move_absolute correctly receives
        synthetic cursor targets derived from projection results.

        This simulates the full pipeline: projection (u,v) → screen (x,y)
        → input backend move_absolute(x, y).
        """
        from airmouse.input.linux_input import UInputBackend

        # Create a mock backend (we can't actually use /dev/uinput in tests)
        backend = UInputBackend()
        # Mock the virtual mouse
        backend._virtual_mouse = MagicMock()
        backend._virtual_mouse.is_created.return_value = True
        backend._virtual_mouse.get_position.return_value = (100, 100)
        backend._initialized = True

        # Project fingertip to center
        hand = create_mock_hand_at_position(0.0, 0.0, -0.5)
        face = create_mock_face()
        result = self.projector.project(hand, face)
        assert result.valid

        # Simulate cursor controller: map u,v to screen coords (1920x1080)
        screen_w, screen_h = 1920, 1080
        # Raw u/v without clamping (our policy). If out of bounds, the
        # cursor controller clamps at the edge per §9.
        x = int(np.clip(result.u * screen_w, 0, screen_w - 1))
        y = int(np.clip(result.v * screen_h, 0, screen_h - 1))

        # Deliver to backend
        success = backend.move_absolute(x, y)
        assert success is True

        # Verify the virtual mouse received the correct relative movement
        expected_dx = x - 100  # 960 - 100 = 860
        expected_dy = y - 100  # 540 - 100 = 440
        backend._virtual_mouse.move.assert_called_with(expected_dx, expected_dy)

    def test_out_of_bounds_projection_does_not_crash(self):
        """
        Test that projecting with the hand very far from center
        (beyond the 0.70x0.50 plane) returns valid raw u/v values
        that are simply out of [0,1] — the caller's responsibility
        to clamp.
        """
        # Fingertip far to the right and up
        # In head coords this should produce u > 1.0 and v < 0.0
        hand = create_mock_hand_at_position(0.50, -0.50, -0.5)
        face = create_mock_face()

        result = self.projector.project(hand, face)
        assert result.valid is True

        # u should exceed 1.0 (hand is beyond right edge of plane)
        assert result.u > 1.0 or abs(result.u - 1.0) < 0.01
        # v should be below 0 (hand is beyond top edge of plane)
        assert result.v < 0.0 or abs(result.v) < 0.01

    def test_projection_debug_logging(self):
        """Test that projection debug mode can be enabled and emits events."""
        self.projector.set_debug(True)

        hand = create_mock_hand_at_position(0.0, 0.0, -0.5)
        face = create_mock_face()
        result = self.projector.project(hand, face)
        assert result.valid

        # Check that debug events were recorded
        events = self.projector.diagnostics.get_events()
        assert len(events) > 0
        # Should contain trace-level projection_debug events
        debug_events = [e for e in events if e.get("event") == "projection_debug"]
        assert len(debug_events) > 0

        self.projector.set_debug(False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])