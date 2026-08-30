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
    # Create 468 landmarks with proper values at key indices
    landmarks = [FaceLandmark(0.0, 0.0, 0.0, 1.0) for _ in range(468)]

    # Set key landmarks for head coordinate system
    # LEFT_EYE_INNER = 133, LEFT_EYE_OUTER = 33
    landmarks[133] = FaceLandmark(-0.03, 0.0, -0.1, 1.0)  # Left eye inner
    landmarks[33] = FaceLandmark(-0.06, 0.0, -0.1, 1.0)   # Left eye outer
    # RIGHT_EYE_INNER = 362, RIGHT_EYE_OUTER = 263
    landmarks[362] = FaceLandmark(0.03, 0.0, -0.1, 1.0)   # Right eye inner
    landmarks[263] = FaceLandmark(0.06, 0.0, -0.1, 1.0)   # Right eye outer
    # NOSE_TIP = 1
    landmarks[1] = FaceLandmark(0.0, 0.0, -0.2, 1.0)      # Nose tip (forward)
    # FOREHEAD = 10
    landmarks[10] = FaceLandmark(0.0, -0.1, -0.1, 1.0)    # Forehead (up)

    face = Face(landmarks=landmarks, confidence=1.0)
    return face


def create_mock_face_rotated_yaw(yaw_deg: float) -> Face:
    """Create a mock face rotated by yaw degrees around Y axis."""
    landmarks = [FaceLandmark(0.0, 0.0, 0.0, 1.0) for _ in range(468)]

    yaw_rad = np.deg2rad(yaw_deg)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)

    # Eye midpoint at camera origin (0, 0, -0.1) - rotation center
    eye_mid_x, eye_mid_y, eye_mid_z = 0.0, 0.0, -0.1

    # Eye centers (before rotation): left at (-0.045, 0, 0) relative to eye midpoint, right at (0.045, 0, 0)
    # After yaw rotation around eye midpoint (origin)
    left_eye_rel_x = -0.045 * cos_y
    left_eye_rel_z = -0.045 * sin_y
    right_eye_rel_x = 0.045 * cos_y
    right_eye_rel_z = 0.045 * sin_y

    # LEFT_EYE_INNER = 133, LEFT_EYE_OUTER = 33 (relative to eye center: ±0.015 in X)
    landmarks[133] = FaceLandmark(eye_mid_x + left_eye_rel_x - 0.015, eye_mid_y, eye_mid_z + left_eye_rel_z, 1.0)
    landmarks[33] = FaceLandmark(eye_mid_x + left_eye_rel_x + 0.015, eye_mid_y, eye_mid_z + left_eye_rel_z, 1.0)
    # RIGHT_EYE_INNER = 362, RIGHT_EYE_OUTER = 263
    landmarks[362] = FaceLandmark(eye_mid_x + right_eye_rel_x - 0.015, eye_mid_y, eye_mid_z + right_eye_rel_z, 1.0)
    landmarks[263] = FaceLandmark(eye_mid_x + right_eye_rel_x + 0.015, eye_mid_y, eye_mid_z + right_eye_rel_z, 1.0)

    # NOSE_TIP = 1 - forward vector rotates
    # Original offset from eye midpoint: (0, 0, -0.1) - 10cm forward (negative camera Z)
    # After yaw: x' = 0*cos - (-0.1)*sin = 0.1*sin, z' = 0*sin + (-0.1)*cos = -0.1*cos
    nose_rel_x = 0.1 * sin_y
    nose_rel_z = -0.1 * cos_y
    landmarks[1] = FaceLandmark(eye_mid_x + nose_rel_x, eye_mid_y, eye_mid_z + nose_rel_z, 1.0)

    # FOREHEAD = 10 - up vector (0, -0.1, 0) relative to eye midpoint - doesn't change with yaw
    landmarks[10] = FaceLandmark(eye_mid_x, eye_mid_y - 0.1, eye_mid_z, 1.0)

    face = Face(landmarks=landmarks, confidence=1.0)
    return face


def create_mock_face_rotated_pitch(pitch_deg: float) -> Face:
    """Create a mock face rotated by pitch degrees around X axis."""
    landmarks = [FaceLandmark(0.0, 0.0, 0.0, 1.0) for _ in range(468)]

    pitch_rad = np.deg2rad(pitch_deg)
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)

    # Eye midpoint at camera origin (0, 0, -0.1) - rotation center
    eye_mid_x, eye_mid_y, eye_mid_z = 0.0, 0.0, -0.1

    # Eye centers (before rotation): left at (-0.045, 0, 0) relative to eye midpoint, right at (0.045, 0, 0)
    # After pitch rotation around eye midpoint: y' = y*cos - z*sin, z' = y*sin + z*cos
    # Eyes are at z=0 relative to eye midpoint, so they only move in Y
    left_eye_rel_y = 0.0 * cos_p - 0.0 * sin_p  # 0
    left_eye_rel_z = 0.0 * sin_p + 0.0 * cos_p  # 0
    right_eye_rel_y = left_eye_rel_y
    right_eye_rel_z = left_eye_rel_z

    # LEFT_EYE_INNER = 133, LEFT_EYE_OUTER = 33 (relative to eye center: ±0.015 in X)
    landmarks[133] = FaceLandmark(eye_mid_x - 0.06, eye_mid_y + left_eye_rel_y, eye_mid_z + left_eye_rel_z, 1.0)
    landmarks[33] = FaceLandmark(eye_mid_x - 0.03, eye_mid_y + left_eye_rel_y, eye_mid_z + left_eye_rel_z, 1.0)
    # RIGHT_EYE_INNER = 362, RIGHT_EYE_OUTER = 263
    landmarks[362] = FaceLandmark(eye_mid_x + 0.03, eye_mid_y + right_eye_rel_y, eye_mid_z + right_eye_rel_z, 1.0)
    landmarks[263] = FaceLandmark(eye_mid_x + 0.06, eye_mid_y + right_eye_rel_y, eye_mid_z + right_eye_rel_z, 1.0)

    # NOSE_TIP = 1 - forward vector (0, 0, -0.1) relative to eye midpoint rotates
    # After pitch: y' = 0*cos - (-0.1)*sin = 0.1*sin, z' = 0*sin + (-0.1)*cos = -0.1*cos
    nose_rel_y = 0.1 * sin_p
    nose_rel_z = -0.1 * cos_p
    landmarks[1] = FaceLandmark(eye_mid_x, eye_mid_y + nose_rel_y, eye_mid_z + nose_rel_z, 1.0)

    # FOREHEAD = 10 - up vector (0, -0.1, 0) relative to eye midpoint rotates
    # After pitch: y' = -0.1*cos, z' = -0.1*sin
    forehead_rel_y = -0.1 * cos_p
    forehead_rel_z = -0.1 * sin_p
    landmarks[10] = FaceLandmark(eye_mid_x, eye_mid_y + forehead_rel_y, eye_mid_z + forehead_rel_z, 1.0)

    face = Face(landmarks=landmarks, confidence=1.0)
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

    def test_head_movement_invariance_center(self):
        """Test head-movement invariance: head rotates but hand fixed relative to head -> cursor stable."""
        # Create initial face and hand with index finger pointing at center of virtual plane
        face = create_mock_face()
        hand = create_mock_hand()
        hand.landmarks[8] = Landmark(0.0, 0.0, -0.5, 1.0)  # Center in camera coords

        # Project initial position
        result1 = self.projector.project(hand, face)
        assert result1.valid is True
        u1, v1 = result1.u, result1.v

        # Now simulate head rotated 15 degrees right (yaw)
        # The hand moves WITH the head in camera coords (fixed relative to head)
        # In head coords, hand position should remain the same
        #
        # Initial hand in camera: (0, 0, -0.5), eye midpoint: (0, 0, -0.1)
        # Hand in head coords = (0, 0, 0.4) - 40cm forward from eye
        # After yaw rotation, transform fixed head-coords position to camera coords:
        # hand_cam = R @ hand_head + eye_midpoint
        # R @ [0, 0, 0.4] = 0.4 * forward_vector
        # For 15° yaw: forward = [sin_y, 0, -cos_y] in camera coords
        # hand_cam = [0.4*sin_y, 0, -0.4*cos_y] + [0, 0, -0.1] = [0.4*sin_y, 0, -0.4*cos_y - 0.1]
        yaw_rad = np.deg2rad(15)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)

        hand_moved = Hand(
            landmarks=[
                Landmark(0.4 * sin_y, 0.0, -0.4 * cos_y - 0.1, 1.0) if i == 8 else Landmark(0.0, 0.0, 0.0, 1.0)
                for i in range(21)
            ],
            confidence=1.0,
            handedness="Right"
        )

        # Face rotated by same amount
        face_rotated = create_mock_face_rotated_yaw(15)

        # Recreate head_coords from rotated face
        head_coords_rotated = HeadCoordinateSystem.from_face(face_rotated)
        plane_rotated = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25, head_coords=head_coords_rotated)
        projector_rotated = HandProjector(
            virtual_plane=plane_rotated,
            head_coords=head_coords_rotated,
            use_head_coords_for_ray=True
        )

        result2 = projector_rotated.project(hand_moved, face_rotated)
        assert result2.valid is True
        u2, v2 = result2.u, result2.v

        # Cursor position should be nearly identical (head-relative invariance)
        assert abs(u1 - u2) < 0.02, f"Head movement invariance failed: u changed from {u1:.3f} to {u2:.3f}"
        assert abs(v1 - v2) < 0.02, f"Head movement invariance failed: v changed from {v1:.3f} to {v2:.3f}"

    def test_head_movement_invariance_edges(self):
        """Test head-movement invariance at edges of virtual plane."""
        face = create_mock_face()
        # Hand at top-left corner of plane in head coords
        # Plane is 0.4m wide x 0.25m high, at 0.3m distance
        # Corners in head coords: (±0.2, ±0.125, -0.3)
        hand = Hand(
            landmarks=[
                Landmark(-0.2, 0.125, -0.3, 1.0) if i == 8 else Landmark(0.0, 0.0, 0.0, 1.0)
                for i in range(21)
            ],
            confidence=1.0,
            handedness="Right"
        )

        result1 = self.projector.project(hand, face)
        assert result1.valid is True
        u1, v1 = result1.u, result1.v

        # Rotate head 10 degrees down (pitch)
        pitch_rad = np.deg2rad(10)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)

        # Hand moves with head in camera coords
        # Original: (-0.2, 0.125, -0.3) in head coords
        # After pitch: x' = x, y' = y*cos - z*sin, z' = y*sin + z*cos
        y_new = 0.125 * cos_p - (-0.3) * sin_p
        z_new = 0.125 * sin_p + (-0.3) * cos_p
        hand_moved = Hand(
            landmarks=[
                Landmark(-0.2, y_new, z_new, 1.0) if i == 8 else Landmark(0.0, 0.0, 0.0, 1.0)
                for i in range(21)
            ],
            confidence=1.0,
            handedness="Right"
        )

        # Face rotated by same pitch
        face_rotated = create_mock_face_rotated_pitch(10)

        head_coords_rotated = HeadCoordinateSystem.from_face(face_rotated)
        plane_rotated = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25, head_coords=head_coords_rotated)
        projector_rotated = HandProjector(
            virtual_plane=plane_rotated,
            head_coords=head_coords_rotated,
            use_head_coords_for_ray=True
        )

        result2 = projector_rotated.project(hand_moved, face_rotated)
        assert result2.valid is True
        u2, v2 = result2.u, result2.v

        # Should be at same normalized position
        assert abs(u1 - u2) < 0.02, f"Head movement invariance failed: u changed from {u1:.3f} to {u2:.3f}"
        assert abs(v1 - v2) < 0.02, f"Head movement invariance failed: v changed from {v1:.3f} to {v2:.3f}"

    def test_projection_only_mode_synthetic(self):
        """Test projection-only mode: bypass camera, feed synthetic landmarks, verify u,v coordinates."""
        # This tests the projection pipeline in isolation without camera dependency
        face = create_mock_face()

        # In head coords (no rotation): eye at (0,0,0), forward=+Z, right=+X, up=+Y
        # Plane at z = distance = 0.3 in head coords
        # Mock face has eye_midpoint at camera (0, 0, -0.1), forward_camera = (0, 0, -1)
        # Head->Camera: p_cam = eye_mid + x*right + y*up + z*forward
        # right=(1,0,0), up=(0,-1,0), forward=(0,0,-1) in camera coords
        # p_cam = (x, -y, -0.1 - z)

        test_positions = [
            # (head_x, head_y, head_z, expected_u, expected_v, description)
            (0.0, 0.0, 0.3, 0.5, 0.5, "center"),
            (-0.2, 0.0, 0.3, 0.0, 0.5, "left edge"),
            (0.2, 0.0, 0.3, 1.0, 0.5, "right edge"),
            (0.0, 0.125, 0.3, 0.5, 1.0, "top edge"),
            (0.0, -0.125, 0.3, 0.5, 0.0, "bottom edge"),
            (-0.1, 0.0625, 0.3, 0.25, 0.75, "quarter positions"),
            (0.1, -0.0625, 0.3, 0.75, 0.25, "three-quarter positions"),
        ]

        for head_x, head_y, head_z, expected_u, expected_v, desc in test_positions:
            # Convert head coords to camera coords (no head rotation)
            # eye_midpoint = (0, 0, -0.1), right=(1,0,0), up=(0,-1,0), forward=(0,0,-1)
            cam_x = head_x
            cam_y = -head_y  # head +Y = up, camera +Y = down
            cam_z = -0.1 - head_z  # head +Z = forward (-Z in camera)

            hand = Hand(
                landmarks=[
                    Landmark(cam_x, cam_y, cam_z, 1.0) if i == 8 else Landmark(0.0, 0.0, 0.0, 1.0)
                    for i in range(21)
                ],
                confidence=1.0,
                handedness="Right"
            )

            result = self.projector.project(hand, face)
            assert result.valid is True, f"Projection failed for {desc}"
            assert abs(result.u - expected_u) < 0.02, f"{desc}: u={result.u:.3f} != {expected_u:.3f}"
            assert abs(result.v - expected_v) < 0.02, f"{desc}: v={result.v:.3f} != {expected_v:.3f}"

    def test_two_hand_precision_mode_projection(self):
        """Test two-hand precision mode: secondary hand enables precision tracking."""
        face = create_mock_face()

        # Primary hand (right) controls cursor
        primary_hand = Hand(
            landmarks=[
                Landmark(0.0, 0.0, -0.5, 1.0) if i == 8 else Landmark(0.0, 0.0, 0.0, 1.0)
                for i in range(21)
            ],
            confidence=1.0,
            handedness="Right"
        )

        # Secondary hand (left) present - enables precision mode
        secondary_hand = Hand(
            landmarks=[
                Landmark(0.0, 0.0, -0.5, 1.0) if i == 8 else Landmark(0.0, 0.0, 0.0, 1.0)
                for i in range(21)
            ],
            confidence=1.0,
            handedness="Left"
        )

        # Test with two hands
        result_primary = self.projector.project(primary_hand, face)
        assert result_primary.valid is True

        # Simulate secondary hand detection (this would be tracked separately in TrackingProcessor)
        # The precision mode is handled in TrackingProcessor/GestureRecognizer
        # Here we just verify projection works for both hands
        result_secondary = self.projector.project(secondary_hand, face)
        assert result_secondary.valid is True

        # Both hands should project to same position when at same relative position
        assert abs(result_primary.u - result_secondary.u) < 0.01
        assert abs(result_primary.v - result_secondary.v) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])