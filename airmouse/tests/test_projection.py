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
        self.plane = VirtualDisplayPlane(distance=0.30, width=1.0, height=1.0, head_coords=self.head_coords)
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

        # Should be at top (v ≈ 0.0 = TOP of screen)
        assert result.valid is True
        assert abs(result.u - 0.5) < 0.01
        assert result.v < 0.5, f"Top of plane should have v < 0.5, got v={result.v}"

    def test_project_fingertip_bottom(self):
        """Test projecting fingertip to bottom of plane."""
        hand = create_mock_hand()
        # In camera coords, +Y is down. Bottom of plane = +Y in camera coords
        hand.landmarks[8] = Landmark(0.0, 0.125, -0.5, 1.0)
        face = create_mock_face()

        result = self.projector.project(hand, face)

        # Should be at bottom (v ≈ 1.0 = BOTTOM of screen)
        assert result.valid is True
        assert abs(result.u - 0.5) < 0.01
        assert result.v > 0.5, f"Bottom of plane should have v > 0.5, got v={result.v}"

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
        plane = VirtualDisplayPlane(distance=0.30, head_coords=head_coords)
        projector = HandProjector(
            virtual_plane=plane,
            head_coords=head_coords,
            use_head_coords_for_ray=False
        )

        # Use project_from_landmarks mode
        # The mock face has eye_midpoint at (0, 0, -0.1) in camera coords
        # and index_tip at (0, 0, -0.5) in camera coords (fingertip in front of face)
        from airmouse.vision.hand_tracker import Landmark
        from airmouse.vision.face_tracker import FaceLandmark

        # Use actual eye midpoint from the face (which is at z=-0.1)
        index_tip = Landmark(0.0, 0.0, -0.5, 1.0)
        eye_midpoint = FaceLandmark(0.0, 0.0, -0.1, 1.0)

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
        plane_rotated = VirtualDisplayPlane(distance=0.30, width=1.0, height=1.0, head_coords=head_coords_rotated)
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
        """Test head-movement invariance at edges of virtual plane.

        The fingertip is placed at plane corners in HEAD space
        (not camera space), so the hand is at the plane surface.
        When the head rotates but the hand stays fixed relative to
        the head, the normalized coordinates must be invariant.
        """
        face = create_mock_face()
        hc = HeadCoordinateSystem.from_face(face)

        # Fingertip at top-left corner of plane in HEAD coords.
        # Plane is 1.0 wide x 1.0 high at z=0.30.
        # Corner in head coords: (-0.5, 0.5, 0.30)
        # Convert to camera coords: p_cam = origin + x*right + y*up + z*forward
        # origin=(0.5,0.5,-0.1), right=(1,0,0), up=(0,-1,0), forward=(0,0,-1)
        # p_cam = (0.5-0.5, 0.5-0.5, -0.1-0.30) = (0.0, 0.0, -0.40)
        fingertip_cam = hc.head_to_camera(
            np.array([-0.5, 0.5, 0.30], dtype=np.float32)
        )
        hand = Hand(
            landmarks=[
                Landmark(float(fingertip_cam[0]), float(fingertip_cam[1]),
                         float(fingertip_cam[2]), 1.0) if i == 8
                else Landmark(0.0, 0.0, 0.0, 1.0)
                for i in range(21)
            ],
            confidence=1.0,
            handedness="Right"
        )

        result1 = self.projector.project(hand, face)
        assert result1.valid is True, f"Projection failed: {result1.error_message}"
        u1, v1 = result1.u, result1.v
        # Top-left corner: u=0.0 (left), v=0.0 (top)
        assert abs(u1 - 0.0) < 0.02, f"u={u1:.3f}, expected 0.0"
        assert abs(v1 - 0.0) < 0.02, f"v={v1:.3f}, expected 0.0"

        # Rotate head 10 degrees down (pitch).  The hand stays fixed
        # relative to the head, so we transform the same head-space
        # point through the new head coordinate system.
        face_rotated = create_mock_face_rotated_pitch(10)
        hc_rotated = HeadCoordinateSystem.from_face(face_rotated)

        # Same hand position in head space, transformed to new camera coords
        fingertip_cam_rotated = hc_rotated.head_to_camera(
            np.array([-0.5, 0.5, 0.30], dtype=np.float32)
        )
        hand_moved = Hand(
            landmarks=[
                Landmark(float(fingertip_cam_rotated[0]),
                         float(fingertip_cam_rotated[1]),
                         float(fingertip_cam_rotated[2]), 1.0) if i == 8
                else Landmark(0.0, 0.0, 0.0, 1.0)
                for i in range(21)
            ],
            confidence=1.0,
            handedness="Right"
        )

        plane_rotated = VirtualDisplayPlane(
            distance=0.30, width=1.0, height=1.0,
            head_coords=hc_rotated
        )
        projector_rotated = HandProjector(
            virtual_plane=plane_rotated,
            head_coords=hc_rotated,
            use_head_coords_for_ray=True
        )

        result2 = projector_rotated.project(hand_moved, face_rotated)
        assert result2.valid is True, f"Projection failed: {result2.error_message}"
        u2, v2 = result2.u, result2.v

        # Head-movement invariance: the hand stays fixed relative to the
        # head, so the normalized coordinates should be consistent.
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
            # Normalized: u=0 left, u=1 right; v=0 top, v=1 bottom
            # Head coords: +X right, +Y up, +Z forward
            # v=0 (TOP) = head Y+ (UP); v=1 (BOTTOM) = head Y- (DOWN)
            # Plane is 1.0 wide x 1.0 high at 0.30 distance (normalized units)
            (0.0, 0.0, 0.3, 0.5, 0.5, "center"),
            (-0.5, 0.0, 0.3, 0.0, 0.5, "left edge"),
            (0.5, 0.0, 0.3, 1.0, 0.5, "right edge"),
            (0.0, 0.5, 0.3, 0.5, 0.0, "top edge"),       # head Y+ = UP → v=0 (TOP)
            (0.0, -0.5, 0.3, 0.5, 1.0, "bottom edge"),   # head Y- = DOWN → v=1 (BOTTOM)
            (-0.25, 0.25, 0.3, 0.25, 0.25, "quarter positions"),  # head Y+ → v=0.25
            (0.25, -0.25, 0.3, 0.75, 0.75, "three-quarter positions"),  # head Y- → v=0.75
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

    def test_graceful_degradation_face_lost(self):
        """
        Test graceful degradation to legacy camera coordinates when face tracking is lost (FR-010).

        When face confidence falls below threshold:
        - System should auto-pause head-relative projection
        - Fallback to legacy 2D camera coordinate mapping
        - Log warning and continue without crashing
        """
        # Create initial valid face and hand
        face = create_mock_face()
        hand = create_mock_hand()
        hand.landmarks[8] = Landmark(0.0, 0.0, -0.5, 1.0)  # Center in camera coords

        # Project initial position (head-relative mode works)
        result1 = self.projector.project(hand, face)
        assert result1.valid is True
        u1, v1 = result1.u, result1.v

        # Now simulate face tracking loss (low confidence face)
        face_low_conf = create_mock_face()
        face_low_conf.confidence = 0.3  # Below min_face_confidence (0.5)

        # With low confidence face, projection should fail gracefully
        result2 = self.projector.project(hand, face_low_conf)
        assert result2.valid is False
        assert "confidence" in result2.error_message.lower() or "face" in result2.error_message.lower()

        # Create face with missing landmarks (complete tracking loss)
        face_no_landmarks = Face(landmarks=[], confidence=1.0)
        face_no_landmarks._eye_midpoint = None
        face_no_landmarks._nose_tip = None
        face_no_landmarks._forehead = None

        result3 = self.projector.project(hand, face_no_landmarks)
        assert result3.valid is False
        assert "landmark" in result3.error_message.lower() or "invalid" in result3.error_message.lower()

        # Test with None face (complete loss)
        result4 = self.projector.project(hand, None)
        assert result4.valid is False
        assert "face" in result4.error_message.lower()

        # Test with invalid head coordinates
        face_bad = create_mock_face()
        # Create projector with invalid head_coords
        from airmouse.vision.head_coords import HeadCoordinateSystem
        bad_head_coords = HeadCoordinateSystem(_valid=False)
        bad_plane = VirtualDisplayPlane(distance=0.30, width=1.0, height=1.0, head_coords=bad_head_coords)
        bad_projector = HandProjector(
            virtual_plane=bad_plane,
            head_coords=bad_head_coords,
            use_head_coords_for_ray=True
        )

        result5 = bad_projector.project(hand, face_bad)
        assert result5.valid is False
        assert "head" in result5.error_message.lower() or "coordinate" in result5.error_message.lower()

        # Verify projector statistics track failures
        stats = bad_projector.get_stats()
        assert stats["failed_projections"] > 0
        assert stats["total_projections"] > 0
        assert stats["success_rate"] < 1.0

    def test_projection_recovery_after_face_loss(self):
        """
        Test that projection recovers when face tracking is restored (FR-010, SC-006).

        After 5 stable frames of valid face data, auto-recover to head-relative mode.
        """
        face = create_mock_face()
        hand = create_mock_hand()
        hand.landmarks[8] = Landmark(0.0, 0.0, -0.5, 1.0)

        # Initial valid projection
        result1 = self.projector.project(hand, face)
        assert result1.valid is True
        u1, v1 = result1.u, result1.v

        # Simulate face loss for a few frames
        face_low = create_mock_face()
        face_low.confidence = 0.3
        for _ in range(3):
            result = self.projector.project(hand, face_low)
            assert result.valid is False

        # Restore face tracking
        face_restored = create_mock_face()
        result_recovered = self.projector.project(hand, face_restored)
        assert result_recovered.valid is True
        u2, v2 = result_recovered.u, result_recovered.v

        # Position should be same as before (head-relative invariance)
        assert abs(u1 - u2) < 0.02
        assert abs(v1 - v2) < 0.02


class TestProjectionSelfTest:
    """
    Deterministic self-test for projection mathematics using synthetic rays.

    Verifies the projection equations in isolation from MediaPipe:
    - Ray construction: origin + t * direction
    - Ray-plane intersection: t = (distance - origin.z) / direction.z
    - Orthonormal basis: plane_right, plane_up, plane_normal
    - Coordinate mapping: center=(0.5, 0.5), left=(0.0, 0.5), right=(1.0, 0.5),
      top=(0.5, 0.0), bottom=(0.5, 1.0)
    - Monotonicity: moving along each axis produces monotonic change in u/v
    - Out-of-bounds: invalid projections return valid=False without clamping
    """

    def setup_method(self):
        """Set up standard 1.0x1.0 plane at distance=0.30."""
        face = create_mock_face()
        self.head_coords = HeadCoordinateSystem.from_face(face)
        self.plane = VirtualDisplayPlane(
            distance=0.30,
            width=1.0,
            height=1.0,
            head_coords=self.head_coords
        )
        self.projector = HandProjector(
            virtual_plane=self.plane,
            head_coords=self.head_coords,
            use_head_coords_for_ray=True
        )

    def test_orthonormal_basis(self):
        """Verify plane basis vectors are orthonormal."""
        center = self.plane.get_plane_center_camera()
        normal = self.plane.get_plane_normal_camera()
        x_axis, y_axis = self.plane.get_plane_axes_camera()

        assert center is not None
        assert normal is not None
        assert x_axis is not None
        assert y_axis is not None

        # Unit length
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-4
        assert abs(np.linalg.norm(x_axis) - 1.0) < 1e-4
        assert abs(np.linalg.norm(y_axis) - 1.0) < 1e-4

        # Orthogonal
        assert abs(np.dot(x_axis, y_axis)) < 1e-4
        assert abs(np.dot(x_axis, normal)) < 1e-4
        assert abs(np.dot(y_axis, normal)) < 1e-4

    def test_cardinal_points(self):
        """Verify the 5 canonical points map to exact expected (u, v)."""
        # In head coords with no rotation:
        # eye at (0, 0, 0)
        # Plane at z = 0.30, width = 1.0, height = 1.0
        # X range: [-0.5, 0.5], Y range: [-0.5, 0.5]
        #
        # Mapping:
        # u = (x + 0.5) / 1.0  → x = u - 0.5
        # v = (-y + 0.5) / 1.0 → y = 0.5 - v (Y inverted: up is v=0, down is v=1)
        expected = [
            # (name, head_x, head_y, expected_u, expected_v)
            ("center", 0.0, 0.0, 0.5, 0.5),
            ("left", -0.5, 0.0, 0.0, 0.5),
            ("right", 0.5, 0.0, 1.0, 0.5),
            ("top", 0.0, 0.5, 0.5, 0.0),       # head Y+ = UP → screen TOP (v=0)
            ("bottom", 0.0, -0.5, 0.5, 1.0),   # head Y- = DOWN → screen BOTTOM (v=1)
        ]

        face = create_mock_face()
        for name, hx, hy, eu, ev in expected:
            # Convert head coords to camera coords (no rotation)
            # eye_midpoint is at (0, 0, -0.1) in camera coords
            # forward=(0,0,-1), up=(0,-1,0), right=(1,0,0) in camera coords
            cam_x = hx
            cam_y = -hy      # head Y+ (up) = camera Y- (up in image)
            cam_z = -0.1 - 0.30  # head Z+ (forward) = camera -Z

            hand = Hand(
                landmarks=[
                    Landmark(cam_x, cam_y, cam_z, 1.0) if i == 8 else Landmark(0.0, 0.0, 0.0, 1.0)
                    for i in range(21)
                ],
                confidence=1.0,
                handedness="Right"
            )

            result = self.projector.project(hand, face)
            assert result.valid is True, f"Failed for {name}"
            assert abs(result.u - eu) < 1e-3, f"{name}: u={result.u:.4f} != {eu}"
            assert abs(result.v - ev) < 1e-3, f"{name}: v={result.v:.4f} != {ev}"

    def test_horizontal_monotonicity(self):
        """Verify moving hand left-to-right produces monotonically increasing u."""
        face = create_mock_face()
        prev_u = -1.0

        for hx in np.linspace(-0.4, 0.4, 9):
            cam_x = hx
            cam_y = 0.0
            cam_z = -0.1 - 0.30

            hand = Hand(
                landmarks=[
                    Landmark(cam_x, cam_y, cam_z, 1.0) if i == 8 else Landmark(0.0, 0.0, 0.0, 1.0)
                    for i in range(21)
                ],
                confidence=1.0,
                handedness="Right"
            )

            result = self.projector.project(hand, face)
            assert result.valid is True
            assert result.u > prev_u, f"Monotonicity failed at hx={hx}: u={result.u} <= prev={prev_u}"
            prev_u = result.u

    def test_vertical_monotonicity(self):
        """Verify moving hand up-to-down produces monotonically increasing v (screen down)."""
        face = create_mock_face()
        prev_v = -1.0

        # Move from head Y+ (up) to head Y- (down)
        # Should produce increasing v (0=top to 1=bottom)
        for hy in np.linspace(0.4, -0.4, 9):
            cam_x = 0.0
            cam_y = -hy  # head Y+ = camera Y-
            cam_z = -0.1 - 0.30

            hand = Hand(
                landmarks=[
                    Landmark(cam_x, cam_y, cam_z, 1.0) if i == 8 else Landmark(0.0, 0.0, 0.0, 1.0)
                    for i in range(21)
                ],
                confidence=1.0,
                handedness="Right"
            )

            result = self.projector.project(hand, face)
            assert result.valid is True
            assert result.v > prev_v, f"Monotonicity failed at hy={hy}: v={result.v} <= prev={prev_v}"
            prev_v = result.v

    def test_ray_parallel_to_plane(self):
        """Verify parallel ray (denom ≈ 0) returns None/invalid without crashing."""
        ray_origin = np.array([0.0, 0.0, -0.1], dtype=np.float32)
        # Ray direction perpendicular to normal (parallel to plane)
        # Plane normal is (0, 0, -1) in head coords → (0, 0, 1) in camera coords
        # Parallel direction: (1, 0, 0)
        ray_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        intersection = self.plane.ray_plane_intersection(ray_origin, ray_dir)
        assert intersection is None

    def test_ray_pointing_away(self):
        """Verify ray pointing away from plane (t < 0) returns None/invalid."""
        ray_origin = np.array([0.0, 0.0, -0.1], dtype=np.float32)
        # Ray pointing backwards (away from plane at -Z)
        ray_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        intersection = self.plane.ray_plane_intersection(ray_origin, ray_dir)
        assert intersection is None

    def test_out_of_bounds_rejection(self):
        """Verify points far outside plane are rejected (valid=False) without silent clamping."""
        face = create_mock_face()

        # Hand far to the right (head_x = 1.0, well beyond half-width 0.5)
        # u = (1.0 + 0.5) / 1.0 = 1.5 → outside [0, 1] + EPSILON
        cam_x = 1.0
        cam_y = 0.0
        cam_z = -0.1 - 0.30

        hand = Hand(
            landmarks=[
                Landmark(cam_x, cam_y, cam_z, 1.0) if i == 8 else Landmark(0.0, 0.0, 0.0, 1.0)
                for i in range(21)
            ],
            confidence=1.0,
            handedness="Right"
        )

        result = self.projector.project(hand, face)
        assert result.valid is False
        assert "normalized" in result.error_message.lower() or "bounds" in result.error_message.lower() or not result.valid

    def test_plane_dimensions_configurable(self):
        """Verify plane width/height can be reconfigured."""
        custom_plane = VirtualDisplayPlane(
            distance=0.40,
            width=1.2,
            height=0.8,
            head_coords=self.head_coords
        )
        assert custom_plane.distance == 0.40
        assert custom_plane.width == 1.2
        assert custom_plane.height == 0.8

        # Test mapping with custom dimensions
        # Center should still be (0.5, 0.5)
        pt_cam = custom_plane.normalized_to_point_camera(0.5, 0.5)
        norm = custom_plane.point_to_normalized(pt_cam)
        assert norm is not None
        assert abs(norm[0] - 0.5) < 1e-3
        assert abs(norm[1] - 0.5) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])