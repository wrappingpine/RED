"""
Unit tests for TrackingProcessor module.

Tests full pipeline with mock data.
"""
import sys
sys.path.insert(0, '/home/shubham/airmouse')

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from airmouse.vision.tracking_processor import TrackingProcessor, TrackingConfig
from airmouse.vision.hand_tracker import Hand
from airmouse.vision.gestures import TrackingState


class MockHand(Hand):
    """Mock hand for testing."""
    def __init__(self, handedness="Right", landmarks=None, pinch_dist=0.1, is_fist=False, confidence=0.9):
        self._handedness = handedness
        self._pinch_dist = pinch_dist
        self._is_fist = is_fist
        self._extended_fingers = ["index", "middle"] if pinch_dist > 0.05 else []
        self._confidence = confidence

        # Create proper Landmark objects from landmarks array
        if landmarks is not None:
            self._landmarks_list = []
            for i in range(landmarks.shape[0]):
                from airmouse.vision.hand_tracker import Landmark
                self._landmarks_list.append(Landmark(
                    x=landmarks[i, 0],
                    y=landmarks[i, 1],
                    z=landmarks[i, 2]
                ))
        else:
            # Default: create 21 landmarks at a realistic hand position
            # IMPORTANT: For head-relative tracking, hand should be in front of face (negative Z)
            # Camera coordinates: +X right, +Y down, +Z forward
            # So a hand in front of face has NEGATIVE Z
            from airmouse.vision.hand_tracker import Landmark
            from airmouse.vision.hand_tracker import HandLandmark

            # Create landmarks for a realistic right hand at center (in front of face)
            self._landmarks_list = [Landmark(x=0.5, y=0.5, z=-0.3) for _ in range(21)]

            # Wrist
            self._landmarks_list[HandLandmark.WRIST.value] = Landmark(x=0.5, y=0.65, z=-0.3)
            # Index finger
            self._landmarks_list[HandLandmark.INDEX_MCP.value] = Landmark(x=0.5, y=0.58, z=-0.3)
            self._landmarks_list[HandLandmark.INDEX_PIP.value] = Landmark(x=0.5, y=0.53, z=-0.3)
            self._landmarks_list[HandLandmark.INDEX_DIP.value] = Landmark(x=0.5, y=0.50, z=-0.3)
            self._landmarks_list[HandLandmark.INDEX_TIP.value] = Landmark(x=0.5, y=0.48, z=-0.3)
            # Thumb
            self._landmarks_list[HandLandmark.THUMB_CMC.value] = Landmark(x=0.45, y=0.62, z=-0.3)
            self._landmarks_list[HandLandmark.THUMB_MCP.value] = Landmark(x=0.43, y=0.58, z=-0.3)
            self._landmarks_list[HandLandmark.THUMB_IP.value] = Landmark(x=0.42, y=0.54, z=-0.3)
            self._landmarks_list[HandLandmark.THUMB_TIP.value] = Landmark(x=0.41, y=0.50, z=-0.3)
            # Middle finger
            self._landmarks_list[HandLandmark.MIDDLE_MCP.value] = Landmark(x=0.55, y=0.58, z=-0.3)
            self._landmarks_list[HandLandmark.MIDDLE_PIP.value] = Landmark(x=0.55, y=0.53, z=-0.3)
            self._landmarks_list[HandLandmark.MIDDLE_DIP.value] = Landmark(x=0.55, y=0.50, z=-0.3)
            self._landmarks_list[HandLandmark.MIDDLE_TIP.value] = Landmark(x=0.55, y=0.48, z=-0.3)
            # Ring finger
            self._landmarks_list[HandLandmark.RING_MCP.value] = Landmark(x=0.6, y=0.58, z=-0.3)
            self._landmarks_list[HandLandmark.RING_PIP.value] = Landmark(x=0.6, y=0.53, z=-0.3)
            self._landmarks_list[HandLandmark.RING_DIP.value] = Landmark(x=0.6, y=0.50, z=-0.3)
            self._landmarks_list[HandLandmark.RING_TIP.value] = Landmark(x=0.6, y=0.48, z=-0.3)
            # Pinky
            self._landmarks_list[HandLandmark.PINKY_MCP.value] = Landmark(x=0.65, y=0.58, z=-0.3)
            self._landmarks_list[HandLandmark.PINKY_PIP.value] = Landmark(x=0.65, y=0.53, z=-0.3)
            self._landmarks_list[HandLandmark.PINKY_DIP.value] = Landmark(x=0.65, y=0.50, z=-0.3)
            self._landmarks_list[HandLandmark.PINKY_TIP.value] = Landmark(x=0.65, y=0.48, z=-0.3)

    @property
    def handedness(self):
        return self._handedness

    @property
    def landmarks(self):
        return self._landmarks_list

    @landmarks.setter
    def landmarks(self, value):
        self._landmarks_list = value

    @property
    def confidence(self):
        return self._confidence

    @property
    def index_tip(self):
        return self._landmarks_list[8] if len(self._landmarks_list) > 8 else None

    @property
    def palm_center(self):
        return self._landmarks_list[0] if len(self._landmarks_list) > 0 else None

    @property
    def thumb_tip(self):
        return self._landmarks_list[4] if len(self._landmarks_list) > 4 else None

    @property
    def middle_tip(self):
        return self._landmarks_list[12] if len(self._landmarks_list) > 12 else None

    def pinch_distance(self, finger1="thumb", finger2="index"):
        return self._pinch_dist

    def is_fist(self):
        return self._is_fist

    def is_finger_extended(self, finger):
        return finger in self._extended_fingers

    def set_index_tip(self, x: float, y: float, z: float):
        """Update index fingertip position for testing."""
        if len(self._landmarks_list) > 8:
            self._landmarks_list[8].x = x
            self._landmarks_list[8].y = y
            self._landmarks_list[8].z = z


class MockFace:
    """Mock face for testing."""
    def __init__(self, landmarks=None, confidence=0.9):
        self._confidence = confidence

        # Create proper Landmark objects from landmarks array
        if landmarks is not None:
            from airmouse.vision.hand_tracker import Landmark
            self._landmarks_list = []
            for i in range(landmarks.shape[0]):
                self._landmarks_list.append(Landmark(
                    x=landmarks[i, 0],
                    y=landmarks[i, 1],
                    z=landmarks[i, 2]
                ))
        else:
            # Default: create 468 landmarks at neutral position
            from airmouse.vision.hand_tracker import Landmark
            self._landmarks_list = [Landmark(x=0.0, y=0.0, z=0.0) for _ in range(468)]

        # Compute derived properties needed for head coordinate system
        self._compute_derived()

    def _compute_derived(self):
        """Compute derived head pose properties from landmarks."""
        if len(self._landmarks_list) < 468:
            return

        # Get key landmarks (same indices as real Face class)
        LEFT_EYE_INNER = 133
        LEFT_EYE_OUTER = 33
        RIGHT_EYE_INNER = 362
        RIGHT_EYE_OUTER = 263
        NOSE_TIP = 1
        FOREHEAD = 10

        left_eye_inner = self._landmarks_list[LEFT_EYE_INNER]
        left_eye_outer = self._landmarks_list[LEFT_EYE_OUTER]
        right_eye_inner = self._landmarks_list[RIGHT_EYE_INNER]
        right_eye_outer = self._landmarks_list[RIGHT_EYE_OUTER]
        nose_tip = self._landmarks_list[NOSE_TIP]
        forehead = self._landmarks_list[FOREHEAD]

        # Compute eye centers
        from airmouse.vision.hand_tracker import Landmark
        self._left_eye_center = Landmark(
            x=(left_eye_inner.x + left_eye_outer.x) / 2,
            y=(left_eye_inner.y + left_eye_outer.y) / 2,
            z=(left_eye_inner.z + left_eye_outer.z) / 2
        )
        self._right_eye_center = Landmark(
            x=(right_eye_inner.x + right_eye_outer.x) / 2,
            y=(right_eye_inner.y + right_eye_outer.y) / 2,
            z=(right_eye_inner.z + right_eye_outer.z) / 2
        )

        # Eye midpoint = origin of head coordinate system
        self._eye_midpoint = Landmark(
            x=(self._left_eye_center.x + self._right_eye_center.x) / 2,
            y=(self._left_eye_center.y + self._right_eye_center.y) / 2,
            z=(self._left_eye_center.z + self._right_eye_center.z) / 2
        )

        self._nose_tip = nose_tip
        self._forehead = forehead

    @property
    def landmarks(self):
        return self._landmarks_list

    @property
    def confidence(self):
        return self._confidence

    @property
    def eye_midpoint(self):
        return self._eye_midpoint

    @property
    def nose_tip(self):
        return self._nose_tip

    @property
    def forehead(self):
        return self._forehead

    @property
    def left_eye_center(self):
        return self._left_eye_center

    @property
    def right_eye_center(self):
        return self._right_eye_center


class TestTrackingConfig:
    """Tests for TrackingConfig."""

    def test_default_config(self):
        """Test default tracking configuration."""
        config = TrackingConfig()

        assert config.use_head_relative is True
        assert config.virtual_plane_distance == 0.30
        assert config.virtual_plane_width == 0.40
        assert config.virtual_plane_height == 0.25
        assert config.enable_two_hand is True
        assert config.preferred_handedness == "Right"


class TestTrackingProcessor:
    """Tests for TrackingProcessor."""

    def setup_method(self):
        """Set up processor with test config."""
        config = TrackingConfig(
            use_head_relative=True,
            virtual_plane_distance=0.30,
            virtual_plane_width=0.40,
            virtual_plane_height=0.25,
            enable_two_hand=True,
            preferred_handedness="Right"
        )
        self.processor = TrackingProcessor(config)

    def test_no_hands_no_face(self):
        """Test processing with no hands and no face."""
        hands = []
        faces = []

        result = self.processor.process(hands, faces)

        assert result is not None
        assert len(result.tracked_hands) == 0
        assert result.tracking_state == TrackingState.NO_HAND

    def test_one_hand_no_face(self):
        """Test processing with one hand but no face (head-relative mode returns LOST_TRACK)."""
        hand = MockHand(handedness="Right")
        hands = [hand]
        faces = []

        result = self.processor.process(hands, faces)

        # With use_head_relative=True and no face, returns LOST_TRACK
        assert result.tracking_state == TrackingState.LOST_TRACK

    def test_one_hand_with_face(self):
        """Test processing with one hand and face (head-relative)."""
        # Create mock face with proper landmarks for head coordinate system
        face_landmarks = np.zeros((468, 3))
        face_landmarks[33] = [0.05, 0.0, 0.0]    # Left eye inner
        face_landmarks[133] = [0.05, 0.0, 0.0]   # Left eye outer
        face_landmarks[263] = [-0.05, 0.0, 0.0]  # Right eye inner
        face_landmarks[362] = [-0.05, 0.0, 0.0]  # Right eye outer
        face_landmarks[1] = [0.0, -0.05, -0.1]   # Nose tip (negative Z = in front of eyes)
        face_landmarks[10] = [0.0, -0.1, -0.05]  # Forehead (negative Y = up in camera coords)

        face = MockFace(landmarks=face_landmarks)
        hand = MockHand(handedness="Right")
        # Set index fingertip in front of face
        hand.set_index_tip(0.0, 0.0, -0.5)

        result = self.processor.process([hand], [face])

        assert result.tracking_state == TrackingState.TRACKING_ONE_HAND
        assert len(result.tracked_hands) == 1

    def test_two_hands_with_face(self):
        """Test processing with two hands and face."""
        face_landmarks = np.zeros((468, 3))
        face_landmarks[33] = [0.05, 0.0, 0.0]    # Left eye inner
        face_landmarks[133] = [0.05, 0.0, 0.0]   # Left eye outer
        face_landmarks[263] = [-0.05, 0.0, 0.0]  # Right eye inner
        face_landmarks[362] = [-0.05, 0.0, 0.0]  # Right eye outer
        face_landmarks[1] = [0.0, -0.05, -0.1]   # Nose tip (negative Z = in front of eyes)
        face_landmarks[10] = [0.0, -0.1, -0.05]  # Forehead (negative Y = up in camera coords)

        face = MockFace(landmarks=face_landmarks)

        hand1 = MockHand(handedness="Right")
        hand1.set_index_tip(0.0, 0.0, -0.5)

        hand2 = MockHand(handedness="Left")
        hand2.set_index_tip(-0.05, 0.0, -0.5)

        result = self.processor.process([hand1, hand2], [face])

        assert result.tracking_state == TrackingState.TRACKING_TWO_HANDS
        assert len(result.tracked_hands) == 2

    def test_primary_hand_selection_right_preferred(self):
        """Test primary hand selection prefers Right hand."""
        config = TrackingConfig(preferred_handedness="Right")
        processor = TrackingProcessor(config)

        face_landmarks = np.zeros((468, 3))
        face_landmarks[33] = [0.05, 0.0, 0.0]    # Left eye inner
        face_landmarks[133] = [0.05, 0.0, 0.0]   # Left eye outer
        face_landmarks[263] = [-0.05, 0.0, 0.0]  # Right eye inner
        face_landmarks[362] = [-0.05, 0.0, 0.0]  # Right eye outer
        face_landmarks[1] = [0.0, -0.05, -0.1]   # Nose tip (negative Z = in front of eyes)
        face_landmarks[10] = [0.0, -0.1, -0.05]  # Forehead (negative Y = up in camera coords)
        face = MockFace(landmarks=face_landmarks)

        hand_left = MockHand(handedness="Left")
        hand_left.set_index_tip(-0.05, 0.0, -0.5)
        hand_right = MockHand(handedness="Right")
        hand_right.set_index_tip(0.05, 0.0, -0.5)

        result = processor.process([hand_left, hand_right], [face])

        # Right hand should be primary
        assert result.primary_hand is not None
        assert result.primary_hand.handedness == "Right"
        assert result.secondary_hand is not None
        assert result.secondary_hand.handedness == "Left"

    def test_primary_hand_selection_left_preferred(self):
        """Test primary hand selection prefers Left hand."""
        config = TrackingConfig(preferred_handedness="Left")
        processor = TrackingProcessor(config)

        face_landmarks = np.zeros((468, 3))
        face_landmarks[33] = [0.05, 0.0, 0.0]    # Left eye inner
        face_landmarks[133] = [0.05, 0.0, 0.0]   # Left eye outer
        face_landmarks[263] = [-0.05, 0.0, 0.0]  # Right eye inner
        face_landmarks[362] = [-0.05, 0.0, 0.0]  # Right eye outer
        face_landmarks[1] = [0.0, -0.05, -0.1]   # Nose tip (negative Z = in front of eyes)
        face_landmarks[10] = [0.0, -0.1, -0.05]  # Forehead (negative Y = up in camera coords)
        face = MockFace(landmarks=face_landmarks)

        hand_left = MockHand(handedness="Left")
        hand_left.set_index_tip(-0.05, 0.0, -0.5)
        hand_right = MockHand(handedness="Right")
        hand_right.set_index_tip(0.05, 0.0, -0.5)

        result = processor.process([hand_left, hand_right], [face])

        # Left hand should be primary
        assert result.primary_hand is not None
        assert result.primary_hand.handedness == "Left"
        assert result.secondary_hand is not None
        assert result.secondary_hand.handedness == "Right"

    def test_cursor_movement_from_plane(self):
        """Test getting cursor movement from plane coordinates."""
        face_landmarks = np.zeros((468, 3))
        face_landmarks[33] = [0.05, 0.0, 0.0]    # Left eye inner
        face_landmarks[133] = [0.05, 0.0, 0.0]   # Left eye outer
        face_landmarks[263] = [-0.05, 0.0, 0.0]  # Right eye inner
        face_landmarks[362] = [-0.05, 0.0, 0.0]  # Right eye outer
        face_landmarks[1] = [0.0, -0.05, -0.1]   # Nose tip (negative Z = in front of eyes)
        face_landmarks[10] = [0.0, -0.1, -0.05]  # Forehead (negative Y = up in camera coords)
        face = MockFace(landmarks=face_landmarks)

        hand = MockHand(handedness="Right")
        hand.set_index_tip(0.0, 0.0, -0.5)

        result = self.processor.process([hand], [face])

        # Get cursor movement
        dx, dy = result.get_cursor_movement()

        # Should return tuple of normalized movement
        assert isinstance(dx, (int, float))
        assert isinstance(dy, (int, float))

    def test_hand_identity_stability(self):
        """Test hand identity is stable across frames by handedness."""
        face_landmarks = np.zeros((468, 3))
        face_landmarks[33] = [0.05, 0.0, 0.0]    # Left eye inner
        face_landmarks[133] = [0.05, 0.0, 0.0]   # Left eye outer
        face_landmarks[263] = [-0.05, 0.0, 0.0]  # Right eye inner
        face_landmarks[362] = [-0.05, 0.0, 0.0]  # Right eye outer
        face_landmarks[1] = [0.0, -0.05, -0.1]   # Nose tip (negative Z = in front of eyes)
        face_landmarks[10] = [0.0, -0.1, -0.05]  # Forehead (negative Y = up in camera coords)
        face = MockFace(landmarks=face_landmarks)

        # Frame 1
        hand1 = MockHand(handedness="Right")
        hand1.set_index_tip(0.0, 0.0, -0.5)
        result1 = self.processor.process([hand1], [face])

        # Frame 2 (same hand)
        hand2 = MockHand(handedness="Right")
        hand2.set_index_tip(0.01, 0.0, -0.5)  # Slightly moved
        result2 = self.processor.process([hand2], [face])

        # Should maintain same primary hand
        assert result1.primary_hand.handedness == result2.primary_hand.handedness

    def test_hand_loss_and_reacquisition(self):
        """Test handling of hand loss and reacquisition."""
        face_landmarks = np.zeros((468, 3))
        face_landmarks[33] = [0.05, 0.0, 0.0]    # Left eye inner
        face_landmarks[133] = [0.05, 0.0, 0.0]   # Left eye outer
        face_landmarks[263] = [-0.05, 0.0, 0.0]  # Right eye inner
        face_landmarks[362] = [-0.05, 0.0, 0.0]  # Right eye outer
        face_landmarks[1] = [0.0, -0.05, -0.1]   # Nose tip (negative Z = in front of eyes)
        face_landmarks[10] = [0.0, -0.1, -0.05]  # Forehead (negative Y = up in camera coords)
        face = MockFace(landmarks=face_landmarks)

        # Frame 1: hand present
        hand = MockHand(handedness="Right")
        hand.set_index_tip(0.0, 0.0, -0.5)
        result1 = self.processor.process([hand], [face])
        assert result1.tracking_state == TrackingState.TRACKING_ONE_HAND

        # Frame 2: hand lost
        result2 = self.processor.process([], [face])
        assert result2.tracking_state in (TrackingState.LOST_TRACK, TrackingState.NO_HAND)

        # Frame 3: hand reappears
        hand = MockHand(handedness="Right")
        hand.set_index_tip(0.0, 0.0, -0.5)
        result3 = self.processor.process([hand], [face])
        assert result3.tracking_state == TrackingState.TRACKING_ONE_HAND

    def test_head_relative_disabled(self):
        """Test fallback when head-relative tracking is disabled."""
        config = TrackingConfig(use_head_relative=False)
        processor = TrackingProcessor(config)

        hand = MockHand(handedness="Right")
        hand.set_index_tip(0.5, 0.5, 0.5)  # Screen coords

        result = processor.process([hand], [])

        # Should still process but without head-relative projection
        assert result is not None

    def test_two_hand_disabled(self):
        """Test two-hand tracking can be disabled."""
        config = TrackingConfig(enable_two_hand=False)
        processor = TrackingProcessor(config)

        face_landmarks = np.zeros((468, 3))
        face_landmarks[33] = [0.05, 0.0, 0.0]    # Left eye inner
        face_landmarks[133] = [0.05, 0.0, 0.0]   # Left eye outer
        face_landmarks[263] = [-0.05, 0.0, 0.0]  # Right eye inner
        face_landmarks[362] = [-0.05, 0.0, 0.0]  # Right eye outer
        face_landmarks[1] = [0.0, -0.05, -0.1]   # Nose tip (negative Z = in front of eyes)
        face_landmarks[10] = [0.0, -0.1, -0.05]  # Forehead (negative Y = up in camera coords)
        face = MockFace(landmarks=face_landmarks)

        hand1 = MockHand(handedness="Right")
        hand1.set_index_tip(0.0, 0.0, -0.5)
        hand2 = MockHand(handedness="Left")
        hand2.set_index_tip(-0.05, 0.0, -0.5)

        result = processor.process([hand1, hand2], [face])

        # Should only track one hand
        assert result.tracking_state == TrackingState.TRACKING_ONE_HAND
        assert len(result.tracked_hands) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])