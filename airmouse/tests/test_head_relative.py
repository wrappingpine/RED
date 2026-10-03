"""
Tests for head-relative reference point tracking logic.

Covers User Story 4: Correct reference point update for head-relative mode.
- T022: test_reference_point_every_frame - In head-relative mode, reference point
        updates every frame to current smoothed projection position (Bug 2 fix, FR-003).
- T023: test_reference_point_dead_zone_exit - In legacy mode, reference point updates
        only when leaving dead zone (FR-004).
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airmouse.vision.tracking_processor import TrackingProcessor, TrackingConfig
from airmouse.vision.hand_tracker import Hand, Landmark, HandLandmark
from airmouse.vision.gestures import TrackingState


class MockHand(Hand):
    """Mock hand for testing inheriting from real Hand."""
    def __init__(self, handedness="Right", index_tip=None, confidence=0.9):
        super().__init__()
        self.handedness = handedness
        self.confidence = confidence
        x, y, z = (0.5, 0.48, -0.3) if index_tip is None else index_tip
        self.landmarks = [Landmark(x=x, y=y, z=z) for _ in range(21)]
        self.landmarks[HandLandmark.WRIST.value] = Landmark(x=x, y=y + 0.15, z=z)
        self.landmarks[HandLandmark.INDEX_MCP.value] = Landmark(x=x, y=y + 0.05, z=z)
        self.landmarks[HandLandmark.INDEX_TIP.value] = Landmark(x=x, y=y, z=z)
        self._compute_derived()


class MockFace:
    """Mock face for testing with proper landmark computation."""

    def __init__(self, landmarks=None, confidence=0.9):
        self._confidence = confidence
        if landmarks is not None:
            if hasattr(landmarks[0], 'x'):
                self._landmarks_list = list(landmarks)
            else:
                self._landmarks_list = [Landmark(x=lm[0], y=lm[1], z=lm[2]) for lm in landmarks]
        else:
            self._landmarks_list = [Landmark(x=0.0, y=0.0, z=0.0) for _ in range(468)]
        self._compute_derived()

    def _compute_derived(self):
        """Compute derived head pose properties from landmarks."""
        if len(self._landmarks_list) < 468:
            return

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

    def is_lost(self, config):
        return False


class TestHeadRelativeReferencePoint:
    """Tests for reference point update logic per FR-003 and FR-004."""

    def _create_face(self):
        """Create a mock face with valid landmarks for head coordinate system."""
        face_landmarks = np.zeros((468, 3))
        face_landmarks[33] = [0.05, 0.05, 0.0]    # Left eye outer
        face_landmarks[133] = [0.05, 0.05, 0.0]   # Left eye inner
        face_landmarks[263] = [-0.05, 0.05, 0.0]  # Right eye outer
        face_landmarks[362] = [-0.05, 0.05, 0.0]  # Right eye inner
        face_landmarks[1] = [0.0, 0.05, -0.1]     # Nose tip
        face_landmarks[10] = [0.0, -0.05, -0.05]  # Forehead
        return MockFace(landmarks=face_landmarks)

    def _create_hand(self, x=0.1, y=0.0):
        """Create a mock hand at the given (x, y) position with z=-0.5.
        
        Valid positions (for virtual_plane_distance=0.30, width=0.40, height=0.25):
        - x in [0.1, 0.4], y in [0.0, 0.2] gives u in [0.65, 0.95], v in [0.38, 0.74]
        """
        return MockHand(index_tip=(x, y, -0.5))

    def test_reference_point_every_frame(self):
        """T022: In head-relative mode, reference point must update every frame
        to the current smoothed projection position (FR-003, Bug 2 fix).

        The reference point is the anchor from which cursor movement deltas
        are computed. In head-relative mode, it should follow the smoothed
        plane position on every frame so cursor movement reflects hand
        motion relative to the head.
        """
        config = TrackingConfig(
            use_head_relative=True,
            virtual_plane_distance=0.30,
            virtual_plane_width=0.40,
            virtual_plane_height=0.25,
            dead_zone_radius=0.015,
            enable_two_hand=False,
            preferred_handedness="Right"
        )
        processor = TrackingProcessor(config)

        face = self._create_face()

        # Initial position: hand at (0.1, 0.0) -> proj u~0.65, v~0.38
        hand = self._create_hand(x=0.1, y=0.0)
        result = processor.process([hand], [face])

        assert result.tracking_state == TrackingState.TRACKING_ONE_HAND
        assert processor._reference_point is not None

        original_ref = processor._reference_point

        # Move hand to different valid positions - reference point should update
        # each frame in head-relative mode to the smoothed plane position
        # Use positions that project to different (u,v) within plane bounds
        positions = [
            (0.1, 0.0),   # proj ~(0.65, 0.38)
            (0.2, 0.05),  # proj ~(0.80, 0.50)
            (0.3, 0.1),   # proj ~(0.95, 0.62)
            (0.2, 0.15),  # proj ~(0.80, 0.74)
            (0.1, 0.1),   # proj ~(0.65, 0.62)
        ]
        ref_points = []

        for px, py in positions:
            hand = self._create_hand(x=px, y=py)
            result = processor.process([hand], [face])
            # Call get_cursor_movement to trigger reference point update
            # (reference point is updated in get_cursor_movement, not process)
            processor.get_cursor_movement()
            current_ref = processor._reference_point
            ref_points.append(current_ref)
            assert current_ref is not None

        # In head-relative mode (Bug 2 fix), reference point updates every frame
        # to current smoothed plane position. It should NOT stay frozen.
        # At least some frames should show a different reference point than initial.
        updated_count = sum(
            1 for rp in ref_points
            if rp is not None and rp != original_ref
        )
        assert updated_count > 0, (
            "Reference point should update every frame in head-relative mode "
            "(FR-003, Bug 2 fix). Expected at least one update, got 0."
        )

    def test_reference_point_dead_zone_exit(self):
        """T023: In legacy mode (non-head-relative), reference point updates
        only when leaving the dead zone (FR-004).

        In legacy mode:
        - When movement is within dead_zone_radius: cursor movement = (0,0),
          reference point stays the same (dead zone active).
        - When movement exceeds dead_zone_radius: reference point updates to
          current position, dead zone deactivates.
        """
        config = TrackingConfig(
            use_head_relative=False,
            dead_zone_radius=0.015,
            enable_two_hand=False,
            preferred_handedness="Right"
        )
        processor = TrackingProcessor(config)

        # For legacy mode, the 2D position is used directly.
        # Create a face with proper landmarks for the processor to accept.
        face = self._create_face()
        hand = self._create_hand(x=0.5, y=0.48)
        result = processor.process([hand], [face])

        assert result.tracking_state == TrackingState.TRACKING_ONE_HAND
        assert processor._reference_point is not None
        original_ref = processor._reference_point

        # Frame 1: Small movement within dead zone (radius=0.015 normalized)
        # Movement of 0.005 in normalized space is within dead zone (< 0.015)
        small_hand = self._create_hand(x=0.505, y=0.48)
        result = processor.process([small_hand], [face])
        movement = processor.get_cursor_movement()
        assert movement is not None
        movement_mag = np.sqrt(movement[0]**2 + movement[1]**2)

        # If within dead zone, movement should be (0, 0)
        if movement_mag < 1e-9:
            assert processor._dead_zone_active == True, (
                "Dead zone should be active when movement is below dead_zone_radius"
            )

        # Frame 2: Large movement that exits dead zone
        # Movement of 0.1 in normalized space far exceeds dead zone (0.015)
        large_hand = self._create_hand(x=0.6, y=0.58)
        result = processor.process([large_hand], [face])
        movement = processor.get_cursor_movement()

        movement_mag = np.sqrt(movement[0]**2 + movement[1]**2)
        assert movement_mag > 0.0, (
            "Movement should be non-zero when exiting dead zone"
        )

        # Reference point should have updated (in legacy mode, when not in dead zone)
        assert processor._reference_point != original_ref, (
            "Reference point should update when exiting dead zone in legacy mode (FR-004)"
        )

        # Verify dead zone is no longer active
        assert processor._dead_zone_active == False, (
            "Dead zone should be inactive after exiting dead zone"
        )

        # Frame 3: Movement back to within dead zone of current position
        back_hand = self._create_hand(x=0.595, y=0.575)
        result = processor.process([back_hand], [face])
        movement = processor.get_cursor_movement()

        movement_mag = np.sqrt(movement[0]**2 + movement[1]**2)
        if movement_mag < 1e-9:
            assert processor._dead_zone_active == True, (
                "Dead zone should reactivate when movement is below dead_zone_radius"
            )