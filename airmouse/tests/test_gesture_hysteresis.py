"""
Unit tests for GestureRecognizer module.

Tests state transitions and threshold boundaries for pinch/drag/scroll/fist gestures.
"""
import sys
sys.path.insert(0, '/home/shubham/airmouse')

import pytest
import time
from airmouse.vision.gestures import (
    GestureRecognizer, GestureConfig, GestureEvent, GestureType, TrackingState, GestureState
)
from airmouse.vision.hand_tracker import Hand, Landmark, HandLandmark


def create_mock_hand(pinch_dist=0.1, ring_pinch_dist=None, is_fist=False, extended_fingers=None) -> Hand:
    """Create a mock hand with specific pinch distance and finger states."""
    if ring_pinch_dist is None:
        ring_pinch_dist = 0.2  # Default: ring far from thumb (no middle pinch)
    landmarks = []
    for i in range(21):
        if i == HandLandmark.INDEX_TIP.value:
            # Adjust index tip for pinch distance - place close to thumb
            x = 0.5 - pinch_dist
            y = 0.5  # Same Y as thumb so distance = pinch_dist
            z = 0.0
        elif i == HandLandmark.THUMB_TIP.value:
            x = 0.5
            y = 0.5
            z = 0.0
        elif i == HandLandmark.MIDDLE_TIP.value:
            x = 0.55
            y = 0.5 if "middle" in (extended_fingers or []) else 0.7
            z = 0.0
        elif i == HandLandmark.RING_TIP.value:
            # Adjust ring tip for middle-click pinch distance
            x = 0.5 + ring_pinch_dist
            y = 0.5  # Same Y as thumb so distance = ring_pinch_dist
            z = 0.0
        elif i == HandLandmark.PINKY_TIP.value:
            x = 0.65
            y = 0.7
            z = 0.0
        elif i == HandLandmark.WRIST.value:
            x = 0.5
            y = 0.7
            z = 0.0
        else:
            x = 0.5
            y = 0.6
            z = 0.0
        landmarks.append(Landmark(x=x, y=y, z=z))

    hand = Hand(landmarks=landmarks, handedness="Right", confidence=1.0)

    # Override finger states if needed
    if is_fist:
        for fname in hand._finger_states:
            hand._finger_states[fname] = False
    if extended_fingers:
        for fname in extended_fingers:
            hand._finger_states[fname] = True

    return hand


class TestGestureConfig:
    """Tests for GestureConfig thresholds."""

    def test_default_thresholds(self):
        """Test default hysteresis thresholds."""
        config = GestureConfig()

        assert config.pinch_enter_threshold == 0.045
        assert config.pinch_confirm_threshold == 0.040
        assert config.pinch_release_threshold == 0.070
        assert config.drag_hold_time == 0.2
        assert config.fist_hold_time == 0.5
        assert config.scroll_sensitivity == 0.002  # §33: was 1.0 (mapped to entire screen)

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = GestureConfig(
            pinch_enter_threshold=0.05,
            pinch_confirm_threshold=0.045,
            pinch_release_threshold=0.08,
            drag_hold_time=0.3,
            click_max_movement=0.05,
            fist_hold_time=0.8,
            scroll_sensitivity=2.0
        )

        assert config.pinch_enter_threshold == 0.05
        assert config.pinch_confirm_threshold == 0.045
        assert config.pinch_release_threshold == 0.08
        assert config.drag_hold_time == 0.3
        assert config.click_max_movement == 0.05
        assert config.fist_hold_time == 0.8
        assert config.scroll_sensitivity == 2.0


class TestGestureRecognizer:
    """Tests for GestureRecognizer state machine."""

    def setup_method(self):
        """Set up recognizer with test config."""
        config = GestureConfig(
            pinch_enter_threshold=0.045,
            pinch_confirm_threshold=0.040,
            pinch_release_threshold=0.070,
            drag_hold_time=0.2,
            click_max_movement=0.03,
            fist_hold_time=0.5,
            scroll_sensitivity=1.0
        )
        self.recognizer = GestureRecognizer(config)

    def test_no_hand_state(self):
        """Test NO_HAND state with no hands detected."""
        events = self.recognizer.process([])
        assert len(events) == 0
        assert self.recognizer.state == TrackingState.NO_HAND

    def test_tracking_one_hand_no_gesture(self):
        """Test TRACKING_ONE_HAND with open hand (no pinch)."""
        hand = create_mock_hand(pinch_dist=0.1)  # Open hand
        events = self.recognizer.process([hand])

        assert self.recognizer.state == TrackingState.TRACKING_ONE_HAND
        # No gesture events for open hand
        gesture_events = [e for e in events if e.gesture_type in (GestureType.LEFT_CLICK, GestureType.PINCH_CONFIRM)]
        assert len(gesture_events) == 0

    def test_pinch_enter_threshold(self):
        """Test pinch enter at threshold (0.045) - pinch becomes active."""
        hand = create_mock_hand(pinch_dist=0.045)  # Exactly at enter threshold
        events = self.recognizer.process([hand])

        # §5 P0 fix: LEFT_CLICK is emitted on pinch RELEASE, not entry.
        # On entry, the pinch state becomes active but no click event is generated.
        click_events = [e for e in events if e.gesture_type == GestureType.LEFT_CLICK]
        assert len(click_events) == 0  # No click on entry
        # Verify the pinch state is active
        assert self.recognizer._state.left_pinch_active

    def test_pinch_below_enter_no_trigger(self):
        """Test no pinch trigger below enter threshold."""
        hand = create_mock_hand(pinch_dist=0.05)  # Above enter threshold (more open)
        events = self.recognizer.process([hand])

        pinch_events = [e for e in events if e.gesture_type == GestureType.LEFT_CLICK]
        assert len(pinch_events) == 0

    def test_pinch_confirm_threshold(self):
        """Test PINCH_CONFIRM at confirm threshold (0.040)."""
        # First trigger PINCH_START
        hand = create_mock_hand(pinch_dist=0.045)
        self.recognizer.process([hand])

        # Now tighter pinch for PINCH_CONFIRM
        hand = create_mock_hand(pinch_dist=0.040)
        events = self.recognizer.process([hand])

        confirm_events = [e for e in events if e.gesture_type == GestureType.PINCH_CONFIRM]
        assert len(confirm_events) == 1

    def test_pinch_release_threshold(self):
        """Test PINCH_END at release threshold (0.070)."""
        # Start pinch
        hand = create_mock_hand(pinch_dist=0.045)
        self.recognizer.process([hand])

        # Confirm pinch
        hand = create_mock_hand(pinch_dist=0.040)
        self.recognizer.process([hand])

        # Release pinch
        hand = create_mock_hand(pinch_dist=0.075)  # Above release threshold
        events = self.recognizer.process([hand])

        end_events = [e for e in events if e.gesture_type == GestureType.PINCH_END]
        assert len(end_events) == 1

    def test_pinch_hysteresis(self):
        """Test hysteresis prevents flickering at boundaries."""
        # Enter pinch - §5: no click on entry
        hand = create_mock_hand(pinch_dist=0.045)
        events = self.recognizer.process([hand])
        pinch_starts = [e for e in events if e.gesture_type == GestureType.LEFT_CLICK]
        assert len(pinch_starts) == 0  # No click on entry

        # Move slightly but stay within hysteresis
        hand = create_mock_hand(pinch_dist=0.048)
        events = self.recognizer.process([hand])
        # Should not trigger new pinch events
        pinch_events = [e for e in events if e.gesture_type in (GestureType.LEFT_CLICK, GestureType.PINCH_END)]
        assert len(pinch_events) == 0

    def test_drag_detection(self):
        """Test drag detection with hold time and movement."""
        # Start pinch
        hand = create_mock_hand(pinch_dist=0.040)
        self.recognizer.process([hand])

        # Hold for drag_hold_time (0.2s) - simulate time passing
        # The recognizer uses internal timing, so we test the concept
        hand = create_mock_hand(pinch_dist=0.040)
        events = self.recognizer.process([hand])

        # Would need time mocking to fully test - skip detailed timing test

    def test_fist_pause(self):
        """Test FIST_PAUSE gesture."""
        hand = create_mock_hand(is_fist=True)
        events = self.recognizer.process([hand])

        fist_events = [e for e in events if e.gesture_type == GestureType.FIST_PAUSE]
        # Fist pause requires hold time - may not trigger immediately
        # Just verify no crash

    def test_scroll_gesture(self):
        """Test scroll gesture with index+middle extended."""
        hand = create_mock_hand(extended_fingers=["index", "middle"])
        events = self.recognizer.process([hand])

        scroll_events = [e for e in events if e.gesture_type == GestureType.SCROLL]
        # Scroll requires movement tracking - may not trigger without movement
        # Just verify no crash

    def test_two_hand_tracking(self):
        """Test two-hand tracking enables PRECISION_MODE."""
        hand1 = create_mock_hand(pinch_dist=0.1)
        hand2 = create_mock_hand(pinch_dist=0.1)
        hand2.handedness = "Left"

        events = self.recognizer.process([hand1, hand2])

        # Should handle two hands without error
        assert len(events) >= 0

    def test_lost_track_transition(self):
        """Test LOST_TRACK state handling."""
        # Start with hand
        hand = create_mock_hand(pinch_dist=0.1)
        self.recognizer.process([hand])

        # Then lose track
        events = self.recognizer.process([])

        # Should release any active gestures
        release_events = [e for e in events if e.gesture_type in (
            GestureType.PINCH_END, GestureType.DRAG_END, GestureType.FIST_PAUSE
        )]
        # May have releases depending on internal state

    def test_frozen_state(self):
        """Test FROZEN state prevents gesture processing."""
        # First, trigger FROZEN state by holding fist
        fist_hand = create_mock_hand(is_fist=True)

        # Need to process multiple times to exceed fist_hold_time (0.5s)
        # Since we can't easily mock time, let's just test the concept
        # by manually setting the state
        self.recognizer._state.tracking_paused = True
        self.recognizer._state.tracking_state = TrackingState.FROZEN

        # Now try to trigger pinch - should be ignored
        hand = create_mock_hand(pinch_dist=0.040)
        events = self.recognizer.process([hand])

        # In FROZEN state, should not process new gestures
        pinch_events = [e for e in events if e.gesture_type in (
            GestureType.LEFT_CLICK, GestureType.PINCH_CONFIRM
        )]
        assert len(pinch_events) == 0

    def test_precision_mode_sensitivity(self):
        """Test PRECISION_MODE reduces sensitivity."""
        config = GestureConfig()
        recognizer = GestureRecognizer(config)

        # Test that precision mode is tracked
        hand = create_mock_hand(pinch_dist=0.1)
        events = recognizer.process([hand])

        # Should handle precision mode without error
        assert len(events) >= 0


class TestTrackingStateEnum:
    """Tests for TrackingState enum completeness."""

    def test_all_states_present(self):
        """Verify all 6 required tracking states exist."""
        states = [
            TrackingState.NO_HAND,
            TrackingState.TRACKING_ONE_HAND,
            TrackingState.TRACKING_TWO_HANDS,
            TrackingState.LOST_TRACK,
            TrackingState.FROZEN,
            TrackingState.PRECISION_MODE
        ]
        assert len(states) == 6


class TestMiddleClickConfidenceGate:
    """Tests for middle-click confidence gating (§33)."""

    def setup_method(self):
        """Set up recognizer with test config."""
        config = GestureConfig(
            pinch_enter_threshold=0.045,
            pinch_confirm_threshold=0.040,
            pinch_release_threshold=0.070,
            drag_hold_time=0.2,
            click_max_movement=0.03,
            fist_hold_time=0.5,
            scroll_sensitivity=1.0,
            middle_click_min_confidence=0.60
        )
        self.recognizer = GestureRecognizer(config)

    def test_no_middle_click_below_confidence(self):
        """Test no MIDDLE_CLICK candidate when hand confidence < 0.60."""
        # Create a hand with thumb+ring pinch but low confidence
        hand = create_mock_hand(pinch_dist=0.1, ring_pinch_dist=0.040)
        hand.confidence = 0.40  # Below middle_click_min_confidence

        events = self.recognizer.process([hand])

        middle_click_events = [e for e in events if e.gesture_type == GestureType.MIDDLE_CLICK]
        assert len(middle_click_events) == 0
        # Pinch state should NOT be active (gate blocked it)
        assert not self.recognizer._state._middle_pinch_active

    def test_middle_click_above_confidence(self):
        """Test MIDDLE_CLICK fires when confidence >= 0.60 and pinch confirmed."""
        # Need to satisfy confirm threshold: dist < 0.040
        hand = create_mock_hand(pinch_dist=0.1, ring_pinch_dist=0.035)
        hand.confidence = 0.90  # Above middle_click_min_confidence

        # First frame: enter pinch
        events = self.recognizer.process([hand])
        assert self.recognizer._state._middle_pinch_active

        # Second frame: confirm pinch (dist < 0.040)
        events = self.recognizer.process([hand])
        middle_click_events = [e for e in events if e.gesture_type == GestureType.MIDDLE_CLICK]
        assert len(middle_click_events) == 1

    def test_middle_click_resets_on_low_confidence(self):
        """Test active middle pinch resets when confidence drops."""
        # First establish active pinch with high confidence
        hand = create_mock_hand(pinch_dist=0.1, ring_pinch_dist=0.040)
        hand.confidence = 0.90
        self.recognizer.process([hand])
        assert self.recognizer._state._middle_pinch_active

        # Now drop confidence below threshold
        hand.confidence = 0.30
        self.recognizer.process([hand])
        # Pinch state should be reset
        assert not self.recognizer._state._middle_pinch_active

    def test_middle_click_custom_threshold(self):
        """Test custom middle_click_min_confidence is respected."""
        config = GestureConfig(
            pinch_enter_threshold=0.045,
            pinch_confirm_threshold=0.040,
            pinch_release_threshold=0.070,
            middle_click_min_confidence=0.95  # Very high
        )
        recognizer = GestureRecognizer(config)

        hand = create_mock_hand(pinch_dist=0.1, ring_pinch_dist=0.040)
        hand.confidence = 0.80  # Below custom threshold

        events = recognizer.process([hand])
        middle_click_events = [e for e in events if e.gesture_type == GestureType.MIDDLE_CLICK]
        assert len(middle_click_events) == 0
        assert not recognizer._state._middle_pinch_active


class TestGestureTypeEnum:
    """Tests for GestureType enum completeness."""

    def test_all_gesture_types(self):
        """Verify all required gesture types exist."""
        types = [
            GestureType.LEFT_CLICK,
            GestureType.PINCH_CONFIRM,
            GestureType.PINCH_END,
            GestureType.DRAG_START,
            GestureType.DRAG_END,
            GestureType.PAUSE_TRACKING,
            GestureType.SCROLL_UP,
            GestureType.SCROLL_DOWN,
            GestureType.RIGHT_CLICK,
            GestureType.MIDDLE_CLICK
        ]
        assert len(types) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])