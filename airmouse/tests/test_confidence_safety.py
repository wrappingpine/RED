"""
Tests for Tracking Confidence System (§16) and Tracking Loss Safety (§17-18)
"""

import unittest
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from airmouse.vision.confidence import (
    ConfidenceState, ConfidenceInfo, get_confidence_state, CONFIDENCE_THRESHOLDS
)
from airmouse.vision.tracking_status import (
    TrackingStatus, TrackingPhase, LostReason, ConfidenceState as TS_ConfidenceState
)
from airmouse.vision.gestures import GestureType


class TestConfidenceState(unittest.TestCase):
    """Tests for ConfidenceState classification."""

    def test_high_confidence(self):
        """Confidence >= 0.85 should be HIGH."""
        state = get_confidence_state(0.90)
        self.assertEqual(state, ConfidenceState.HIGH)

    def test_medium_confidence(self):
        """Confidence 0.50-0.84 should be MEDIUM."""
        state = get_confidence_state(0.75)
        self.assertEqual(state, ConfidenceState.MEDIUM)
        
        state = get_confidence_state(0.50)
        self.assertEqual(state, ConfidenceState.MEDIUM)

    def test_low_confidence(self):
        """Confidence 0.15-0.49 should be LOW."""
        state = get_confidence_state(0.40)
        self.assertEqual(state, ConfidenceState.LOW)
        
        state = get_confidence_state(0.15)
        self.assertEqual(state, ConfidenceState.LOW)

    def test_lost_confidence(self):
        """Confidence < 0.15 should be LOST."""
        state = get_confidence_state(0.10)
        self.assertEqual(state, ConfidenceState.LOST)
        
        state = get_confidence_state(0.0)
        self.assertEqual(state, ConfidenceState.LOST)


class TestConfidenceInfo(unittest.TestCase):
    """Tests for ConfidenceInfo tracking."""

    def test_movement_safety_multiplier(self):
        """Higher confidence = more responsive movement."""
        info = ConfidenceInfo()
        
        info.update(0.90)  # HIGH
        self.assertEqual(info.get_movement_safety_multiplier(), 1.0)
        
        info.update(0.60)  # MEDIUM
        self.assertEqual(info.get_movement_safety_multiplier(), 1.5)
        
        info.update(0.20)  # LOW
        self.assertEqual(info.get_movement_safety_multiplier(), 3.0)
        
        info.update(0.05)  # LOST
        self.assertEqual(info.get_movement_safety_multiplier(), float('inf'))

    def test_click_threshold_multiplier(self):
        """Higher confidence = more lenient click detection."""
        info = ConfidenceInfo()
        
        info.update(0.90)  # HIGH - should be 0.9
        self.assertAlmostEqual(info.get_click_threshold_multiplier(), 0.9, places=1)
        
        info.update(0.60)  # MEDIUM - should be 1.0
        self.assertEqual(info.get_click_threshold_multiplier(), 1.0)
        
        info.update(0.20)  # LOW - should be 1.5
        self.assertEqual(info.get_click_threshold_multiplier(), 1.5)


class TestTrackingStatus(unittest.TestCase):
    """Tests for TrackingStatus with full lifecycle."""

    def test_initial_state(self):
        """Initial state should be STARTING."""
        status = TrackingStatus()
        self.assertEqual(status.phase, TrackingPhase.STARTING)
        self.assertFalse(status.get_movement_permission())
        self.assertFalse(status.get_gesture_permission())

    def test_tracking_flow(self):
        """Test full tracking lifecycle."""
        status = TrackingStatus()
        
        # Start tracking
        status.record_hand_detected(0.90)
        self.assertEqual(status.phase, TrackingPhase.STARTING)
        self.assertTrue(status.get_movement_permission())
        
        # Lose tracking
        status.record_hand_lost(LostReason.NO_HAND_DETECTED)
        self.assertEqual(status.phase, TrackingPhase.LOST)
        self.assertFalse(status.get_movement_permission())
        
        # Reacquire and stabilize
        status.record_reacquisition_start()
        self.assertEqual(status.phase, TrackingPhase.REACQUIRING)
        
        status.record_confidence_verified(0.85)
        status.record_stabilization_frame()
        status.record_stabilization_frame()
        status.record_stabilization_frame()
        status.record_stabilization_frame()
        status.record_stabilization_frame()
        
        # Should stabilize and then recover
        status.record_stabilization_frame()  # 6th frame (HIGH needs 3)
        self.assertEqual(status.phase, TrackingPhase.STABILIZING)
        
        status.record_motion_baseline_established()
        self.assertEqual(status.phase, TrackingPhase.TRACKING)
        self.assertTrue(status.get_movement_permission())

    def test_freeze_state(self):
        """Test tracking freeze (fist gesture) per §17."""
        status = TrackingStatus()
        status.record_hand_detected(0.90)
        
        status.record_frozen(True)
        self.assertEqual(status.phase, TrackingPhase.FROZEN)
        self.assertFalse(status.get_movement_permission())
        
        status.record_frozen(False)
        self.assertEqual(status.phase, TrackingPhase.TRACKING)
        self.assertTrue(status.get_movement_permission())

    def test_low_confidence_blocking(self):
        """Low confidence should block gestures per §53."""
        status = TrackingStatus()
        status.record_hand_detected(0.20)  # LOW confidence
        
        self.assertFalse(status.get_gesture_permission())
        
        status.record_hand_detected(0.90)  # HIGH confidence
        self.assertTrue(status.get_gesture_permission())


class TestAccidentalActionPrevention(unittest.TestCase):
    """Tests for §53 Accidental Action Prevention."""

    def test_click_thresholds_by_confidence(self):
        """Click detection thresholds should adapt to confidence."""
        # HIGH confidence - standard thresholds
        thresholds_high = CONFIDENCE_THRESHOLDS.copy()
        thresholds_high["click_primary"] = 0.50 * 0.9  # 10% stricter
        
        # LOW confidence - stricter thresholds
        thresholds_low = CONFIDENCE_THRESHOLDS.copy()
        thresholds_low["click_primary"] = 0.50 * 1.5  # 50% stricter
        
        self.assertTrue(thresholds_high["click_primary"] < thresholds_low["click_primary"])


class TestGesturePermissionFiltering(unittest.TestCase):
    """Tests for confidence-based gesture event filtering."""

    def test_low_confidence_filters_gestures(self):
        """Low confidence should filter out high-risk gestures."""
        from airmouse.vision.gestures import GestureEvent
        
        status = TrackingStatus()
        status.record_hand_detected(0.20)  # LOW confidence
        
        # Create test events
        hand = None
        events = [
            GestureEvent(gesture_type=GestureType.LEFT_CLICK, hand=hand),
            GestureEvent(gesture_type=GestureType.RIGHT_CLICK, hand=hand),
            GestureEvent(gesture_type=GestureType.DRAG_START, hand=hand),
            GestureEvent(gesture_type=GestureType.SCROLL_UP, hand=hand),
            GestureEvent(gesture_type=GestureType.PAUSE_TRACKING, hand=hand),
        ]
        
        # All should be blocked
        for event in events:
            should_block = event.gesture_type in (
                GestureType.LEFT_CLICK, GestureType.RIGHT_CLICK,
                GestureType.DRAG_START, GestureType.DRAG_END,
                GestureType.SCROLL_UP, GestureType.SCROLL_DOWN,
                GestureType.PAUSE_TRACKING
            )
            self.assertTrue(should_block)


class TestConfidenceInfoEdgeCases(unittest.TestCase):
    """Tests for ConfidenceInfo edge cases."""

    def test_reset(self):
        """Reset should clear all state."""
        info = ConfidenceInfo()
        info.update(0.90)
        info.reset()
        
        self.assertEqual(info.confidence_value, 0.0)
        self.assertEqual(info.state, ConfidenceState.LOST)
        self.assertEqual(info.frames_in_state, 0)

    def test_update_with_evidence(self):
        """Update should record evidence."""
        info = ConfidenceInfo()
        info.update(0.75, evidence="test_evidence")
        
        self.assertEqual(info.evidence, "test_evidence")


if __name__ == "__main__":
    unittest.main()
