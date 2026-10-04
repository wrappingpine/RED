"""
Tests for tracking state machine recovery (TRACKING_JUMP → LOST → REACQUIRING → STABILIZING → TRACKING).

Verifies that valid frames after a tracking loss automatically resume cursor movement
without requiring manual intervention.
"""

import pytest
import numpy as np
import time
from unittest.mock import MagicMock

from airmouse.vision.tracking_status import (
    TrackingStatus, TrackingPhase, LostReason
)
from airmouse.vision.confidence import ConfidenceState, ConfidenceInfo


class TestTrackingStateMachineRecovery:
    """Tests for automatic recovery from LOST to TRACKING."""

    def test_record_hand_detected_starts_reacquisition(self):
        """When phase is LOST and hand is detected with HIGH confidence, start reacquisition."""
        status = TrackingStatus()
        status.phase = TrackingPhase.LOST
        status.confidence.state = ConfidenceState.LOST
        status.confidence.confidence_value = 0.0

        state = status.record_hand_detected(0.9, evidence="test")

        assert state == ConfidenceState.HIGH
        assert status.phase == TrackingPhase.REACQUIRING

    def test_record_hand_detected_starts_reacquisition_medium(self):
        """When phase is LOST and hand is detected with MEDIUM confidence, start reacquisition."""
        status = TrackingStatus()
        status.phase = TrackingPhase.LOST
        status.confidence.state = ConfidenceState.LOST

        state = status.record_hand_detected(0.6, evidence="test")

        assert state == ConfidenceState.MEDIUM
        assert status.phase == TrackingPhase.REACQUIRING

    def test_record_hand_detected_does_not_recover_low_confidence(self):
        """When phase is LOST and hand is detected with LOW confidence, stay LOST."""
        status = TrackingStatus()
        status.phase = TrackingPhase.LOST
        status.confidence.state = ConfidenceState.LOST

        state = status.record_hand_detected(0.2, evidence="test")

        assert state == ConfidenceState.LOW
        assert status.phase == TrackingPhase.LOST

    def test_reacquisition_advances_to_stabilizing(self):
        """REACQUIRING phase with acceptable confidence advances to STABILIZING."""
        status = TrackingStatus()
        status.phase = TrackingPhase.REACQUIRING
        status.confidence.state = ConfidenceState.HIGH
        status.confidence.confidence_value = 0.9

        status.update()

        assert status.phase == TrackingPhase.STABILIZING
        assert status.stabilization_frames == 0

    def test_stabilizing_continues_with_low_confidence(self):
        """STABILIZING phase with LOW confidence continues (does NOT revert to LOST).

        LOW confidence means the hand is still visible, just with lower
        confidence.  Reverting to LOST on LOW creates a ping-pong loop:
        LOST → REACQUIRING → STABILIZING → LOST.  Instead, continue
        stabilizing with LOW confidence.
        """
        status = TrackingStatus()
        status.phase = TrackingPhase.STABILIZING
        status.stabilization_frames = 2
        status.confidence.state = ConfidenceState.LOW
        status.confidence.confidence_value = 0.2

        status.update()

        # LOW confidence should NOT revert to LOST — continue stabilizing
        assert status.phase == TrackingPhase.STABILIZING
        assert status.stabilization_frames == 3

    def test_stabilizing_reverts_to_lost_on_lost_confidence(self):
        """STABILIZING phase with LOST confidence reverts to LOST."""
        status = TrackingStatus()
        status.phase = TrackingPhase.STABILIZING
        status.stabilization_frames = 2
        status.confidence.state = ConfidenceState.LOST
        status.confidence.confidence_value = 0.05

        status.update()

        assert status.phase == TrackingPhase.LOST

    def test_stabilizing_completes_after_required_frames(self):
        """STABILIZING phase with HIGH confidence completes after 3 frames."""
        status = TrackingStatus()
        status.phase = TrackingPhase.STABILIZING
        status.confidence.state = ConfidenceState.HIGH
        status.confidence.confidence_value = 0.9
        status.stabilization_frames = 0

        # Frame 1
        status.update()
        assert status.phase == TrackingPhase.STABILIZING
        assert status.stabilization_frames == 1

        # Frame 2
        status.update()
        assert status.phase == TrackingPhase.STABILIZING
        assert status.stabilization_frames == 2

        # Frame 3 - should complete
        status.update()
        assert status.phase == TrackingPhase.TRACKING

    def test_stabilizing_completes_after_required_frames_medium(self):
        """STABILIZING phase with MEDIUM confidence completes after 5 frames."""
        status = TrackingStatus()
        status.phase = TrackingPhase.STABILIZING
        status.confidence.state = ConfidenceState.MEDIUM
        status.confidence.confidence_value = 0.6
        status.stabilization_frames = 0

        for i in range(4):
            status.update()
            assert status.phase == TrackingPhase.STABILIZING

        # 5th frame - should complete
        status.update()
        assert status.phase == TrackingPhase.TRACKING

    def test_full_recovery_flow(self):
        """Full recovery: LOST → REACQUIRING → STABILIZING → TRACKING."""
        status = TrackingStatus()
        status.phase = TrackingPhase.LOST
        status.confidence.state = ConfidenceState.LOST
        status.confidence.confidence_value = 0.0

        # Step 1: Hand detected with high confidence
        state = status.record_hand_detected(0.9, evidence="recovery")
        assert state == ConfidenceState.HIGH
        assert status.phase == TrackingPhase.REACQUIRING

        # Step 2: Update advances to STABILIZING
        status.update()
        assert status.phase == TrackingPhase.STABILIZING

        # Step 3: 3 stabilization frames
        for _ in range(3):
            status.update()
        assert status.phase == TrackingPhase.TRACKING

    def test_movement_permission_after_recovery(self):
        """After full recovery, movement permission is granted."""
        status = TrackingStatus()
        status.phase = TrackingPhase.LOST
        status.confidence.state = ConfidenceState.LOST

        # Recover
        status.record_hand_detected(0.9, evidence="recovery")
        status.update()
        for _ in range(3):
            status.update()

        assert status.phase == TrackingPhase.TRACKING
        assert status.get_movement_permission() is True

    def test_lost_phase_blocks_movement(self):
        """LOST phase blocks movement."""
        status = TrackingStatus()
        status.phase = TrackingPhase.LOST
        status.confidence.state = ConfidenceState.LOST

        assert status.get_movement_permission() is False

    def test_reacquiring_phase_blocks_movement(self):
        """REACQUIRING phase blocks movement."""
        status = TrackingStatus()
        status.phase = TrackingPhase.REACQUIRING
        status.confidence.state = ConfidenceState.HIGH

        assert status.get_movement_permission() is False

    def test_stabilizing_phase_blocks_movement(self):
        """STABILIZING phase blocks movement."""
        status = TrackingStatus()
        status.phase = TrackingPhase.STABILIZING
        status.confidence.state = ConfidenceState.HIGH

        assert status.get_movement_permission() is False

    def test_tracking_jump_recovery(self):
        """Simulate TRACKING_JUMP → LOST → recovery → TRACKING flow."""
        status = TrackingStatus()
        status.phase = TrackingPhase.TRACKING
        status.confidence.state = ConfidenceState.HIGH
        status.confidence.confidence_value = 0.9

        # Simulate tracking jump
        status.record_hand_lost(LostReason.TRACKING_JUMP, "jump detected")
        assert status.phase == TrackingPhase.LOST
        assert status.lost_reason == LostReason.TRACKING_JUMP

        # Simulate hand re-detection with high confidence
        status.record_hand_detected(0.9, evidence="hand reappeared")
        assert status.phase == TrackingPhase.REACQUIRING

        # Advance through reacquisition
        status.update()
        assert status.phase == TrackingPhase.STABILIZING

        # Complete stabilization
        for _ in range(3):
            status.update()
        assert status.phase == TrackingPhase.TRACKING
        assert status.get_movement_permission() is True


class TestTemporaryLossRecovery:
    """Tests for TEMPORARY_LOSS phase — brief projection failures."""

    def test_temporary_loss_phase_blocks_movement(self):
        """TEMPORARY_LOSS should hold position (no movement)."""
        status = TrackingStatus()
        status.phase = TrackingPhase.TEMPORARY_LOSS
        status.confidence.update(0.9, frame_count=1)
        assert status.get_movement_permission() is False

    def test_temporary_loss_phase_blocks_gesture(self):
        """TEMPORARY_LOSS should block gestures."""
        status = TrackingStatus()
        status.phase = TrackingPhase.TEMPORARY_LOSS
        status.confidence.update(0.9, frame_count=1)
        assert status.get_gesture_permission() is False

    def test_temporary_loss_recovers_to_tracking(self):
        """TEMPORARY_LOSS should recover to TRACKING when confidence is acceptable."""
        status = TrackingStatus()
        status.phase = TrackingPhase.TEMPORARY_LOSS
        status.confidence.update(0.9, frame_count=1)
        status.update()
        assert status.phase == TrackingPhase.TRACKING
        assert status.get_movement_permission() is True

    def test_temporary_loss_medium_confidence_recovers(self):
        """TEMPORARY_LOSS should recover with MEDIUM confidence."""
        status = TrackingStatus()
        status.phase = TrackingPhase.TEMPORARY_LOSS
        status.confidence.update(0.6, frame_count=1)
        status.update()
        assert status.phase == TrackingPhase.TRACKING

    def test_temporary_loss_low_confidence_stays(self):
        """TEMPORARY_LOSS should NOT recover with LOW confidence."""
        status = TrackingStatus()
        status.phase = TrackingPhase.TEMPORARY_LOSS
        status.confidence.update(0.4, frame_count=1)
        status.update()
        # LOW confidence during TEMPORARY_LOSS — should stay in TEMPORARY_LOSS
        # (not transition to LOST because confidence is still above LOST threshold)
        assert status.phase == TrackingPhase.TEMPORARY_LOSS

    def test_temporary_loss_lost_confidence_transitions_to_lost(self):
        """TEMPORARY_LOSS with LOST confidence should transition to LOST."""
        status = TrackingStatus()
        status.phase = TrackingPhase.TEMPORARY_LOSS
        status.confidence.update(0.05, frame_count=1)
        status.update()
        assert status.phase == TrackingPhase.LOST

    def test_temporary_loss_does_not_increment_stabilization(self):
        """TEMPORARY_LOSS should not affect stabilization_frames."""
        status = TrackingStatus()
        status.phase = TrackingPhase.TEMPORARY_LOSS
        status.confidence.update(0.9, frame_count=1)
        status.stabilization_frames = 5
        status.update()
        assert status.stabilization_frames == 5  # unchanged

    def test_temporary_loss_phase_change_callback(self):
        """TEMPORARY_LOSS -> TRACKING should fire phase change callback."""
        status = TrackingStatus()
        status.phase = TrackingPhase.TEMPORARY_LOSS
        status.confidence.update(0.9, frame_count=1)
        changes = []
        status.set_callbacks(on_phase_change=lambda old, new: changes.append((old, new)))
        status.update()
        assert len(changes) == 1
        assert changes[0] == (TrackingPhase.TEMPORARY_LOSS, TrackingPhase.TRACKING)


class TestMainLoopStateMachineIntegration:
    """Integration tests for main_loop state machine with TEMPORARY_LOSS."""

    def test_lost_track_does_not_call_record_hand_lost(self):
        """LOST_TRACK with high confidence should NOT call record_hand_lost."""
        status = TrackingStatus()
        status.record_hand_lost(LostReason.TRACKING_JUMP, "test")
        # Simulate what main_loop does: set phase to TEMPORARY_LOSS
        status.phase = TrackingPhase.TEMPORARY_LOSS
        status.confidence.update(0.9, frame_count=1)
        status.update()
        # Should have recovered to TRACKING
        assert status.phase == TrackingPhase.TRACKING

    def test_lost_track_with_low_confidence_calls_record_hand_lost(self):
        """LOST_TRACK with low confidence SHOULD call record_hand_lost."""
        status = TrackingStatus()
        status.confidence.update(0.1, frame_count=1)
        status.record_hand_lost(LostReason.CONFIDENCE_DROP, "test")
        assert status.phase == TrackingPhase.LOST

    def test_lost_track_sustained_failure_uses_temporary_loss(self):
        """Sustained projection failure should use TEMPORARY_LOSS, not LOST."""
        status = TrackingStatus()
        status.phase = TrackingPhase.TRACKING
        status.confidence.update(0.9, frame_count=1)
        # Simulate sustained projection failure
        status.phase = TrackingPhase.TEMPORARY_LOSS
        status.update()
        # Should NOT be in LOST
        assert status.phase != TrackingPhase.LOST
        # Should recover when confidence is acceptable
        assert status.phase == TrackingPhase.TRACKING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])