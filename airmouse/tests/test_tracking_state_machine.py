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

    def test_stabilizing_reverts_to_lost_on_low_confidence(self):
        """STABILIZING phase with low confidence reverts to LOST."""
        status = TrackingStatus()
        status.phase = TrackingPhase.STABILIZING
        status.stabilization_frames = 2
        status.confidence.state = ConfidenceState.LOW
        status.confidence.confidence_value = 0.2

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])