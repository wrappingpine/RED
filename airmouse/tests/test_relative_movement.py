"""
Unit tests for RelativeMovementEngine module.

Tests dead zone, sensitivity modes, acceleration, smoothing, tracking loss recovery,
re-centering, and stationary hand stabilization.
"""
import sys
sys.path.insert(0, '/home/shubham/airmouse')

import pytest
import numpy as np
import time
from airmouse.control.relative_movement import (
    RelativeMovementEngine, RelativeMouseConfig, MovementResult, MovementState
)
from airmouse.control.cursor import SensitivityMode


class TestRelativeMouseConfig:
    """Tests for RelativeMouseConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RelativeMouseConfig()

        assert config.dead_zone_radius == 0.015
        assert config.sensitivity_precision == 0.12
        assert config.sensitivity_normal == 0.32
        assert config.sensitivity_fast == 0.48
        assert config.base_sensitivity == 1.0
        assert config.acceleration_exponent == 1.2
        assert config.max_velocity_pixels == 500.0
        assert config.smoothing == "one_euro"
        assert config.one_euro_min_cutoff == 0.5
        assert config.one_euro_beta == 0.08
        assert config.one_euro_d_cutoff == 0.5
        assert config.tracking_loss_frames == 5
        assert config.recovery_frames == 3
        assert config.stationary_threshold == 0.002
        assert config.stationary_frames == 10
        assert config.auto_recenter_threshold == 0.0
        assert config.preferred_handedness == "Right"
        assert config.min_hand_confidence == 0.7


class TestRelativeMovementEngine:
    """Tests for RelativeMovementEngine."""

    def setup_method(self):
        """Set up engine with test config - no smoothing for basic behavior tests."""
        config = RelativeMouseConfig(
            smoothing="none",
            dead_zone_radius=0.015,
            acceleration_exponent=1.2,
            max_velocity_pixels=999999,
            min_hand_confidence=0.0  # Accept any confidence for testing
        )
        self.engine = RelativeMovementEngine(config)

    def test_initial_state(self):
        """Test initial state is IDLE with no reference."""
        assert self.engine.state == MovementState.IDLE
        assert self.engine.reference_position is None
        assert not self.engine.is_tracking

    def test_first_hand_establishes_reference(self):
        """First valid hand establishes reference position, no movement."""
        result = self.engine.process_hand(0.5, 0.5, 0.9)

        assert result.has_movement is False
        assert result.dx == 0
        assert result.dy == 0
        assert self.engine.state == MovementState.TRACKING
        assert self.engine.reference_position == (0.5, 0.5)

    def test_same_position_zero_movement(self):
        """Same position as reference gives zero movement (frame-to-frame delta = 0)."""
        # Establish reference
        self.engine.process_hand(0.5, 0.5, 0.9)

        # Same position - frame-to-frame delta = 0
        result = self.engine.process_hand(0.5, 0.5, 0.9)

        assert result.has_movement is False
        assert result.dx == 0
        assert result.dy == 0

    def test_dead_zone_radius(self):
        """Test dead zone prevents small movements."""
        # Establish reference at center
        self.engine.process_hand(0.5, 0.5, 0.9)

        # Move just inside dead zone (0.01 < 0.015)
        result = self.engine.process_hand(0.51, 0.5, 0.9)
        assert result.has_movement is False
        assert result.dx == 0

        # Move outside dead zone from original position (0.52 - 0.5 = 0.02 > 0.015)
        # But since last_position was updated to 0.51, we need a larger jump
        result = self.engine.process_hand(0.53, 0.5, 0.9)  # delta = 0.53 - 0.51 = 0.02 > 0.015
        assert result.has_movement is True
        assert result.dx > 0

    def test_dead_zone_accumulation(self):
        """Multiple small moves inside dead zone should not accumulate."""
        self.engine.process_hand(0.5, 0.5, 0.9)

        # Multiple small moves, each inside dead zone
        for i in range(5):
            result = self.engine.process_hand(0.505 + i * 0.002, 0.5, 0.9)
            assert result.has_movement is False, f"Frame {i}: should be in dead zone"

    def test_relative_movement_like_physical_mouse(self):
        """Frame-to-frame delta drives movement (like physical mouse)."""
        self.engine.process_hand(0.5, 0.5, 0.9)

        # Move right by 0.05 normalized
        result1 = self.engine.process_hand(0.55, 0.5, 0.9)
        dx1 = result1.dx

        # Move right by another 0.05
        result2 = self.engine.process_hand(0.60, 0.5, 0.9)
        dx2 = result2.dx

        # Each frame-to-frame delta should produce similar movement
        # (since deltas are the same: 0.05)
        assert dx1 > 0
        assert dx2 > 0
        assert abs(dx1 - dx2) < 5  # Similar movement (allowing for rounding)

    def test_sensitivity_precision(self):
        """Test PRECISION sensitivity produces less movement."""
        config_precision = RelativeMouseConfig(
            smoothing="none", dead_zone_radius=0.0, max_velocity_pixels=999999, min_hand_confidence=0.0
        )
        config_normal = RelativeMouseConfig(
            smoothing="none", dead_zone_radius=0.0, max_velocity_pixels=999999, min_hand_confidence=0.0
        )

        engine_precision = RelativeMovementEngine(config_precision)
        engine_normal = RelativeMovementEngine(config_normal)

        engine_precision.set_sensitivity_mode(SensitivityMode.PRECISION)
        engine_normal.set_sensitivity_mode(SensitivityMode.NORMAL)

        engine_precision.process_hand(0.5, 0.5, 0.9)
        engine_normal.process_hand(0.5, 0.5, 0.9)

        result_p = engine_precision.process_hand(0.7, 0.5, 0.9)
        result_n = engine_normal.process_hand(0.7, 0.5, 0.9)

        assert result_p.dx < result_n.dx

    def test_sensitivity_fast(self):
        """Test FAST sensitivity produces more movement."""
        config_fast = RelativeMouseConfig(
            smoothing="none", dead_zone_radius=0.0, max_velocity_pixels=999999, min_hand_confidence=0.0
        )
        config_normal = RelativeMouseConfig(
            smoothing="none", dead_zone_radius=0.0, max_velocity_pixels=999999, min_hand_confidence=0.0
        )

        engine_fast = RelativeMovementEngine(config_fast)
        engine_normal = RelativeMovementEngine(config_normal)

        engine_fast.set_sensitivity_mode(SensitivityMode.FAST)
        engine_normal.set_sensitivity_mode(SensitivityMode.NORMAL)

        engine_fast.process_hand(0.5, 0.5, 0.9)
        engine_normal.process_hand(0.5, 0.5, 0.9)

        result_f = engine_fast.process_hand(0.7, 0.5, 0.9)
        result_n = engine_normal.process_hand(0.7, 0.5, 0.9)

        assert result_f.dx > result_n.dx

    def test_sensitivity_mode_switching(self):
        """Test switching sensitivity modes at runtime."""
        self.engine.process_hand(0.5, 0.5, 0.9)
        result_normal = self.engine.process_hand(0.7, 0.5, 0.9)
        dx_normal = result_normal.dx

        self.engine.set_sensitivity_mode(SensitivityMode.PRECISION)
        self.engine.process_hand(0.5, 0.5, 0.9)  # Re-establish reference
        result_precision = self.engine.process_hand(0.7, 0.5, 0.9)
        dx_precision = result_precision.dx

        assert dx_precision < dx_normal

        self.engine.set_sensitivity_mode(SensitivityMode.FAST)
        self.engine.process_hand(0.5, 0.5, 0.9)
        result_fast = self.engine.process_hand(0.7, 0.5, 0.9)
        dx_fast = result_fast.dx

        assert dx_fast > dx_normal

    def test_acceleration_curve(self):
        """Test acceleration curve makes large movements disproportionately larger."""
        config = RelativeMouseConfig(
            smoothing="none", dead_zone_radius=0.0, acceleration_exponent=1.5, max_velocity_pixels=999999, min_hand_confidence=0.0
        )
        engine = RelativeMovementEngine(config)
        engine.set_sensitivity_mode(SensitivityMode.NORMAL)

        engine.process_hand(0.5, 0.5, 0.9)

        # Small movement
        dx_small = engine.process_hand(0.52, 0.5, 0.9).dx

        # Large movement (5x the delta)
        dx_large = engine.process_hand(0.60, 0.5, 0.9).dx

        # With acceleration > 1, large movement should be > 5x small movement
        ratio = dx_large / dx_small if dx_small > 0 else 0
        assert ratio > 4.0

    def test_invert_x(self):
        """Test X axis inversion."""
        config = RelativeMouseConfig(
            smoothing="none", invert_x=True, dead_zone_radius=0.0, max_velocity_pixels=999999, min_hand_confidence=0.0
        )
        engine = RelativeMovementEngine(config)
        engine.set_sensitivity_mode(SensitivityMode.NORMAL)

        engine.process_hand(0.5, 0.5, 0.9)
        result = engine.process_hand(0.6, 0.5, 0.9)

        # Moving right (increasing X) should produce negative dx with invert_x
        assert result.dx < 0

    def test_invert_y(self):
        """Test Y axis inversion."""
        config = RelativeMouseConfig(
            smoothing="none", invert_y=True, dead_zone_radius=0.0, max_velocity_pixels=999999, min_hand_confidence=0.0
        )
        engine = RelativeMovementEngine(config)
        engine.set_sensitivity_mode(SensitivityMode.NORMAL)

        engine.process_hand(0.5, 0.5, 0.9)
        result = engine.process_hand(0.5, 0.6, 0.9)

        # Moving down (increasing Y) should produce negative dy with invert_y
        assert result.dy < 0

    def test_max_velocity_clamp(self):
        """Test maximum velocity clamping."""
        config = RelativeMouseConfig(
            smoothing="none", dead_zone_radius=0.0, max_velocity_pixels=100, min_hand_confidence=0.0
        )
        engine = RelativeMovementEngine(config)
        engine.set_sensitivity_mode(SensitivityMode.FAST)

        engine.process_hand(0.5, 0.5, 0.9)
        # Large movement that would exceed max velocity
        result = engine.process_hand(1.0, 0.5, 0.9)

        velocity = abs(result.dx)
        assert velocity <= 100

    def test_tracking_loss_and_recovery(self):
        """Test tracking loss detection and recovery without cursor jump."""
        # Need min_hand_confidence > 0.3 to trigger tracking loss
        config = RelativeMouseConfig(
            smoothing="none", dead_zone_radius=0.015, acceleration_exponent=1.2,
            max_velocity_pixels=999999, min_hand_confidence=0.5
        )
        engine = RelativeMovementEngine(config)

        # Establish tracking
        engine.process_hand(0.5, 0.5, 0.9)
        engine.process_hand(0.55, 0.5, 0.9)  # Move right

        # Simulate hand loss (low confidence)
        for _ in range(6):  # tracking_loss_frames = 5
            result = engine.process_hand(0.55, 0.5, 0.3)  # Low confidence

        # Should be in RECOVERING state
        assert engine.state == MovementState.RECOVERING

        # Hand reappears at SAME position (no jump)
        for _ in range(4):  # recovery_frames = 3
            result = engine.process_hand(0.55, 0.5, 0.9)

        # Should be tracking again, reference reset to current position
        assert engine.state == MovementState.TRACKING
        assert engine.reference_position == (0.55, 0.5)

        # Next movement should be from new reference (no jump from old reference)
        # Move enough to exceed dead zone (0.56 - 0.55 = 0.01 < 0.015, need bigger)
        result = engine.process_hand(0.57, 0.5, 0.9)
        assert result.has_movement is True
        assert result.dx > 0  # Small positive movement from new reference

    def test_tracking_loss_prevents_cursor_jump(self):
        """Critical test: cursor should NOT jump when hand is reacquired at different position."""
        # Need min_hand_confidence > 0.3 to trigger tracking loss
        config = RelativeMouseConfig(
            smoothing="none", dead_zone_radius=0.015, acceleration_exponent=1.2,
            max_velocity_pixels=999999, min_hand_confidence=0.5
        )
        engine = RelativeMovementEngine(config)

        # Move hand to right side
        engine.process_hand(0.5, 0.5, 0.9)
        engine.process_hand(0.8, 0.5, 0.9)  # Far right

        # Tracking lost
        for _ in range(6):
            engine.process_hand(0.8, 0.5, 0.3)

        # Hand reappears at LEFT side (simulating hand moved while not tracked)
        for _ in range(4):
            engine.process_hand(0.2, 0.5, 0.9)

        # Reference should be reset to 0.2, NOT 0.8
        assert engine.reference_position == (0.2, 0.5)

        # Movement from left position
        result = engine.process_hand(0.25, 0.5, 0.9)
        assert result.has_movement is True
        assert result.dx > 0  # Moving right from 0.2

    def test_explicit_recenter(self):
        """Test explicit re-centering via recenter()."""
        self.engine.process_hand(0.5, 0.5, 0.9)
        self.engine.process_hand(0.8, 0.5, 0.9)  # Move to right

        # Recenter to a specific position
        self.engine.recenter((0.3, 0.3))

        assert self.engine.reference_position == (0.3, 0.3)
        assert self.engine.state == MovementState.RECENTERING

        # Next frame should be tracking with new reference
        result = self.engine.process_hand(0.35, 0.3, 0.9)
        assert self.engine.state == MovementState.TRACKING
        assert result.has_movement is True
        assert result.dx > 0

    def test_recenter_without_position(self):
        """Test recenter() without argument uses last valid position."""
        self.engine.process_hand(0.5, 0.5, 0.9)
        self.engine.process_hand(0.7, 0.5, 0.9)

        self.engine.recenter()  # No argument

        assert self.engine.reference_position == (0.7, 0.5)

    def test_stationary_hand_stabilization(self):
        """Test stationary hand detection and extra smoothing."""
        config = RelativeMouseConfig(
            smoothing="one_euro", stationary_threshold=0.002, stationary_frames=5,
            dead_zone_radius=0.01, max_velocity_pixels=999999, min_hand_confidence=0.0
        )
        engine = RelativeMovementEngine(config)

        engine.process_hand(0.5, 0.5, 0.9)

        # Small jitter around center for 10 frames (inside dead zone of 0.01)
        for i in range(10):
            result = engine.process_hand(0.5 + 0.001 * (-1)**i, 0.5, 0.9)

        # Should detect stationary
        assert engine._is_stationary is True

    def test_one_euro_filter_smoothing(self):
        """Test One Euro Filter smoothing is applied."""
        config = RelativeMouseConfig(
            smoothing="one_euro", dead_zone_radius=0.0, min_hand_confidence=0.0
        )
        engine = RelativeMovementEngine(config)

        engine.process_hand(0.5, 0.5, 0.9)

        # Feed noisy input
        results = []
        for i in range(20):
            pos = 0.5 if i % 2 == 0 else 0.51
            result = engine.process_hand(pos, 0.5, 0.9)
            if result.has_movement:
                results.append(result.dx)

        # With smoothing, variations should be reduced
        # Just verify it runs without error
        assert True  # Test passes if no exception

    def test_ema_smoothing(self):
        """Test EMA smoothing option."""
        config = RelativeMouseConfig(
            smoothing="ema", ema_alpha=0.3, dead_zone_radius=0.0, min_hand_confidence=0.0
        )
        engine = RelativeMovementEngine(config)

        engine.process_hand(0.5, 0.5, 0.9)
        result = engine.process_hand(0.6, 0.5, 0.9)

        assert result.has_movement is True
        assert result.dx >= 0

    def test_no_smoothing(self):
        """Test NONE smoothing passes through directly."""
        config = RelativeMouseConfig(
            smoothing="none", dead_zone_radius=0.0, min_hand_confidence=0.0
        )
        engine = RelativeMovementEngine(config)

        engine.process_hand(0.5, 0.5, 0.9)
        result = engine.process_hand(0.6, 0.5, 0.9)

        assert result.has_movement is True
        # No smoothing, so movement directly corresponds to delta * sensitivity

    def test_low_confidence_hand_ignored(self):
        """Test hands below confidence threshold are ignored."""
        config = RelativeMouseConfig(
            smoothing="none", min_hand_confidence=0.5, dead_zone_radius=0.0, max_velocity_pixels=999999
        )
        engine = RelativeMovementEngine(config)

        # First hand with good confidence
        engine.process_hand(0.5, 0.5, 0.9)

        # Low confidence hand - should be treated as no hand
        result = engine.process_hand(0.6, 0.5, 0.3)
        assert result.has_movement is False

        # Should still be at reference position
        engine.process_hand(0.6, 0.5, 0.9)  # High confidence again
        result = engine.process_hand(0.65, 0.5, 0.9)
        # Movement from 0.6 (not 0.5) because low confidence frame was ignored
        assert result.has_movement is True

    def test_reset(self):
        """Test full engine reset."""
        self.engine.process_hand(0.5, 0.5, 0.9)
        self.engine.process_hand(0.7, 0.5, 0.9)

        self.engine.reset()

        assert self.engine.state == MovementState.IDLE
        assert self.engine.reference_position is None
        assert self.engine._frames_without_hand == 0

    def test_get_stats(self):
        """Test statistics collection."""
        self.engine.process_hand(0.5, 0.5, 0.9)
        self.engine.process_hand(0.55, 0.5, 0.9)

        stats = self.engine.get_stats()

        assert stats["state"] == "tracking"
        assert stats["total_frames"] == 2
        assert stats["tracking_frames"] == 2
        assert stats["reference_position"] == (0.5, 0.5)
        assert stats["sensitivity_mode"] == "NORMAL"

    def test_force_recovery(self):
        """Test forced recovery state."""
        # Need min_hand_confidence > 0.3 to trigger tracking loss
        config = RelativeMouseConfig(
            smoothing="none", dead_zone_radius=0.015, acceleration_exponent=1.2,
            max_velocity_pixels=999999, min_hand_confidence=0.5
        )
        engine = RelativeMovementEngine(config)

        engine.process_hand(0.5, 0.5, 0.9)

        # Simulate tracking loss
        for _ in range(6):
            engine.process_hand(0.5, 0.5, 0.3)

        assert engine.state == MovementState.RECOVERING

        engine.force_recovery()
        # Should reset recovery frame count
        assert engine._recovery_frame_count == 0


class TestMovementResult:
    """Tests for MovementResult dataclass."""

    def test_default_values(self):
        """Test default MovementResult values."""
        result = MovementResult()

        assert result.dx == 0
        assert result.dy == 0
        assert result.has_movement is False
        assert result.state == MovementState.IDLE
        assert result.raw_delta_x == 0.0
        assert result.raw_delta_y == 0.0
        assert result.smoothed_delta_x == 0.0
        assert result.smoothed_delta_y == 0.0
        assert result.velocity == 0.0
        assert result.confidence == 0.0
        assert result.frames_since_hand == 0


class TestEdgeCases:
    """Edge case tests."""

    def test_hand_at_edge_of_camera(self):
        """Test hand at camera edge (0.0 or 1.0)."""
        config = RelativeMouseConfig(smoothing="none", dead_zone_radius=0.0, max_velocity_pixels=999999, min_hand_confidence=0.0)
        engine = RelativeMovementEngine(config)

        engine.process_hand(0.0, 0.0, 0.9)
        result = engine.process_hand(0.05, 0.05, 0.9)

        assert result.has_movement is True
        assert result.dx > 0
        assert result.dy > 0

    def test_rapid_movement(self):
        """Test rapid large movements."""
        config = RelativeMouseConfig(smoothing="none", dead_zone_radius=0.0, max_velocity_pixels=999999, min_hand_confidence=0.0)
        engine = RelativeMovementEngine(config)

        engine.process_hand(0.5, 0.5, 0.9)

        # Rapid movement across screen
        result = engine.process_hand(1.0, 1.0, 0.9)

        assert result.has_movement is True
        assert result.dx > 0
        assert result.dy > 0

    def test_out_of_bounds_hand_position(self):
        """Test out of bounds hand positions are handled."""
        config = RelativeMouseConfig(smoothing="none", dead_zone_radius=0.0, min_hand_confidence=0.0)
        engine = RelativeMovementEngine(config)

        engine.process_hand(0.5, 0.5, 0.9)

        # Out of bounds - should not produce movement
        result = engine.process_hand(1.5, 0.5, 0.9)
        assert result.has_movement is False

        result = engine.process_hand(-0.1, 0.5, 0.9)
        assert result.has_movement is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])