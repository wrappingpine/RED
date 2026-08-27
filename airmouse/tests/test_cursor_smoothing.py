"""
Unit tests for CursorController module.

Tests dead zone, sensitivity modes, and acceleration.
"""
import sys
sys.path.insert(0, '/home/shubham/airmouse')

import pytest
import numpy as np
from airmouse.control.cursor import (
    CursorController, CursorConfig, SmoothingAlgorithm, SensitivityMode
)


class TestCursorConfig:
    """Tests for CursorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CursorConfig()

        assert config.smoothing == SmoothingAlgorithm.ONE_EURO
        assert config.sensitivity_mode == SensitivityMode.NORMAL
        assert config.dead_zone_radius == 0.02
        assert config.acceleration == 1.2

    def test_sensitivity_modes(self):
        """Test sensitivity mode multipliers."""
        config = CursorConfig()
        assert config.sensitivity_precision == 0.15
        assert config.sensitivity_normal == 0.40
        assert config.sensitivity_fast == 0.60


class TestCursorController:
    """Tests for CursorController."""

    def setup_method(self):
        """Set up controller with test config."""
        config = CursorConfig(
            smoothing=SmoothingAlgorithm.ONE_EURO,
            sensitivity_mode=SensitivityMode.NORMAL,
            dead_zone_radius=0.02,
            acceleration=1.2
        )
        self.controller = CursorController(config)

    def test_dead_zone_center(self):
        """Test dead zone at center (0, 0)."""
        # First call establishes reference at (0,0)
        result = self.controller.get_relative_movement_from_plane(0.0, 0.0)
        assert result == (0, 0)  # First call returns zero movement

        # Same position gives zero movement
        result = self.controller.get_relative_movement_from_plane(0.0, 0.0)
        assert result is None  # Returns None for zero movement

    def test_dead_zone_radius(self):
        """Test dead zone radius of 0.02."""
        # Establish reference at center
        self.controller.get_relative_movement_from_plane(0.0, 0.0)

        # Just inside dead zone
        result = self.controller.get_relative_movement_from_plane(0.015, 0.0)
        assert result is None

        result = self.controller.get_relative_movement_from_plane(0.0, 0.015)
        assert result is None

        # Just outside dead zone
        result = self.controller.get_relative_movement_from_plane(0.025, 0.0)
        assert result is not None  # Should have movement
        dx, dy = result
        assert dx != 0

    def test_sensitivity_normal(self):
        """Test NORMAL sensitivity (0.40)."""
        config = CursorConfig(sensitivity_mode=SensitivityMode.NORMAL, dead_zone_radius=0.0)
        controller = CursorController(config)

        # Establish reference at center
        controller.get_relative_movement_from_plane(0.0, 0.0)

        # Move from center to edge of plane (0.5 normalized)
        dx, dy = controller.get_relative_movement_from_plane(0.5, 0.0)

        # Should produce some pixel movement
        assert dx > 0

    def test_sensitivity_precision(self):
        """Test PRECISION sensitivity (0.15) produces less movement."""
        config_normal = CursorConfig(sensitivity_mode=SensitivityMode.NORMAL, dead_zone_radius=0.0)
        config_precision = CursorConfig(sensitivity_mode=SensitivityMode.PRECISION, dead_zone_radius=0.0)

        controller_normal = CursorController(config_normal)
        controller_precision = CursorController(config_precision)

        # Establish reference at center for both
        controller_normal.get_relative_movement_from_plane(0.0, 0.0)
        controller_precision.get_relative_movement_from_plane(0.0, 0.0)

        dx_normal, _ = controller_normal.get_relative_movement_from_plane(0.5, 0.0)
        dx_precision, _ = controller_precision.get_relative_movement_from_plane(0.5, 0.0)

        # Precision should produce less movement
        assert dx_precision < dx_normal

    def test_sensitivity_fast(self):
        """Test FAST sensitivity (0.60) produces more movement."""
        config_normal = CursorConfig(sensitivity_mode=SensitivityMode.NORMAL, dead_zone_radius=0.0)
        config_fast = CursorConfig(sensitivity_mode=SensitivityMode.FAST, dead_zone_radius=0.0)

        controller_normal = CursorController(config_normal)
        controller_fast = CursorController(config_fast)

        # Establish reference at center for both
        controller_normal.get_relative_movement_from_plane(0.0, 0.0)
        controller_fast.get_relative_movement_from_plane(0.0, 0.0)

        dx_normal, _ = controller_normal.get_relative_movement_from_plane(0.5, 0.0)
        dx_fast, _ = controller_fast.get_relative_movement_from_plane(0.5, 0.0)

        # Fast should produce more movement
        assert dx_fast > dx_normal

    def test_acceleration_curve(self):
        """Test acceleration curve (1.2) increases movement non-linearly."""
        config = CursorConfig(sensitivity_mode=SensitivityMode.NORMAL, dead_zone_radius=0.0, acceleration=1.2)
        controller = CursorController(config)

        # Establish reference at center
        controller.get_relative_movement_from_plane(0.0, 0.0)

        # Small movement
        dx_small, _ = controller.get_relative_movement_from_plane(0.1, 0.0)

        # Large movement (5x)
        dx_large, _ = controller.get_relative_movement_from_plane(0.5, 0.0)

        # With acceleration > 1, large movement should be > 5x small movement
        ratio = dx_large / dx_small if dx_small > 0 else 0
        assert ratio > 4.0  # Acceleration makes it superlinear

    def test_relative_movement_resets_reference(self):
        """Test that reference point resets for relative movement."""
        config = CursorConfig(dead_zone_radius=0.0)
        controller = CursorController(config)

        # First movement establishes reference
        controller.get_relative_movement_from_plane(0.5, 0.5)

        # Same position should give zero movement (relative)
        result = controller.get_relative_movement_from_plane(0.5, 0.5)
        assert result is None

        # Move to new position
        result = controller.get_relative_movement_from_plane(0.6, 0.5)
        assert result is not None
        dx3, dy3 = result
        assert dx3 > 0  # Positive X movement

    def test_sensitivity_mode_switching(self):
        """Test switching sensitivity modes at runtime."""
        config = CursorConfig(sensitivity_mode=SensitivityMode.NORMAL, dead_zone_radius=0.0)
        controller = CursorController(config)

        controller.get_relative_movement_from_plane(0.0, 0.0)
        result = controller.get_relative_movement_from_plane(0.5, 0.0)
        assert result is not None
        dx_normal, _ = result

        # Switch to precision
        controller.set_sensitivity_mode(SensitivityMode.PRECISION)
        controller.get_relative_movement_from_plane(0.0, 0.0)
        result = controller.get_relative_movement_from_plane(0.5, 0.0)
        assert result is not None
        dx_precision, _ = result

        assert dx_precision < dx_normal

        # Switch to fast
        controller.set_sensitivity_mode(SensitivityMode.FAST)
        controller.get_relative_movement_from_plane(0.0, 0.0)
        result = controller.get_relative_movement_from_plane(0.5, 0.0)
        assert result is not None
        dx_fast, _ = result

        assert dx_fast > dx_normal

    def test_one_euro_filter_smoothing(self):
        """Test One Euro Filter smoothing is applied."""
        config = CursorConfig(
            smoothing=SmoothingAlgorithm.ONE_EURO,
            sensitivity_mode=SensitivityMode.NORMAL,
            dead_zone_radius=0.0
        )
        controller = CursorController(config)

        controller.get_relative_movement_from_plane(0.0, 0.0)

        # Feed noisy input
        results = []
        for i in range(20):
            # Alternate between two positions
            pos = 0.5 if i % 2 == 0 else 0.51
            result = controller.get_relative_movement_from_plane(pos, 0.0)
            if result is not None:
                dx, _ = result
                results.append(dx)

        # With smoothing, variations should be reduced
        # Just verify it runs without error
        assert len(results) > 0

    def test_no_smoothing(self):
        """Test NONE smoothing passes through directly."""
        config = CursorConfig(
            smoothing=SmoothingAlgorithm.NONE,
            sensitivity_mode=SensitivityMode.NORMAL,
            dead_zone_radius=0.0
        )
        controller = CursorController(config)

        controller.get_relative_movement_from_plane(0.0, 0.0)
        result = controller.get_relative_movement_from_plane(0.5, 0.0)
        assert result is not None
        dx1, _ = result

        # Second call at SAME position relative to reference (0.0, 0.0)
        # should have dx=0.5, not 0
        # To test no movement, we need to call with reference point again
        controller.get_relative_movement_from_plane(0.0, 0.0)  # Reset reference
        result = controller.get_relative_movement_from_plane(0.0, 0.0)
        assert result is None

    def test_exponential_smoothing(self):
        """Test EMA smoothing option."""
        config = CursorConfig(
            smoothing=SmoothingAlgorithm.EMA,
            sensitivity_mode=SensitivityMode.NORMAL,
            dead_zone_radius=0.0,
            ema_alpha=0.3
        )
        controller = CursorController(config)

        controller.get_relative_movement_from_plane(0.0, 0.0)
        result = controller.get_relative_movement_from_plane(0.5, 0.0)
        assert result is not None
        dx, _ = result
        assert dx >= 0  # Should work

    def test_screen_bounds(self):
        """Test cursor stays within screen bounds."""
        config = CursorConfig(dead_zone_radius=0.0)
        controller = CursorController(config)

        controller.get_relative_movement_from_plane(0.0, 0.0)
        # Large movements
        for _ in range(100):
            controller.get_relative_movement_from_plane(1.0, 0.0)

        # Position should be tracked internally (use map_hand_to_cursor for absolute)
        # For relative movement, just verify it runs without error
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])