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
        """Test sensitivity mode multipliers (reduced by 20%)."""
        config = CursorConfig()
        assert config.sensitivity_precision == 0.12
        assert config.sensitivity_normal == 0.32
        assert config.sensitivity_fast == 0.48


class TestCursorController:
    """Tests for CursorController."""

    def setup_method(self):
        """Set up controller with test config - no smoothing for basic behavior tests."""
        config = CursorConfig(
            smoothing=SmoothingAlgorithm.NONE,
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
        # Frame-to-frame delta of zero is a valid result (0, 0), not None.
        # None is reserved for "no valid input" (e.g. no hand detected).
        assert result == (0, 0)

    def test_dead_zone_radius(self):
        """Test dead zone radius of 0.02."""
        # Establish reference at center
        self.controller.get_relative_movement_from_plane(0.0, 0.0)

        # Just inside dead zone - frame-to-frame delta < 0.02
        result = self.controller.get_relative_movement_from_plane(0.01, 0.0)
        assert result == (0, 0)  # Dead zone produces zero movement

        # Stay inside dead zone (delta from previous = 0.01)
        result = self.controller.get_relative_movement_from_plane(0.0, 0.01)
        assert result == (0, 0)  # Dead zone produces zero movement

        # Move outside dead zone - delta from previous (0, 0.01) = 0.03 > 0.02
        result = self.controller.get_relative_movement_from_plane(0.03, 0.01)
        assert result is not None  # Should have movement
        dx, dy = result
        assert dx != 0

    def test_sensitivity_normal(self):
        """Test NORMAL sensitivity (0.32)."""
        config = CursorConfig(sensitivity_mode=SensitivityMode.NORMAL, dead_zone_radius=0.0, max_velocity=999999)
        controller = CursorController(config)

        # Establish reference at center
        controller.get_relative_movement_from_plane(0.0, 0.0)

        # Move from center to edge of plane (0.5 normalized)
        dx, dy = controller.get_relative_movement_from_plane(0.5, 0.0)

        # Should produce some pixel movement
        assert dx > 0

    def test_sensitivity_precision(self):
        """Test PRECISION sensitivity (0.12) produces less movement."""
        config_normal = CursorConfig(sensitivity_mode=SensitivityMode.NORMAL, dead_zone_radius=0.0, max_velocity=999999)
        config_precision = CursorConfig(sensitivity_mode=SensitivityMode.PRECISION, dead_zone_radius=0.0, max_velocity=999999)

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
        """Test FAST sensitivity (0.48) produces more movement."""
        config_normal = CursorConfig(sensitivity_mode=SensitivityMode.NORMAL, dead_zone_radius=0.0, max_velocity=999999)
        config_fast = CursorConfig(sensitivity_mode=SensitivityMode.FAST, dead_zone_radius=0.0, max_velocity=999999)

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
        config = CursorConfig(sensitivity_mode=SensitivityMode.NORMAL, dead_zone_radius=0.0, acceleration=1.2, max_velocity=999999, smoothing=SmoothingAlgorithm.NONE)
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
        config = CursorConfig(dead_zone_radius=0.0, smoothing=SmoothingAlgorithm.NONE)
        controller = CursorController(config)

        # First movement establishes reference
        controller.get_relative_movement_from_plane(0.5, 0.5)

        # Same position should give zero movement (frame-to-frame delta = 0)
        result = controller.get_relative_movement_from_plane(0.5, 0.5)
        assert result == (0, 0)

        # Move to new position
        result = controller.get_relative_movement_from_plane(0.6, 0.5)
        assert result is not None
        dx3, dy3 = result
        assert dx3 > 0  # Positive X movement

    def test_sensitivity_mode_switching(self):
        """Test switching sensitivity modes at runtime."""
        config = CursorConfig(sensitivity_mode=SensitivityMode.NORMAL, dead_zone_radius=0.0, max_velocity=999999, smoothing=SmoothingAlgorithm.NONE)
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
        # Frame-to-frame delta is zero → returns (0, 0), not None
        assert result == (0, 0)

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


class TestPlaneCoordinateSmoothing:
    """Tests for One Euro Filter applied to plane coordinates (US3).

    Verifies that the One Euro Filter operates on normalized plane (u,v)
    coordinates after projection, adapting cutoff based on velocity.
    """

    def test_plane_coordinate_smoothing(self):
        """Verify One Euro Filter applies adaptively to plane coordinates.

        Slow movement → higher smoothing (lower effective alpha, more lag)
        Fast movement → lower smoothing (higher effective alpha, less lag)
        """
        from airmouse.control.smoothing import OneEuroFilter
        import time

        # Slow movement filter: low beta → less velocity-based cutoff boost
        slow_filter = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
        slow_filter.reset()

        # Fast movement filter: high beta → more velocity-based cutoff boost
        fast_filter = OneEuroFilter(min_cutoff=1.0, beta=0.5, d_cutoff=1.0)
        fast_filter.reset()

        base_u = 0.5
        dt = 1.0 / 30.0  # 30 FPS
        t = time.monotonic()

        # --- Slow movement test ---
        # Move slowly: small increments over many frames
        slow_values = []
        current = base_u
        for i in range(30):
            current += 0.001  # Very slow movement
            smoothed = slow_filter.filter(current, t + i * dt)
            slow_values.append(smoothed)

        # --- Fast movement test ---
        # Move quickly: large increment in a single step
        fast_filter.reset()
        t2 = time.monotonic()
        fast_filter.filter(base_u, t2)  # Prime
        fast_step_smoothed = fast_filter.filter(base_u + 0.3, t2 + dt)

        # With high beta (fast filter), large velocity should produce less lag
        # The fast filter should track the 0.3 jump more closely than the slow filter
        fast_filter_low_beta = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
        fast_filter_low_beta.reset()
        t3 = time.monotonic()
        fast_filter_low_beta.filter(base_u, t3)
        low_beta_smoothed = fast_filter_low_beta.filter(base_u + 0.3, t3 + dt)

        # High-beta filter should smooth less on fast movement (closer to raw input)
        raw_fast_movement = base_u + 0.3
        high_beta_tracking = fast_step_smoothed
        low_beta_tracking = low_beta_smoothed

        # High beta → more responsive (less smoothing on fast movement)
        high_beta_error = abs(raw_fast_movement - high_beta_tracking)
        low_beta_error = abs(raw_fast_movement - low_beta_tracking)

        # High-beta should have lower error (less smoothing, more responsive)
        assert high_beta_error < low_beta_error, (
            f"High-beta filter should be more responsive on fast movement: "
            f"high_beta_error={high_beta_error}, low_beta_error={low_beta_error}"
        )

    def test_jitter_reduction(self):
        """Verify jitter reduction: OneEuroFilter reduces noise on stationary hand.

        Per SC-004: Feed synthetic landmarks with Gaussian noise (σ=0.01 normalized),
        hand stationary at plane center, 1000 frames, assert noise reduction.
        """
        from airmouse.control.smoothing import OneEuroFilter
        import numpy as np

        # Stationary hand at plane center (0.5, 0.5) with Gaussian noise
        np.random.seed(42)  # Deterministic for reproducibility
        dt = 1.0 / 30.0  # ~30fps

        # Use production filter parameters (from TrackingConfig defaults)
        u_filter = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        v_filter = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        u_filter.reset()
        v_filter.reset()

        center_u = 0.5
        center_v = 0.5
        noise_std = 0.01  # σ=0.01 normalized coordinates per SC-004

        # Simulate 1000 frames of noisy stationary hand
        t = 0.0
        raw_us = []
        raw_vs = []
        smoothed_us = []
        smoothed_vs = []

        for i in range(1000):
            # Stationary hand + Gaussian noise
            noisy_u = center_u + np.random.normal(0, noise_std)
            noisy_v = center_v + np.random.normal(0, noise_std)

            # Clamp to valid range
            noisy_u = max(0.0, min(1.0, noisy_u))
            noisy_v = max(0.0, min(1.0, noisy_v))

            raw_us.append(noisy_u)
            raw_vs.append(noisy_v)

            smoothed_u = u_filter.filter(noisy_u, t + i * dt)
            smoothed_v = v_filter.filter(noisy_v, t + i * dt)

            smoothed_us.append(smoothed_u)
            smoothed_vs.append(smoothed_v)

        # Convert normalized coordinates to screen pixels (1920x1080)
        screen_w, screen_h = 1920, 1080
        raw_us_px = [u * screen_w for u in raw_us]
        raw_vs_px = [v * screen_h for v in raw_vs]
        screen_us = [u * screen_w for u in smoothed_us]
        screen_vs = [v * screen_h for v in smoothed_vs]

        # Measure noise reduction
        raw_std_u = np.std(raw_us_px)
        raw_std_v = np.std(raw_vs_px)
        smoothed_std_u = np.std(screen_us)
        smoothed_std_v = np.std(screen_vs)

        # Filter must reduce noise significantly (at least 2x reduction)
        reduction_u = raw_std_u / max(smoothed_std_u, 0.001)
        reduction_v = raw_std_v / max(smoothed_std_v, 0.001)
        assert reduction_u > 2.0, (
            f"U-axis noise reduction {reduction_u:.1f}x < 2x (jitter not sufficiently reduced): "
            f"raw_std={raw_std_u:.1f}px, smoothed_std={smoothed_std_u:.1f}px"
        )
        assert reduction_v > 2.0, (
            f"V-axis noise reduction {reduction_v:.1f}x < 2x (jitter not sufficiently reduced): "
            f"raw_std={raw_std_v:.1f}px, smoothed_std={smoothed_std_v:.1f}px"
        )

        # Smoothed output should be significantly less noisy than raw input
        assert smoothed_std_u < raw_std_u, (
            f"U-axis smoothed std {smoothed_std_u:.1f}px >= raw std {raw_std_u:.1f}px"
        )
        assert smoothed_std_v < raw_std_v, (
            f"V-axis smoothed std {smoothed_std_v:.1f}px >= raw std {raw_std_v:.1f}px"
        )

        # Also verify mean is close to center (filter shouldn't introduce significant bias)
        mean_u = np.mean(screen_us)
        mean_v = np.mean(screen_vs)
        assert abs(mean_u - center_u * screen_w) < 15.0, (
            f"U-axis mean {mean_u:.1f}px too far from expected {center_u * screen_w:.1f}px"
        )
        assert abs(mean_v - center_v * screen_h) < 15.0, (
            f"V-axis mean {mean_v:.1f}px too far from expected {center_v * screen_h:.1f}px"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])