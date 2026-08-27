"""
Unit tests for OneEuroFilter module.

Tests frequency response and adaptive cutoff behavior.
"""
import sys
sys.path.insert(0, '/home/shubham/airmouse')

import pytest
import numpy as np
import math
from airmouse.control.cursor import OneEuroFilter


class TestOneEuroFilter:
    """Tests for OneEuroFilter class."""

    def test_filter_creation_defaults(self):
        """Test filter creation with default parameters."""
        filter = OneEuroFilter()

        assert filter.min_cutoff == 1.0
        assert filter.beta == 0.0
        assert filter.d_cutoff == 1.0

    def test_filter_creation_custom(self):
        """Test filter creation with custom parameters."""
        filter = OneEuroFilter(min_cutoff=2.0, beta=0.5, d_cutoff=2.0)

        assert filter.min_cutoff == 2.0
        assert filter.beta == 0.5
        assert filter.d_cutoff == 2.0

    def test_filter_initialization(self):
        """Test filter initial state."""
        filter = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)

        # First value should pass through unchanged
        result = filter.filter(5.0, t=0.0)
        assert result == 5.0

        # Second value with same timestamp - dt gets clamped to 1e-3, so some smoothing occurs
        result = filter.filter(10.0, t=0.0)
        # With dt=1e-3 and min_cutoff=1.0, alpha is very small, so result is close to previous (5.0)
        assert result < 10.0  # Some smoothing applied
        assert result > 5.0   # But moved toward 10.0

    def test_filter_smoothing(self):
        """Test basic smoothing behavior."""
        filter = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)

        # Feed a step input
        filter.filter(0.0, t=0.0)
        result = filter.filter(10.0, t=0.01)  # 10ms later

        # Should be somewhat smoothed (not full 10.0)
        assert 0.0 < result < 10.0

    def test_filter_convergence(self):
        """Test that filter converges to steady state."""
        filter = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)

        # Feed constant value
        for i in range(100):
            result = filter.filter(5.0, t=i * 0.01)

        # Should converge to 5.0
        assert abs(result - 5.0) < 0.01

    def test_filter_adaptive_cutoff(self):
        """Test adaptive cutoff based on velocity."""
        # With beta > 0, cutoff increases with velocity
        filter_slow = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
        filter_fast = OneEuroFilter(min_cutoff=1.0, beta=1.0, d_cutoff=1.0)

        # Same input, different velocities
        filter_slow.filter(0.0, t=0.0)
        filter_fast.filter(0.0, t=0.0)

        # High velocity input
        result_slow = filter_slow.filter(10.0, t=0.001)  # Fast change
        result_fast = filter_fast.filter(10.0, t=0.001)

        # With beta > 0, should respond faster (less smoothing)
        assert result_fast > result_slow

    def test_filter_zero_velocity(self):
        """Test filter with zero velocity (constant input)."""
        filter = OneEuroFilter(min_cutoff=1.0, beta=1.0, d_cutoff=1.0)

        # Constant input - velocity is zero
        for i in range(10):
            result = filter.filter(5.0, t=i * 0.01)

        # Should track perfectly with zero velocity
        assert abs(result - 5.0) < 0.001

    def test_filter_noisy_signal(self):
        """Test filter on noisy signal."""
        filter = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)

        np.random.seed(42)
        true_signal = 5.0
        noise_std = 0.5

        results = []
        for i in range(50):
            noisy = true_signal + np.random.normal(0, noise_std)
            result = filter.filter(noisy, t=i * 0.01)
            results.append(result)

        # Filtered signal should have less variance than noisy
        filtered_std = np.std(results)
        # Should reduce noise significantly
        assert filtered_std < noise_std * 0.5

    def test_filter_derivative_cutoff(self):
        """Test derivative cutoff parameter with beta > 0 to see adaptive effect."""
        # With beta > 0, the derivative affects the adaptive cutoff
        filter_low_d = OneEuroFilter(min_cutoff=1.0, beta=1.0, d_cutoff=0.1)
        filter_high_d = OneEuroFilter(min_cutoff=1.0, beta=1.0, d_cutoff=10.0)

        filter_low_d.filter(0.0, t=0.0)
        filter_high_d.filter(0.0, t=0.0)

        # Step input - high velocity
        result_low = filter_low_d.filter(10.0, t=0.01)
        result_high = filter_high_d.filter(10.0, t=0.01)

        # With high d_cutoff, derivative is tracked more closely, higher adaptive cutoff
        # With beta > 0, higher velocity => higher cutoff => less smoothing => result closer to 10
        assert result_high > result_low

    def test_reset(self):
        """Test filter reset functionality."""
        filter = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)

        filter.filter(10.0, t=0.0)
        filter.filter(10.0, t=0.01)
        filter.filter(10.0, t=0.02)

        # Reset
        filter.reset()

        # Next update should pass through
        result = filter.filter(5.0, t=0.1)
        assert result == 5.0

    def test_filter_timestamp_handling(self):
        """Test proper timestamp handling."""
        filter = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)

        # First update
        filter.filter(0.0, t=0.0)

        # Same timestamp - dt gets clamped to 1e-3, so some smoothing occurs
        result = filter.filter(10.0, t=0.0)
        assert result < 10.0  # Some smoothing applied
        assert result > 0.0   # But moved toward 10.0

        # Backwards timestamp - dt gets clamped to 1e-3
        # The filter uses the new timestamp as t_prev, which can cause unexpected behavior
        # Just verify it doesn't crash and produces a valid result
        result = filter.filter(5.0, t=-0.01)
        assert result is not None
        assert not math.isnan(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])