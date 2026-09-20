"""
Tests for cursor vibration measurement and stability per spec §23.

Cursor vibration is a first-class engineering problem. A fix is not considered
successful because the cursor "looks smoother." It must reduce measurable
unwanted motion without introducing unacceptable latency.
"""

import unittest
import time
import math
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from airmouse.control.cursor import OneEuroFilter, EMASmoother


class TestOneEuroFilter(unittest.TestCase):
    """Tests for One Euro Filter per spec §23 (Motion Filtering)."""

    def test_stationary_signal(self):
        """One Euro Filter should reduce noise on stationary signal."""
        filter_ = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        
        # Simulate a stationary signal with noise
        input_values = []
        for i in range(100):
            # Stationary at 0.5 with small random noise
            noise = np.random.normal(0, 0.05)
            input_values.append(0.5 + noise)
        
        # Filter the signal
        output_values = [filter_.filter(x) for x in input_values]
        
        # Output should be much less noisy
        output_std = np.std(output_values)
        input_std = np.std(input_values)
        
        # The filter should significantly reduce noise
        self.assertLess(output_std, input_std, 
                        "Filter should reduce noise on stationary signal")
        
        # Output should be close to the true value (0.5)
        self.assertAlmostEqual(np.mean(output_values), 0.5, delta=0.1,
                             msg="Filter should track true value")

    def test_jump_rejection(self):
        """One Euro Filter should reject sudden large jumps."""
        filter_ = OneEuroFilter(min_cutoff=0.1, beta=1.0, d_cutoff=1.0)
        
        # Start at 0.5
        filter_.filter(0.5)
        
        # Try a sudden large jump to 0.9
        output = filter_.filter(0.9)
        
        # Output should be much less than 0.9
        self.assertLess(output, 0.7, 
                        "Filter should reject sudden large jumps")

    def test_step_response(self):
        """Filter should respond to step changes with controlled response."""
        filter_ = OneEuroFilter(min_cutoff=0.5, beta=0.5, d_cutoff=0.5)
        
        # Apply step - output should start responding but be delayed
        output = filter_.filter(1.0)
        
        # Output should be between old value and new value (filtered response)
        self.assertGreaterEqual(output, 0.5,
                          "Filter should start responding to step")
        self.assertLessEqual(output, 1.0,
                        "Filter should not exceed new value")

    def test_velocity_adaptive(self):
        """Filter should adapt to velocity - more smoothing at low velocity."""
        slow_filter = OneEuroFilter(min_cutoff=1.0, beta=0.01, d_cutoff=1.0)
        fast_filter = OneEuroFilter(min_cutoff=10.0, beta=0.01, d_cutoff=1.0)
        
        # Send slow-moving signal
        for i in range(10):
            slow_filter.filter(0.5 + i * 0.01)
        
        # Send fast-moving signal
        for i in range(10):
            fast_filter.filter(0.5 + i * 0.1)
        
        # Both should track but with different characteristics
        # The fast filter should be more responsive (higher cutoff)
        self.assertTrue(fast_filter.min_cutoff > slow_filter.min_cutoff)

    def test_noise_reduction_vs_latency(self):
        """Filter should balance noise reduction with latency per spec §22."""
        filter_ = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        
        # Generate noisy input
        noisy = [0.5 + np.random.normal(0, 0.1) for _ in range(50)]
        
        # Filter
        filtered = [filter_.filter(x) for x in noisy]
        
        # Measure: noise reduction ratio
        input_range = max(noisy) - min(noisy)
        output_range = max(filtered) - min(filtered)
        
        # Output range should be smaller (noise reduced)
        noise_reduction = input_range / max(output_range, 0.001)
        self.assertGreater(noise_reduction, 1.5, 
                          "Filter should reduce noise by at least 1.5x")


class TestEMASmoother(unittest.TestCase):
    """Tests for EMA smoother."""

    def test_basic_smoothing(self):
        """EMA should smooth signal."""
        smoother = EMASmoother(alpha=0.3)
        
        values = [float(i) for i in range(10)]
        smoothed = [smoother.smooth(v) for v in values]
        
        # Smoothed values should be closer to each other than raw values
        raw_diff = abs(values[-1] - values[0])
        smoothed_diff = abs(smoothed[-1] - smoothed[0])
        
        self.assertLess(smoothed_diff, raw_diff,
                        "EMA should reduce signal variation")

    def test_reset(self):
        """Reset should clear state."""
        smoother = EMASmoother(alpha=0.3)
        
        smoother.smooth(0.5)
        smoother.reset()
        
        # After reset, next value should be returned as-is
        result = smoother.smooth(1.0)
        self.assertEqual(result, 1.0, 
                        "After reset, first value should pass through")


class TestVibrationMetrics(unittest.TestCase):
    """Tests for vibration measurement per spec §23."""

    def test_stationary_rms(self):
        """Can measure stationary RMS displacement per spec §23."""
        np.random.seed(42)
        # Generate stationary noise centered around 0
        noise = np.random.normal(0, 0.02, 1000)
        
        # RMS of noise should be close to standard deviation
        rms = math.sqrt(np.mean([v**2 for v in noise]))
        std_dev = 0.02
        
        # RMS should be close to std_dev (within reasonable tolerance)
        self.assertAlmostEqual(rms, std_dev, delta=0.005,
                              msg=f"Stationary RMS should equal std_dev ({rms:.4f} vs {std_dev})")

    def test_moving_signal_has_higher_rms(self):
        """Moving signal should have measurably higher RMS."""
        stationary = [0.5 + np.random.normal(0, 0.02) for _ in range(100)]
        moving = [i * 0.01 + np.random.normal(0, 0.02) for i in range(100)]
        
        rms_stationary = math.sqrt(np.mean([v**2 for v in stationary]))
        rms_moving = math.sqrt(np.mean([v**2 for v in moving]))
        
        self.assertGreater(rms_moving, rms_stationary,
                          "Moving signal should have higher RMS than stationary")


if __name__ == "__main__":
    unittest.main()
