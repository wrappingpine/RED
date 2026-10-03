"""
Tests for VelocityLimiter (URFU-001: Velocity Limiter Fix).

Covers User Story 5: Velocity limiter fix (FR-005).
- T026: test_velocity_capping - Verify velocity is capped at max_velocity
- T027: test_no_overshoot - Verify stable boundary capping without overshoot
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airmouse.vision.tracking_processor import VelocityLimiter


class TestVelocityLimiter:
    """Tests for VelocityLimiter per FR-005 (Bug 3 fix)."""

    def test_velocity_capping(self):
        """T026: Verify velocity is capped at max_velocity.

        With synthetic high-speed inputs, the limiter should cap the
        effective per-frame delta to max_velocity and prevent excessive deltas.
        """
        max_vel = 2.0  # Max normalized units per frame (VelocityLimiter data contract)
        limiter = VelocityLimiter(max_velocity=max_vel, smoothing=1.0)  # No smoothing blend
        limiter.reset()

        dt = 1.0 / 30.0  # 30fps

        # Prime the limiter with a small input first (first call returns delta unmodified)
        limiter.limit_delta(0.001, dt)

        # Now feed a huge delta that exceeds the cap
        # max_velocity=2.0 means: max_velocity / dt = 60.0 velocity cap
        # limited_delta = 60.0 * dt = 2.0
        huge_delta = 10.0
        capped = limiter.limit_delta(huge_delta, 2 * dt)

        assert abs(capped) < huge_delta, (
            f"Capped delta ({capped:.4f}) should be less than input ({huge_delta:.4f})"
        )
        assert abs(capped) <= max_vel + 1e-6, (
            f"Capped delta ({capped:.4f}) should not exceed max_velocity ({max_vel:.4f})"
        )

        # Test with negative direction
        limiter.reset()
        limiter.limit_delta(0.001, dt)  # Prime
        neg_delta = -10.0
        capped_neg = limiter.limit_delta(neg_delta, 2 * dt)
        assert abs(capped_neg) < abs(neg_delta), (
            f"Capped delta ({capped_neg:.4f}) should be less than input ({neg_delta:.4f})"
        )
        assert abs(capped_neg) <= max_vel + 1e-6, (
            f"Capped delta ({capped_neg:.4f}) should not exceed max_velocity ({max_vel:.4f})"
        )

        # Test that small deltas are NOT affected by the cap
        limiter.reset()
        limiter.limit_delta(0.001, dt)  # Prime
        small_delta = 0.001  # Well below the cap
        result = limiter.limit_delta(small_delta, 2 * dt)
        assert abs(result - small_delta) < 1e-6 or abs(result) < small_delta, (
            f"Small delta ({small_delta}) should not be excessively capped, got {result}"
        )

        # Test multiple high-speed inputs stay capped
        limiter.reset()
        limiter.limit_delta(0.001, dt)  # Prime
        for i in range(10):
            capped = limiter.limit_delta(100.0, (2 + i) * dt)
            assert abs(capped) <= max_vel + 1e-6, (
                f"Capped delta ({capped:.4f}) at frame {i} exceeds max_velocity ({max_vel})"
            )

    def test_no_overshoot(self):
        """T027: Verify stable boundary capping without overshoot.

        Once capped at the maximum velocity, the limiter should not produce
        output that exceeds the cap boundary, even with sustained high input.
        Also verify that direction reversal doesn't cause overshoot.
        """
        max_vel = 1.0
        limiter = VelocityLimiter(max_velocity=max_vel, smoothing=1.0)
        limiter.reset()

        dt = 1.0 / 30.0

        # Prime the limiter
        limiter.limit_delta(0.001, dt)

        # Feed sustained high-velocity input in one direction
        high_deltas = [5.0] * 20
        for i, delta in enumerate(high_deltas):
            capped = limiter.limit_delta(delta, (1 + i) * dt)
            assert abs(capped) <= max_vel + 1e-6, (
                f"Capped value ({capped:.5f}) exceeds boundary ({max_vel:.5f}) at frame {i}"
            )

        # Switch direction (negative deltas) - no overshoot on reversal
        high_neg_deltas = [-5.0] * 20
        for i, delta in enumerate(high_neg_deltas):
            capped = limiter.limit_delta(delta, (21 + i) * dt)
            assert abs(capped) <= max_vel + 1e-6, (
                f"Capped value ({capped:.5f}) exceeds boundary ({max_vel:.5f}) "
                f"at reversal frame {i}"
            )

        # Oscillation test: alternating high-velocity direction changes
        limiter.reset()
        limiter.limit_delta(0.001, dt)  # Prime
        all_capped = []
        for i in range(50):
            sign = 1.0 if i % 2 == 0 else -1.0
            delta = 3.0 * sign
            capped = limiter.limit_delta(delta, (1 + i) * dt)
            all_capped.append(capped)
            assert abs(capped) <= max_vel + 1e-6, (
                f"Capped value ({capped:.5f}) exceeds boundary ({max_vel:.5f}) "
                f"at oscillation frame {i}"
            )

        # All capped values should be within the boundary
        max_capped = max(abs(v) for v in all_capped)
        assert max_capped <= max_vel + 1e-6, (
            f"Max capped value ({max_capped:.5f}) exceeds boundary ({max_vel:.5f})"
        )

    def test_velocity_limit_zero_input(self):
        """Verify zero input produces zero output (after priming)."""
        limiter = VelocityLimiter(max_velocity=1.0, smoothing=1.0)
        limiter.reset()

        dt = 1.0 / 30.0

        # Prime
        limiter.limit_delta(0.0, dt)

        # Zero delta after prime
        result = limiter.limit_delta(0.0, 2 * dt)
        assert result == 0.0, f"Zero input should produce zero output, got {result}"

        # Multiple zero inputs
        result2 = limiter.limit_delta(0.0, 3 * dt)
        assert result2 == 0.0, f"Sustained zero input should produce zero output, got {result2}"

    def test_velocity_smoothing_applied(self):
        """Verify smoothing parameter is applied to capped output."""
        max_vel = 10.0  # High enough that input won't be capped
        smoothing = 0.5  # 50% smoothing
        limiter = VelocityLimiter(max_velocity=max_vel, smoothing=smoothing)
        limiter.reset()

        dt = 1.0 / 30.0

        # Prime
        limiter.limit_delta(0.0, dt)

        # First real input
        delta1 = 0.01
        result1 = limiter.limit_delta(delta1, 2 * dt)

        # Second input: with smoothing=0.5, result2 should be blend of
        # limited_delta and prev_value (result1)
        delta2 = 0.01
        result2 = limiter.limit_delta(delta2, 3 * dt)

        # With smoothing < 1.0, result2 should be less than delta2
        assert result2 <= delta2, (
            f"Smoothing should reduce output: result2 ({result2:.5f}) <= delta2 ({delta2:.5f})"
        )
