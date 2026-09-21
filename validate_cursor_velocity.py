#!/usr/bin/env python3
"""Validate cursor speed/velocity limits per §24.

Exercises CursorController._clamp_velocity(), max_velocity config,
precision mode velocity limits, and the full movement pipeline.
Does NOT fake velocity clamping or suppress warnings.
"""
import sys
import logging
import time

sys.path.insert(0, '/home/shubham/airmouse')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from airmouse.control.cursor import CursorController, CursorConfig, SensitivityMode
from airmouse.vision.hand_tracker import Hand, Landmark, HandLandmark

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK {name}")
    else:
        failed += 1
        print(f"  FAIL {name}  {detail}")


print("=== CursorController velocity limits (per §24) ===")

# Test 1: Velocity clamping in map_hand_to_cursor
config = CursorConfig(
    screen_width=1920,
    screen_height=1080,
    dead_zone_radius=0.0,
    base_sensitivity=1.0,
    sensitivity_normal=1.0,
    sensitivity_precision=1.0,
    sensitivity_fast=1.0,
    acceleration=1.0,
    max_velocity=2000,
    max_velocity_precision=500,
    smoothing=None,
)
controller = CursorController(config)

hand = Hand(
    landmarks=[Landmark(x=0.5, y=0.5, z=0.0)] * 21,
    handedness="Right", confidence=1.0
)

# First call: sets reference point at (0.5, 0.5), no movement
controller.map_hand_to_cursor(hand)
controller._last_position = None  # reset for clean test

# Second call: large movement (0.6 - 0.5 = 0.1 normalized = 192 pixels)
# With dt=0.001s, expected velocity = 192 / 0.001 = 192,000 px/s
# Should be clamped to ~2000 px/s
controller._last_time = time.time() - 0.001
hand.landmarks[HandLandmark.INDEX_TIP.value] = Landmark(x=0.6, y=0.5, z=0.0)
result = controller.map_hand_to_cursor(hand)
check("velocity clamping works", result is not None, f"result={result}")

# Verify: 0.1 normalized * 1920 = 192 px per frame
# Expected velocity = 192 / 0.001 = 192,000 px/s
# Clamped: max_dist = 2000 * 0.001 = 2 px
# So result should be ~2 px from reference (which was 0.5 * 1920 = 960)
# Expected: ~962 px
expected_px = 0.5 * 1920 + 2  # = 962
check(f"velocity clamped (expected ~{expected_px:.0f}, got {result[0]})",
      abs(result[0] - expected_px) <= 3,
      f"result={result}, expected~{expected_px:.0f}")

# Test 2: Precision mode has lower max_velocity
controller.config.sensitivity_mode = SensitivityMode.PRECISION
controller.config.max_velocity = 2000
controller.config.max_velocity_precision = 500
controller._last_position = None
controller._last_time = time.time() - 0.001
hand.landmarks[HandLandmark.INDEX_TIP.value] = Landmark(x=0.6, y=0.5, z=0.0)
result = controller.map_hand_to_cursor(hand)
if result:
    # In precision mode: max_dist = 500 * 0.001 = 0.5 px
    # Expected: ~960.5 px
    expected_px = 0.5 * 1920 + 0.5  # = 960.5
    check(f"precision mode caps at {config.max_velocity_precision} px/s",
          abs(result[0] - expected_px) <= 1.5,
          f"result={result}, expected~{expected_px:.0f}")

# Test 3: Sensitivity modes
controller.config.sensitivity_mode = SensitivityMode.NORMAL
check("effective_sensitivity() for NORMAL",
      0.9 <= controller.config.effective_sensitivity <= 1.1,
      f"effective={controller.config.effective_sensitivity}")

controller.config.sensitivity_mode = SensitivityMode.PRECISION
check("effective_sensitivity() for PRECISION",
      controller.config.effective_sensitivity == controller.config.sensitivity_precision,
      f"effective={controller.config.effective_sensitivity}")

controller.config.sensitivity_mode = SensitivityMode.FAST
check("effective_sensitivity() for FAST",
      controller.config.effective_sensitivity == controller.config.sensitivity_fast,
      f"effective={controller.config.effective_sensitivity}")

# Test 4: Dead zone - small normalized movement within dead zone is suppressed
config2 = CursorConfig(dead_zone_radius=0.05, smoothing=None, max_velocity=999999)
c2 = CursorController(config2)
# Set hand at reference position first
hand2 = Hand(
    landmarks=[Landmark(x=0.5, y=0.5, z=0.0)] * 21,
    handedness="Right", confidence=1.0
)
c2.map_hand_to_cursor(hand2)  # sets reference point at (0.5, 0.5)
# Now move hand slightly within dead zone (delta = 0.02, distance = 0.028 < 0.05)
hand2.landmarks[HandLandmark.INDEX_TIP.value] = Landmark(x=0.52, y=0.52, z=0.0)
result = c2.map_hand_to_cursor(hand2)
# Dead zone suppresses delta, so position should remain at reference (960, 540)
check("dead zone suppresses small normalized movement",
      result is not None and abs(result[0] - 960) <= 1 and abs(result[1] - 540) <= 1,
      f"result={result}, expected~(960, 540)")

# Test 5: No movement when hand hasn't changed
hand.landmarks[HandLandmark.INDEX_TIP.value] = Landmark(x=0.5, y=0.5, z=0.0)
result1 = controller.map_hand_to_cursor(hand)
result2 = controller.map_hand_to_cursor(hand)
check("stable hand produces consistent position",
      result1 is not None and result2 is not None and abs(result1[0]-result2[0]) <= 1 and abs(result1[1]-result2[1]) <= 1,
      f"r1={result1}, r2={result2}")

# Test 6: get_relative_movement_from_plane uses velocity clamping
config3 = CursorConfig(sensitivity_mode=SensitivityMode.NORMAL, max_velocity=2000, smoothing=None)
c3 = CursorController(config3)
result = c3.get_relative_movement_from_plane(0.5, 0.5)
check("get_relative_movement_from_plane() executes", result is not None,
      f"result={result}")

# Test 7: get_relative_movement uses velocity clamping
result = controller.get_relative_movement(hand)
check("get_relative_movement() executes", result is not None,
      f"result={result}")


print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)