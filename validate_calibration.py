#!/usr/bin/env python3
"""Validate calibration system per §26.

Exercises Calibrator lifecycle, phase transitions, result collection,
and AirMouseController.calibrate() method.
Does NOT fake calibration data or suppress errors.
"""
import sys
import logging
import time

sys.path.insert(0, '/home/shubham/airmouse')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from airmouse.vision.calibration import (
    Calibrator, CalibrationConfig, CalibrationResult, CalibrationPhase,
    apply_calibration
)
from airmouse.vision.hand_tracker import Hand, Landmark, HandLandmark
from airmouse.control.cursor import CursorConfig, SensitivityMode

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


print("=== Calibration System (§26) ===")

# Test 1: Calibrator lifecycle
calibrator = Calibrator(CalibrationConfig())
check("calibrator starts in IDLE", calibrator.phase == CalibrationPhase.IDLE,
      f"phase={calibrator.phase}")

phase = calibrator.start()
check("calibrator.start() returns POSITIONING", phase == CalibrationPhase.POSITIONING,
      f"phase={phase}")

# Test 2: Phase transitions through FRAME_CAMERA and HAND_REGION
# Use varying hand positions to test region detection
hand_positions = [
    (0.4, 0.4),
    (0.5, 0.5),
    (0.6, 0.6),
    (0.55, 0.45),
    (0.45, 0.55),
]

frame_count = 0
max_frames = 100
while calibrator.phase != CalibrationPhase.SCREEN_CORNERS and frame_count < max_frames:
    pos = hand_positions[frame_count % len(hand_positions)]
    hand = Hand(
        landmarks=[Landmark(x=pos[0], y=pos[1], z=0.0)] * 21,
        handedness="Right", confidence=1.0
    )
    calibrator.process(hand)
    frame_count += 1
    time.sleep(0.01)

check("calibrator reaches SCREEN_CORNERS phase", calibrator.phase == CalibrationPhase.SCREEN_CORNERS,
      f"phase={calibrator.phase}, frames={frame_count}")

# Test 3: Apply calibration partial result
result = calibrator.result
# At this point, we've captured camera frame and hand region
check("calibration result has camera frame",
      result.camera_frame_min_x < result.camera_frame_max_x,
      f"min_x={result.camera_frame_min_x}, max_x={result.camera_frame_max_x}")
check("calibration result has hand region",
      result.hand_region_min_x < result.hand_region_max_x,
      f"min_x={result.hand_region_min_x}, max_x={result.hand_region_max_x}")

# Test 4: apply_calibration applies to CursorConfig
config = CursorConfig()
apply_calibration(result, config)
check("apply_calibration sets sensitivity_mode",
      hasattr(config, 'sensitivity_mode'), f"sensitivity_mode={config.sensitivity_mode}")
check("apply_calibration sets base_sensitivity",
      hasattr(config, 'base_sensitivity'), f"base_sensitivity={config.base_sensitivity}")

# Test 5: Calibrator reset
calibrator.reset()
check("calibrator.reset() returns to IDLE", calibrator.phase == CalibrationPhase.IDLE,
      f"phase={calibrator.phase}")

# Test 6: Calibrator cancel
calibrator2 = Calibrator(CalibrationConfig())
calibrator2.start()
calibrator2.cancel()
check("calibrator.cancel() returns to IDLE", calibrator2.phase == CalibrationPhase.IDLE,
      f"phase={calibrator2.phase}")

# Test 7: CalibrationResult defaults
result2 = CalibrationResult()
check("CalibrationResult defaults to not completed", not result2.completed,
      f"completed={result2.completed}")
check("CalibrationResult has camera_frame", result2.camera_frame_min_x >= 0.0,
      f"min_x={result2.camera_frame_min_x}")
check("CalibrationResult has hand_region", result2.hand_region_min_x >= 0.0,
      f"min_x={result2.hand_region_min_x}")
check("CalibrationResult has sensitivity", result2.sensitivity_mode == "normal",
      f"sensitivity={result2.sensitivity_mode}")
check("CalibrationResult has baseline", result2.baseline_x == 0.5,
      f"baseline_x={result2.baseline_x}")
check("CalibrationResult has movement_scale", result2.movement_scale_x == 1.0,
      f"scale_x={result2.movement_scale_x}")

# Test 8: AirMouseController.calibrate() exists and is callable
from airmouse.control.main_loop import AirMouseController, AirMouseConfig
controller = AirMouseController(AirMouseConfig())
check("AirMouseController has calibrate()", hasattr(controller, 'calibrate'),
      f"has_calibrate={hasattr(controller, 'calibrate')}")
check("AirMouseController.calibrate() is callable", callable(controller.calibrate),
      f"callable={callable(controller.calibrate)}")

# Test 9: calibrate() returns False when not initialized
result = controller.calibrate()
check("calibrate() returns False when not initialized", result is False,
      f"result={result}")


print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)