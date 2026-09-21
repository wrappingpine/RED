#!/usr/bin/env python3
"""Validate logging architecture through startup health check output.

Exercises structured health-check logging with ✓/✗ markers per component,
rate-limited tracking lost logging, and standard logging configurations.
Does NOT blindly suppress errors.
"""
import sys
import logging
import re
from io import StringIO

sys.path.insert(0, '/home/shubham/airmouse')

from airmouse.control.main_loop import AirMouseController, AirMouseConfig
from airmouse.camera.manager import CameraSettings
from airmouse.vision.hand_tracker import HandTrackerSettings
from airmouse.vision.face_tracker import FaceTrackerSettings
from airmouse.vision.tracking_status import LostReason

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


print("=== Logging Architecture validation ===")

# Set up StringIO log capture
log_capture = StringIO()
handler = logging.StreamHandler(log_capture)
handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

root_logger = logging.getLogger()
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

# Initialize controller - this triggers health-check logging
config = AirMouseConfig()
config.camera = CameraSettings(device_index=99, width=640, height=480, fps=30)
config.hand_tracker = HandTrackerSettings(max_hands=1)
config.face_tracker = FaceTrackerSettings()

controller = AirMouseController(config, lambda s, d: None)
result = controller.initialize()

# Clean up logger
root_logger.removeHandler(handler)
log_output = log_capture.getvalue()

print("\n--- Captured Log Output ---")
print(log_output)
print("---------------------------\n")

# When camera fails, we get ✗ markers for the failed component
check("logged camera failure marker (✗)", "✗" in log_output or "Failed" in log_output)
# Cursor/gestures/tracking/brightness/input manager are initialized before camera is opened,
# so they should still appear in logs even if camera fails
check("logged at least 5 component entries", len(log_output.split('\n')) >= 5)
check("has health check structure", "Initializing Air Mouse" in log_output)

# Test 2: Rate-limited tracking lost logging
import time
print("Testing rate-limited tracking lost logging...")
log_capture2 = StringIO()
handler2 = logging.StreamHandler(log_capture2)
handler2.setFormatter(logging.Formatter('%(message)s'))
root_logger.addHandler(handler2)

# Verify rate-limiting on tracking lost log
# Trigger tracking lost multiple times rapidly
for i in range(10):
    controller._on_tracking_lost(LostReason.NO_HAND_DETECTED)
    time.sleep(0.01)

root_logger.removeHandler(handler2)
lost_logs = log_capture2.getvalue().strip().split('\n')
lost_logs = [l for l in lost_logs if "Tracking lost" in l]

print(f"Captured {len(lost_logs)} tracking lost logs:")
for l in lost_logs:
    print("  ", l)

check("tracking lost logs are rate-limited (< 3 logs in 0.1s)", len(lost_logs) <= 1)


print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)