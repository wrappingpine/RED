"""Headless test: Full pipeline with actual camera frames."""
import sys
import os
import time
import logging
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

from airmouse.control.main_loop import AirMouseConfig, AirMouseController, AirMouseState
from airmouse.control.cursor import CursorController, SmoothingAlgorithm, SensitivityMode

# Track mouse movements
mouse_moves = []
gesture_events = []

def on_hand_detected(hand):
    print(f"[HAND] {hand.handedness} conf={hand.confidence:.2f} pos=({hand.index_tip.x:.3f}, {hand.index_tip.y:.3f})")

def on_gesture(event):
    gesture_events.append(event)
    print(f"[GESTURE] {event.gesture_type.name}")

def on_frame_processed(frame, hands):
    if hands:
        h = hands[0]
        print(f"[FRAME] {frame.shape} hands={len(hands)} tip=({h.index_tip.x:.3f}, {h.index_tip.y:.3f})")

def on_stats_update(stats):
    if stats.frames_processed % 30 == 0:
        print(f"[STATS] FPS={stats.fps:.1f} frames={stats.frames_processed} hand={stats.hand_detection_time_ms:.1f}ms cursor={stats.cursor_time_ms:.1f}ms gesture={stats.gesture_time_ms:.1f}ms mouse={stats.mouse_time_ms:.1f}ms")

def on_error(msg):
    print(f"[ERROR] {msg}")

# Create config
config = AirMouseConfig()
config.target_fps = 30
config.cursor.dead_zone_radius = 0.005
config.cursor.base_sensitivity = 1.0
config.cursor.sensitivity_mode = SensitivityMode.NORMAL
config.cursor.acceleration = 1.0
config.cursor.smoothing = SmoothingAlgorithm.NONE

# Create controller
controller = AirMouseController(config)
controller.on_hand_detected = on_hand_detected
controller.on_gesture = on_gesture
controller.on_frame_processed = on_frame_processed
controller.on_stats_update = on_stats_update
controller.on_error = on_error

print("=" * 60)
print("AIRMOUSE PIPELINE TEST")
print("=" * 60)

# Initialize
print("\n[INIT] Initializing...")
if not controller.initialize():
    print("[INIT] FAILED - cannot initialize")
    sys.exit(1)
print("[INIT] SUCCESS")

# Start
print("\n[START] Starting processing loop...")
if not controller.start():
    print("[START] FAILED")
    sys.exit(1)
print("[START] RUNNING")

# Run for 10 seconds
print("\n[RUN] Processing frames for 10 seconds...")
time.sleep(10)

# Stop
print("\n[STOP] Stopping...")
controller.stop()
time.sleep(1)

# Report
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"State: {controller.state.value}")
print(f"Frames processed: {controller.stats.frames_processed}")
print(f"Frames dropped: {controller.stats.frames_dropped}")
print(f"FPS: {controller.stats.fps:.1f}")
print(f"Gesture events: {len(gesture_events)}")
for e in gesture_events:
    print(f"  - {e.gesture_type.name}")

if controller.stats.frames_processed > 0:
    print("\n✅ Pipeline is working!")
else:
    print("\n❌ No frames processed - check camera")