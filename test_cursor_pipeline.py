#!/usr/bin/env python3
"""Test cursor pipeline with real hand tracking."""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
from airmouse.config.settings import AirMouseConfig, CursorConfig, TrackingConfig, HandTrackerConfig, CameraSettings
from airmouse.camera.manager import CameraManager
from airmouse.vision.hand_tracker import HandTracker
from airmouse.vision.tracking_processor import TrackingProcessor
from airmouse.control.cursor import CursorController, CursorConfig as CursorCfg
from airmouse.control.input_manager import LinuxInputManager
from airmouse.control.gestures import GestureRecognizer
import time

logging.getLogger('airmouse').setLevel(logging.DEBUG)

print("Testing cursor pipeline...")
config = AirMouseConfig()
camera_cfg = CameraSettings(device_index=0, width=640, height=480, fps=30)
camera = CameraManager()
if not camera.open_camera(camera_cfg):
    print("Camera failed")
    exit(1)

print("Camera opened")

# Init pipeline
tracking_config = TrackingConfig()
hand_tracker = HandTracker()
tracking_processor = TrackingProcessor(tracking_config)
cursor_controller = CursorController()
input_mgr = LinuxInputManager()
input_mgr.initialize()
print(f"Input backend: {input_mgr.get_backend_type()}, healthy: {input_mgr.is_healthy()}")

gesture_recognizer = GestureRecognizer()

frame_count = 0
try:
    while True:
        ret, frame = camera.read_frame()
        if not ret:
            continue
        hands = hand_tracker.process(frame)
        if hands:
            print(f"Hands detected: {len(hands)}")
            tracking_result = tracking_processor.process(hands, frame)
            norm_pos = tracking_processor.get_cursor_position()
            print(f"Norm pos: {norm_pos}")
            rel_movement = cursor_controller.get_relative_movement_from_plane(norm_pos[0], norm_pos[1]) if norm_pos else None
            print(f"Rel movement: {rel_movement}")
            if rel_movement and rel_movement != (0,0):
                print(f"MOVING CURSOR: {rel_movement}")
                ok = input_mgr.move(*rel_movement)
                print(f"Move result: {ok}")
        frame_count += 1
        if frame_count > 100:
            break
        time.sleep(0.03)
finally:
    camera.close_camera()
    input_mgr.cleanup()