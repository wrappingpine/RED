#!/usr/bin/env python3
"""Test hand tracker module."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from camera.manager import CameraManager, CameraSettings
from vision.hand_tracker import HandTracker, HandTrackerSettings
import cv2

logging.basicConfig(level=logging.DEBUG)

print('Testing Hand Tracker...')

camera = CameraManager()
if not camera.open_camera(CameraSettings(device_index=0, width=640, height=480)):
    print('Failed to open camera')
    sys.exit(1)

print('Camera opened, creating tracker...')
tracker = HandTracker(HandTrackerSettings(max_hands=1))
print('Tracker created, processing frames...')

try:
    frame_count = 0
    while frame_count < 10:
        ret, frame = camera.read_frame()
        print(f'Frame {frame_count}: ret={ret}, frame={frame.shape if frame is not None else None}')
        if not ret:
            break

        hands = tracker.process(frame)
        print(f'  Hands detected: {len(hands)}')

        if hands:
            hand = hands[0]
            print(f'  Hand: {hand.handedness}, Index extended: {hand.is_finger_extended("index")}, Pinch: {hand.is_pinch("thumb", "index"):.3f}, Fist: {hand.is_fist()}')
        else:
            print('  No hand detected')

        frame_count += 1
finally:
    camera.close_camera()
    tracker.close()
print('Done')