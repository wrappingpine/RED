#!/usr/bin/env python3
"""Test hand tracker module with GUI."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from camera.manager import CameraManager, CameraSettings
from vision.hand_tracker import HandTracker, HandTrackerSettings
import cv2

logging.basicConfig(level=logging.INFO)

print('Testing Hand Tracker with GUI...')
print('Show your hand to the camera. Press q to quit.')

camera = CameraManager()
if not camera.open_camera(CameraSettings(device_index=0, width=640, height=480)):
    print('Failed to open camera')
    sys.exit(1)

print('Camera opened, creating tracker...')
tracker = HandTracker(HandTrackerSettings(max_hands=1))
print('Tracker created, processing frames...')

try:
    frame_count = 0
    while True:
        ret, frame = camera.read_frame()
        if not ret:
            break

        hands = tracker.process(frame)

        if hands:
            hand = hands[0]
            print(f'\rHand: {hand.handedness}, Index extended: {hand.is_finger_extended("index")}, Pinch: {hand.is_pinch("thumb", "index"):.3f}, Fist: {hand.is_fist()}', end='')

            annotated = tracker.draw_landmarks(frame, hands)
        else:
            annotated = frame

        # Add status text
        cv2.putText(annotated, f"Hands: {len(hands)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, "Show your hand to camera", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Hand Tracking Test', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1
finally:
    camera.close_camera()
    tracker.close()
    cv2.destroyAllWindows()
print('\nDone')