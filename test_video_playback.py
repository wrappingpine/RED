#!/usr/bin/env python3
"""Playback recorded video with hand and face tracking + virtual display plane."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from airmouse.camera.manager import CameraManager, CameraSettings
from airmouse.vision.hand_tracker import HandTracker, HandTrackerSettings
from airmouse.vision.face_tracker import FaceTracker, FaceTrackerSettings
from airmouse.vision.virtual_plane import VirtualDisplayPlane
from airmouse.vision.head_coords import HeadCoordinateSystem
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

VIDEO_PATH = "/home/shubham/Downloads/2026-08-29 13-55-22.mkv"

print("=" * 50)
print("Air Mouse - Video Playback with Hand & Face Tracking")
print("Press 'q' to quit, 'h' to toggle hand, 'f' to toggle face, 'v' to toggle virtual plane")
print("=" * 50)

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Failed to open video: {VIDEO_PATH}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {width}x{height} @ {fps}fps, {frame_count} frames")

print("Creating trackers...")
hand_tracker = HandTracker(HandTrackerSettings(max_hands=2))
face_tracker = FaceTracker(FaceTrackerSettings(max_faces=1))

# Create virtual display plane
virtual_plane = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25)

print("Processing frames...")

show_hands = True
show_face = True
show_virtual_plane = True

# Approximate camera matrix
h, w = height, width
focal_length = w
camera_matrix = np.array([
    [focal_length, 0, w / 2],
    [0, focal_length, h / 2],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros(4, dtype=np.float32)

frame_idx = 0
paused = False

try:
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue
            frame_idx += 1
        else:
            # Show last frame while paused
            pass

        hands = hand_tracker.process(frame) if show_hands else []
        faces = face_tracker.process(frame) if show_face else []

        annotated = frame.copy()

        # Draw hand landmarks
        if show_hands and hands:
            annotated = hand_tracker.draw_landmarks(annotated, hands)

        # Draw face landmarks and update head coordinate system
        head_coords = None
        ray_origin = None
        ray_direction = None
        intersection = None
        normalized_coords = None

        if show_face and faces:
            face = faces[0]
            annotated = face_tracker.draw_landmarks(annotated, faces, draw_key_points=True, draw_connections=True)

            # Create head coordinate system from face landmarks
            if face.eye_midpoint and face.nose_tip and face.forehead:
                head_coords = HeadCoordinateSystem.from_face(face)

                # Update virtual plane with head coords
                if head_coords.is_valid():
                    virtual_plane.update_head_coords(head_coords)

                    # Compute ray from eye midpoint through index fingertip (if hand detected)
                    if hands:
                        # Use primary hand (right hand preferred, or first detected)
                        primary_hand = None
                        for h in hands:
                            if h.handedness == "Right":
                                primary_hand = h
                                break
                        if primary_hand is None:
                            primary_hand = hands[0]

                        # Get index fingertip (landmark 8)
                        if primary_hand.landmarks and len(primary_hand.landmarks) > 8:
                            index_tip = primary_hand.landmarks[8].to_numpy()
                            eye_midpoint = face.eye_midpoint.to_numpy()

                            # Ray origin = eye midpoint, direction = index_tip - eye_midpoint
                            ray_origin = eye_midpoint
                            ray_dir = index_tip - eye_midpoint
                            ray_norm = np.linalg.norm(ray_dir)
                            if ray_norm > 1e-6:
                                ray_direction = ray_dir / ray_norm

                                # Compute intersection with virtual plane
                                intersection = virtual_plane.ray_plane_intersection(ray_origin, ray_direction)
                                if intersection is not None:
                                    normalized_coords = virtual_plane.point_to_normalized(intersection)

        # Draw virtual plane debug overlay
        if show_virtual_plane and head_coords is not None and head_coords.is_valid():
            annotated = virtual_plane.draw_debug(
                annotated,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                ray_origin=ray_origin,
                ray_direction=ray_direction,
                intersection=intersection,
                normalized_coords=normalized_coords
            )

        # Status overlay
        status = f"Frame: {frame_idx}/{frame_count}  Hands: {len(hands)}  Face: {'Yes' if faces else 'No'}  "
        if intersection is not None and normalized_coords is not None:
            u, v = normalized_coords
            status += f"Plane: u={u:.3f} v={v:.3f}  "
        status += "[h/f/v to toggle, space=pause, q=quit]"
        cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cv2.imshow("Air Mouse - Video Playback", annotated)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_hands = not show_hands
            print(f"\nHand tracking: {'ON' if show_hands else 'OFF'}")
        elif key == ord('f'):
            show_face = not show_face
            print(f"\nFace tracking: {'ON' if show_face else 'OFF'}")
        elif key == ord('v'):
            show_virtual_plane = not show_virtual_plane
            print(f"\nVirtual plane: {'ON' if show_virtual_plane else 'OFF'}")
        elif key == ord(' '):
            paused = not paused
            print(f"\nPaused: {paused}")

finally:
    cap.release()
    hand_tracker.close()
    face_tracker.close()
    cv2.destroyAllWindows()
    print("\nDone")