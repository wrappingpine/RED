#!/usr/bin/env python3
"""Camera preview with both hand and face/head tracking skeleton + virtual display plane."""

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

print("=" * 50)
print("Air Mouse - Hand & Face Tracking + Virtual Plane Preview")
print("Press 'q' to quit, 'h' to toggle hand, 'f' to toggle face, 'v' to toggle virtual plane")
print("=" * 50)

camera = CameraManager()
if not camera.open_camera(CameraSettings(device_index=0, width=640, height=480, fps=30)):
    print("Failed to open camera")
    sys.exit(1)

print("Camera opened, creating trackers...")
hand_tracker = HandTracker(HandTrackerSettings(max_hands=2))
face_tracker = FaceTracker(FaceTrackerSettings(max_faces=1))

# Create virtual display plane (will be updated with head coords each frame)
virtual_plane = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25)

print("Trackers created, processing frames...")

show_hands = True
show_face = True
show_virtual_plane = True

# Approximate camera matrix
h, w = 480, 640
focal_length = w
camera_matrix = np.array([
    [focal_length, 0, w / 2],
    [0, focal_length, h / 2],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros(4, dtype=np.float32)

try:
    while True:
        ret, frame = camera.read_frame()
        if not ret:
            continue

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
        status = f"Hands: {len(hands)}  Face: {'Yes' if faces else 'No'}  "
        if intersection is not None and normalized_coords is not None:
            u, v = normalized_coords
            status += f"Plane: u={u:.3f} v={v:.3f}  "
        status += "[h/f/v to toggle]"
        cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cv2.imshow("Air Mouse - Hand & Face + Virtual Plane", annotated)

        key = cv2.waitKey(1) & 0xFF
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

finally:
    camera.close_camera()
    hand_tracker.close()
    face_tracker.close()
    cv2.destroyAllWindows()
    print("\nDone")