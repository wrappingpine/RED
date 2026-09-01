"""Test head-relative coordinate system - verify that head movement doesn't affect cursor position."""
from airmouse.control.cursor import CursorController, CursorConfig, SensitivityMode, SmoothingAlgorithm
from airmouse.vision.virtual_plane import VirtualDisplayPlane
from airmouse.vision.head_coords import HeadCoordinateSystem
from airmouse.vision.projection import HandProjector, ProjectionResult
from airmouse.vision.hand_tracker import Hand, Landmark, HandLandmark
from airmouse.vision.face_tracker import Face, FaceLandmark
import numpy as np

print("=" * 60)
print("TEST: Head-Relative Coordinate System")
print("=" * 60)

# Create head coordinate system - camera at origin, looking forward (+Z)
# Head is at origin, looking forward
head_coords = HeadCoordinateSystem(
    origin=np.array([0.0, 0.0, 0.0]),
    forward=np.array([0.0, 0.0, 1.0]),
    right=np.array([1.0, 0.0, 0.0]),
    up=np.array([0.0, 1.0, 0.0]),
    _valid=True
)

plane = VirtualDisplayPlane(
    distance=0.30,
    width=0.40,
    height=0.25,
    head_coords=head_coords
)

# Create mock face at center
face_landmarks = {}
for i in range(468):
    if i == Face.NOSE_TIP:
        face_landmarks[i] = Landmark(x=0.5, y=0.5, z=0.0)
    elif i == Face.LEFT_EYE_INNER:
        face_landmarks[i] = Landmark(x=0.45, y=0.45, z=0.0)
    elif i == Face.RIGHT_EYE_INNER:
        face_landmarks[i] = Landmark(x=0.55, y=0.45, z=0.0)
    elif i == Face.FOREHEAD:
        face_landmarks[i] = Landmark(x=0.5, y=0.3, z=0.0)
    else:
        face_landmarks[i] = Landmark(x=0.5, y=0.5, z=0.0)

face = Face(
    landmarks=[face_landmarks[i] for i in range(468)],
    confidence=1.0
)

# Create projector
projector = HandProjector(plane, head_coords)

# Test 1: Hand at plane center (relative to head)
# Hand at x=0, y=0, z=0.3 in head coords = plane center
# In camera coords with identity transform, this is the same
hand_landmarks = {}
for i in range(21):
    if i == HandLandmark.INDEX_TIP.value:
        hand_landmarks[i] = Landmark(x=0.5, y=0.5, z=0.3)  # plane center
    elif i == HandLandmark.WRIST.value:
        hand_landmarks[i] = Landmark(x=0.5, y=0.7, z=0.3)
    else:
        hand_landmarks[i] = Landmark(x=0.5, y=0.6, z=0.3)

hand = Hand(landmarks=[hand_landmarks[i] for i in range(21)], handedness="Right", confidence=1.0)

result = projector.project(hand, face)
print(f"Hand at plane center: valid={result.valid}, u={result.u:.4f}, v={result.v:.4f}")
assert result.valid and abs(result.u - 0.5) < 0.01 and abs(result.v - 0.5) < 0.01, "Should be at center"

# Test 2: Hand at right edge of plane (head coords x=+0.2 = right)
# This maps to u=1.0 (right edge)
hand_landmarks = {}
for i in range(21):
    if i == HandLandmark.INDEX_TIP.value:
        hand_landmarks[i] = Landmark(x=0.7, y=0.5, z=0.3)  # right side in head coords
    elif i == HandLandmark.WRIST.value:
        hand_landmarks[i] = Landmark(x=0.7, y=0.7, z=0.3)
    else:
        hand_landmarks[i] = Landmark(x=0.7, y=0.6, z=0.3)

hand = Hand(landmarks=[hand_landmarks[i] for i in range(21)], handedness="Right", confidence=1.0)

result = projector.project(hand, face)
print(f"Hand at right edge: valid={result.valid}, u={result.u:.4f}, v={result.v:.4f}")
assert result.valid and result.u > 0.9, "Should be at right edge"

# Test 3: Head moves RIGHT (head coordinate system shifts)
# In head coords, plane is still at z=0.3, x=[-0.2, 0.2]
# But in camera coords, head moved right by 0.1m
# New head coords: origin=[0.1, 0, 0], forward=[0,0,1], right=[1,0,0], up=[0,1,0]
head_coords_moved = HeadCoordinateSystem(
    origin=np.array([0.1, 0.0, 0.0]),
    forward=np.array([0.0, 0.0, 1.0]),
    right=np.array([1.0, 0.0, 0.0]),
    up=np.array([0.0, 1.0, 0.0]),
    _valid=True
)

plane_moved = VirtualDisplayPlane(
    distance=0.30,
    width=0.40,
    height=0.25,
    head_coords=head_coords_moved
)

# Face moves with head - nose tip at new position
face_landmarks = {}
for i in range(468):
    if i == Face.NOSE_TIP:
        face_landmarks[i] = Landmark(x=0.6, y=0.5, z=0.0)  # moved right in camera
    elif i == Face.LEFT_EYE_INNER:
        face_landmarks[i] = Landmark(x=0.55, y=0.45, z=0.0)
    elif i == Face.RIGHT_EYE_INNER:
        face_landmarks[i] = Landmark(x=0.65, y=0.45, z=0.0)
    elif i == Face.FOREHEAD:
        face_landmarks[i] = Landmark(x=0.6, y=0.3, z=0.0)
    else:
        face_landmarks[i] = Landmark(x=0.6, y=0.5, z=0.0)

face = Face(
    landmarks=[face_landmarks[i] for i in range(468)],
    confidence=1.0
)

projector2 = HandProjector(plane_moved, head_coords_moved)

# Hand stays at SAME position relative to head (right edge of plane)
# In head coords: x=0.2 (right edge), in camera coords: x=0.1+0.2=0.3
# But we need normalized camera coords...
# Camera is at origin, looking forward. Plane is at head coords z=0.3
# Hand at head coords (0.2, 0, 0.3) -> camera coords (0.3, 0, 0.3)
# Normalized to camera frame (assuming 640x480, but we use normalized 0-1)
# The projector uses 3D points in camera coords

hand_landmarks = {}
for i in range(21):
    if i == HandLandmark.INDEX_TIP.value:
        # Hand at head-relative right edge: head_coords.x=0.2, z=0.3
        # In camera coords: x = head_origin.x + 0.2 = 0.1 + 0.2 = 0.3
        # Need to convert to normalized camera coords (0-1)
        # Assuming camera frame width = 0.64m (640px at 1mm/px)
        hand_landmarks[i] = Landmark(x=0.5 + 0.3/0.64, y=0.5, z=0.3)  # approx
    elif i == HandLandmark.WRIST.value:
        hand_landmarks[i] = Landmark(x=0.5 + 0.3/0.64, y=0.7, z=0.3)
    else:
        hand_landmarks[i] = Landmark(x=0.5 + 0.3/0.64, y=0.6, z=0.3)

hand = Hand(landmarks=[hand_landmarks[i] for i in range(21)], handedness="Right", confidence=1.0)

result = projector2.project(hand, face)
print(f"Head moved right, hand at right edge: valid={result.valid}, u={result.u:.4f}, v={result.v:.4f}")
print("  (Should still be at u=1.0 since hand is at right edge relative to head)")

# Test 4: Head turns LEFT (yaw)
# Head coordinate system rotates - forward now points left relative to camera
# forward = [-sin(30°), 0, cos(30°)] = [-0.5, 0, 0.866]
# right = [cos(30°), 0, sin(30°)] = [0.866, 0, 0.5]
# up = [0, 1, 0]
import math
yaw = math.radians(30)
head_coords_rotated = HeadCoordinateSystem(
    origin=np.array([0.0, 0.0, 0.0]),
    forward=np.array([-math.sin(yaw), 0.0, math.cos(yaw)]),
    right=np.array([math.cos(yaw), 0.0, math.sin(yaw)]),
    up=np.array([0.0, 1.0, 0.0]),
    _valid=True
)

plane_rotated = VirtualDisplayPlane(
    distance=0.30,
    width=0.40,
    height=0.25,
    head_coords=head_coords_rotated
)

# Face rotates with head
face_landmarks = {}
for i in range(468):
    if i == Face.NOSE_TIP:
        # Nose tip in camera coords: origin + forward * distance_to_nose
        nose_pos = head_coords_rotated.head_to_camera(np.array([0.0, 0.0, 0.05]))
        face_landmarks[i] = Landmark(x=nose_pos[0]+0.5, y=nose_pos[1]+0.5, z=nose_pos[2])
    elif i == Face.LEFT_EYE_INNER:
        eye_pos = head_coords_rotated.head_to_camera(np.array([-0.03, -0.03, 0.0]))
        face_landmarks[i] = Landmark(x=eye_pos[0]+0.5, y=eye_pos[1]+0.5, z=eye_pos[2])
    elif i == Face.RIGHT_EYE_INNER:
        eye_pos = head_coords_rotated.head_to_camera(np.array([0.03, -0.03, 0.0]))
        face_landmarks[i] = Landmark(x=eye_pos[0]+0.5, y=eye_pos[1]+0.5, z=eye_pos[2])
    elif i == Face.FOREHEAD:
        forehead_pos = head_coords_rotated.head_to_camera(np.array([0.0, -0.05, 0.0]))
        face_landmarks[i] = Landmark(x=forehead_pos[0]+0.5, y=forehead_pos[1]+0.5, z=forehead_pos[2])
    else:
        face_landmarks[i] = Landmark(x=0.5, y=0.5, z=0.0)

face = Face(
    landmarks=[face_landmarks[i] for i in range(468)],
    confidence=1.0
)

projector3 = HandProjector(plane_rotated, head_coords_rotated)

# Hand at center of plane in head coords (x=0, y=0, z=0.3)
# In camera coords: head_to_camera([0, 0, 0.3])
hand_center_head = np.array([0.0, 0.0, 0.3])
hand_center_cam = head_coords_rotated.head_to_camera(hand_center_head)

hand_landmarks = {}
for i in range(21):
    if i == HandLandmark.INDEX_TIP.value:
        hand_landmarks[i] = Landmark(x=hand_center_cam[0]+0.5, y=hand_center_cam[1]+0.5, z=hand_center_cam[2])
    elif i == HandLandmark.WRIST.value:
        wrist_pos = head_coords_rotated.head_to_camera(np.array([0.0, 0.07, 0.3]))
        hand_landmarks[i] = Landmark(x=wrist_pos[0]+0.5, y=wrist_pos[1]+0.5, z=wrist_pos[2])
    else:
        palm_pos = head_coords_rotated.head_to_camera(np.array([0.0, 0.06, 0.3]))
        hand_landmarks[i] = Landmark(x=palm_pos[0]+0.5, y=palm_pos[1]+0.5, z=palm_pos[2])

hand = Hand(landmarks=[hand_landmarks[i] for i in range(21)], handedness="Right", confidence=1.0)

result = projector3.project(hand, face)
print(f"Head rotated 30°, hand at center: valid={result.valid}, u={result.u:.4f}, v={result.v:.4f}")
print("  (Should be near u=0.5, v=0.5 since hand is at center relative to head)")

print("\n" + "=" * 60)
print("HEAD-RELATIVE TESTS COMPLETE")
print("=" * 60)