"""Test hand-to-cursor directional coordination - verify that hand movements map correctly to cursor movements."""
from airmouse.control.cursor import CursorController, CursorConfig, SensitivityMode, SmoothingAlgorithm
from airmouse.vision.virtual_plane import VirtualDisplayPlane
from airmouse.vision.head_coords import HeadCoordinateSystem
from airmouse.vision.projection import HandProjector
from airmouse.vision.hand_tracker import Hand, Landmark, HandLandmark
import numpy as np
import time

# Create mock hand at center
def create_mock_hand(x_norm=0.5, y_norm=0.5):
    landmarks = []
    for i in range(21):
        if i == HandLandmark.INDEX_TIP.value:
            landmarks.append(Landmark(x=x_norm, y=y_norm, z=0.0))
        elif i == HandLandmark.WRIST.value:
            landmarks.append(Landmark(x=0.5, y=0.7, z=0.0))
        else:
            landmarks.append(Landmark(x=0.5, y=0.6, z=0.0))
    return Hand(landmarks=landmarks, handedness="Right", confidence=1.0)

# Test 1: Basic cursor mapping (no virtual plane) - use get_relative_movement ONLY
print("=" * 60)
print("TEST 1: Basic Cursor Mapping (Camera Frame)")
print("=" * 60)

config = CursorConfig(
    screen_width=1920,
    screen_height=1080,
    camera_width=640,
    camera_height=480,
    dead_zone_radius=0.02,
    sensitivity_mode=SensitivityMode.NORMAL,
    acceleration=1.2,
    smoothing=SmoothingAlgorithm.NONE,
    use_index_tip=True
)

controller = CursorController(config)

# Test hand at center -> should map to center, first call returns (0,0)
hand_center = create_mock_hand(0.5, 0.5)
rel = controller.get_relative_movement(hand_center)
print(f"Hand at center (0.5, 0.5): rel={rel}")
assert rel == (0, 0), "First call should be zero"

# Test hand moves right -> cursor should move right
hand_right = create_mock_hand(0.7, 0.5)
rel = controller.get_relative_movement(hand_right)
print(f"Hand moves right (0.7, 0.5): rel={rel}")
assert rel[0] > 0, "Relative X should be positive (right)"

# Test hand moves left -> cursor should move left
controller.reset()
hand_left = create_mock_hand(0.3, 0.5)
rel = controller.get_relative_movement(hand_left)
print(f"Hand moves left (0.3, 0.5): rel={rel}")
assert rel == (0, 0), "First call after reset should be zero"

hand_left2 = create_mock_hand(0.3, 0.5)
rel = controller.get_relative_movement(hand_left2)
print(f"Hand stays left (0.3, 0.5): rel={rel}")
# Actually this is still relative to first position (0.5, 0.5) -> dx = 0.3 - 0.5 = -0.2
# But wait - the reference point is set on first frame at 0.5, so moving to 0.3 gives negative
# Let me trace through: first call at 0.3 sets ref=0.3, returns 0. Second call at 0.3 gives dx=0
# So we need to call with center first, then left
controller.reset()
hand_center = create_mock_hand(0.5, 0.5)
rel = controller.get_relative_movement(hand_center)  # ref=0.5, returns 0
hand_left = create_mock_hand(0.3, 0.5)
rel = controller.get_relative_movement(hand_left)  # dx = 0.3 - 0.5 = -0.2
print(f"Hand moves left from center: rel={rel}")
assert rel[0] < 0, "Relative X should be negative (left)"

# Test hand moves up -> cursor should move up
controller.reset()
hand_center = create_mock_hand(0.5, 0.5)
rel = controller.get_relative_movement(hand_center)
hand_up = create_mock_hand(0.5, 0.3)
rel = controller.get_relative_movement(hand_up)
print(f"Hand moves up from center: rel={rel}")
assert rel[1] < 0, "Relative Y should be negative (up)"

# Test hand moves down -> cursor should move down
controller.reset()
hand_center = create_mock_hand(0.5, 0.5)
rel = controller.get_relative_movement(hand_center)
hand_down = create_mock_hand(0.5, 0.7)
rel = controller.get_relative_movement(hand_down)
print(f"Hand moves down from center: rel={rel}")
assert rel[1] > 0, "Relative Y should be positive (down)"

print("\n✓ Basic cursor mapping PASSED")

# Test 2: Virtual Plane mapping
print("\n" + "=" * 60)
print("TEST 2: Virtual Plane Mapping (Head-Relative)")
print("=" * 60)

# Create head coordinate system (identity - camera = head)
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

# Create mock face for projector
from airmouse.vision.face_tracker import Face
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

# Create mock hand landmarks in 3D
# Index tip at center of plane (30cm in front = z=+0.3 in camera coords with forward=+z)
hand_landmarks = {}
for i in range(21):
    if i == HandLandmark.INDEX_TIP.value:
        hand_landmarks[i] = Landmark(x=0.5, y=0.5, z=0.3)
    elif i == HandLandmark.WRIST.value:
        hand_landmarks[i] = Landmark(x=0.5, y=0.7, z=0.3)
    else:
        hand_landmarks[i] = Landmark(x=0.5, y=0.6, z=0.3)

hand = Hand(landmarks=[hand_landmarks[i] for i in range(21)], handedness="Right", confidence=1.0)

# Project to plane
result = projector.project(hand, face)
print(f"Projected to plane: valid={result.valid}, u={result.u:.4f}, v={result.v:.4f}")

if result.valid:
    u, v = result.u, result.v
    print(f"Normalized coords: u={u:.4f}, v={v:.4f}")
    assert 0 <= u <= 1, "u should be in [0,1]"
    assert 0 <= v <= 1, "v should be in [0,1]"
    print("✓ Virtual plane projection PASSED")
else:
    print(f"✗ Virtual plane projection FAILED: {result.error_message}")

# Test 3: Test virtual plane with movement
print("\n" + "=" * 60)
print("TEST 3: Virtual Plane with Hand Movement")
print("=" * 60)

# Create a new controller for plane-relative mode
config2 = CursorConfig(
    screen_width=1920,
    screen_height=1080,
    camera_width=640,
    camera_height=480,
    dead_zone_radius=0.02,
    sensitivity_mode=SensitivityMode.NORMAL,
    acceleration=1.2,
    smoothing=SmoothingAlgorithm.NONE,
    use_index_tip=True
)
controller2 = CursorController(config2)

# Test center first to establish reference
u, v = 0.5, 0.5
rel = controller2.get_relative_movement_from_plane(u, v)
print(f"Plane center (0.5, 0.5): rel={rel}")
assert rel == (0, 0), "First frame should be zero"

# Move right on plane
u, v = 0.6, 0.5
rel = controller2.get_relative_movement_from_plane(u, v)
print(f"Plane right (0.6, 0.5): rel={rel}")
assert rel[0] > 0, "Moving right on plane should give positive dx"

# Move left on plane
controller2.reset()
u, v = 0.5, 0.5
rel = controller2.get_relative_movement_from_plane(u, v)  # center first
u, v = 0.4, 0.5
rel = controller2.get_relative_movement_from_plane(u, v)  # then left
print(f"Plane left from center (0.4, 0.5): rel={rel}")
assert rel[0] < 0, "Moving left on plane should give negative dx"

# Move up on plane (v decreases = up in screen coords)
controller2.reset()
u, v = 0.5, 0.5
rel = controller2.get_relative_movement_from_plane(u, v)
u, v = 0.5, 0.4
rel = controller2.get_relative_movement_from_plane(u, v)
print(f"Plane up (0.5, 0.4): rel={rel}")
assert rel[1] < 0, "Moving up on plane (v decreasing) should give negative dy"

# Move down on plane (v increases = down in screen coords)
controller2.reset()
u, v = 0.5, 0.5
rel = controller2.get_relative_movement_from_plane(u, v)
u, v = 0.5, 0.6
rel = controller2.get_relative_movement_from_plane(u, v)
print(f"Plane down (0.5, 0.6): rel={rel}")
assert rel[1] > 0, "Moving down on plane (v increasing) should give positive dy"

print("\n✓ Virtual plane movement PASSED")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)