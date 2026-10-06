"""Detailed trace of projection math for scenario A."""
import sys
sys.path.insert(0, '.')
import numpy as np
from airmouse.vision.face_tracker import Face, FaceLandmark
from airmouse.vision.hand_tracker import Hand, Landmark
from airmouse.vision.head_coords import HeadCoordinateSystem
from airmouse.vision.virtual_plane import VirtualDisplayPlane

def make_face(eye, nose, forehead):
    lm = [FaceLandmark(0, 0, 0) for _ in range(468)]
    lm[133] = FaceLandmark(*eye)
    lm[33] = FaceLandmark(*eye)
    lm[362] = FaceLandmark(*eye)
    lm[263] = FaceLandmark(*eye)
    lm[1] = FaceLandmark(*nose)
    lm[10] = FaceLandmark(*forehead)
    return Face(landmarks=lm, confidence=1.0)

def make_hand(tip, wrist):
    lms = [Landmark(0, 0, 0) for _ in range(21)]
    lms[0] = Landmark(*wrist)
    lms[8] = Landmark(*tip)
    return Hand(landmarks=lms, confidence=1.0, handedness='Right')

# Scenario A
face = make_face(eye=(0.5, 0.4, -0.3), nose=(0.5, 0.45, -0.35), forehead=(0.5, 0.3, -0.3))
hand = make_hand(tip=(0.5, 0.5, -0.5), wrist=(0.5, 0.6, -0.4))

hc = HeadCoordinateSystem.from_face(face)

eye_cam = face.eye_midpoint.to_numpy()
tip_cam = hand.landmarks[8].to_numpy()

print("=== INPUT (camera coords, normalized [0,1]) ===")
print(f"  eye_cam  = {eye_cam}")
print(f"  tip_cam  = {tip_cam}")

# Manual camera_to_head
R = np.column_stack([hc.right, hc.up, hc.forward]).astype(np.float32)
T = np.eye(4, dtype=np.float32)
T[:3, :3] = R.T
T[:3, 3] = -R.T @ hc.origin

def cam_to_head(p):
    ph = np.append(p.astype(np.float32), 1.0)
    return (T @ ph)[:3]

def head_to_cam(p):
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R
    T_inv[:3, 3] = hc.origin
    ph = np.append(p.astype(np.float32), 1.0)
    return (T_inv @ ph)[:3]

eye_head = cam_to_head(eye_cam)
tip_head = cam_to_head(tip_cam)

print("\n=== HEAD COORDINATES ===")
print(f"  origin   = {hc.origin}")
print(f"  forward  = {hc.forward}")
print(f"  right    = {hc.right}")
print(f"  up       = {hc.up}")
print(f"  eye_head = {eye_head}")
print(f"  tip_head = {tip_head}")

dx = tip_head[0] - eye_head[0]
dy = tip_head[1] - eye_head[1]
dz = tip_head[2] - eye_head[2]

print(f"\n=== RAY (head coords) ===")
print(f"  dx={dx:.6f} dy={dy:.6f} dz={dz:.6f}")
ray = np.array([dx, dy, dz], dtype=np.float32)
ray_norm = np.linalg.norm(ray)
print(f"  ray_norm = {ray_norm:.6f}")
ray_dir = ray / ray_norm
print(f"  ray_dir  = {ray_dir}")

# Plane intersection
distance = 0.30
denom = ray_dir[2]
t = (distance - eye_head[2]) / denom
intersection = eye_head + t * ray_dir

print(f"\n=== INTERSECTION ===")
print(f"  denominator (ray_dir[2]) = {denom:.6f}")
print(f"  numerator (distance - eye_z) = {distance - eye_head[2]:.6f}")
print(f"  t = {t:.6f}")
print(f"  intersection_head = {intersection}")
print(f"  intersection_distance = {np.linalg.norm(intersection - eye_head):.6f}")

# Now check what point_to_normalized does
# It receives head_to_camera(intersection)
intersection_cam = head_to_cam(intersection)
print(f"\n=== ROUND-TRIP ===")
print(f"  intersection_cam = {intersection_cam}")

# And back to head
back_to_head = cam_to_head(intersection_cam)
print(f"  back_to_head = {back_to_head}")

# Now compute u,v
width, height = 0.40, 0.25
u = (back_to_head[0] + width/2) / width
v = (-back_to_head[1] + height/2) / height
print(f"\n=== NORMALIZED ===")
print(f"  u = ({back_to_head[0]:.6f} + {width/2}) / {width} = {u:.6f}")
print(f"  v = (-{back_to_head[1]:.6f} + {height/2}) / {height} = {v:.6f}")

# Check: what if we use the projector's head_coords vs virtual_plane's head_coords?
print(f"\n=== OBJECT IDENTITY ===")
print(f"  hc id = {id(hc)}")

vp = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25, head_coords=hc)
print(f"  vp.head_coords id = {id(vp.head_coords)}")
print(f"  same object? {vp.head_coords is hc}")

# Check if transform matrices are cached consistently
print(f"\n=== TRANSFORM MATRICES ===")
T1 = hc.get_transform_matrix()
T2 = vp.head_coords.get_transform_matrix()
print(f"  T (projector) == T (plane)? {np.allclose(T1, T2)}")

T_inv1 = hc.get_inverse_transform_matrix()
T_inv2 = vp.head_coords.get_inverse_transform_matrix()
print(f"  T_inv (projector) == T_inv (plane)? {np.allclose(T_inv1, T_inv2)}")

# Check round-trip with plane's head_coords
back2 = vp.head_coords.camera_to_head(intersection_cam)
print(f"  plane.head_coords.camera_to_head(intersection_cam) = {back2}")
print(f"  matches intersection_head? {np.allclose(back2, intersection)}")