"""Reproduce projection_intersection_failed with synthetic landmarks."""
import sys
sys.path.insert(0, '.')
import numpy as np
from airmouse.vision.face_tracker import Face, FaceLandmark
from airmouse.vision.hand_tracker import Hand, Landmark, HandLandmark
from airmouse.vision.head_coords import HeadCoordinateSystem
from airmouse.vision.virtual_plane import VirtualDisplayPlane
from airmouse.vision.projection import HandProjector


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


# Scenario A: user facing camera, hand in front of face
face = make_face(eye=(0.5, 0.4, -0.3), nose=(0.5, 0.45, -0.35), forehead=(0.5, 0.3, -0.3))
hand = make_hand(tip=(0.5, 0.5, -0.5), wrist=(0.5, 0.6, -0.4))

hc = HeadCoordinateSystem.from_face(face)
print("HEAD COORDS:")
print(f"  origin(eye)={hc.origin}")
print(f"  forward={hc.forward}")
print(f"  right={hc.right}")
print(f"  up={hc.up}")

# Check handedness
det = np.linalg.det(np.column_stack([hc.right, hc.up, hc.forward]))
print(f"  det(right,up,forward)={det:.4f} (should be >0 for right-handed)")

# Test camera->head and back
eye_cam = face.eye_midpoint.to_numpy()
eye_head = hc.camera_to_head(eye_cam)
eye_back = hc.head_to_camera(eye_head)
print(f"  eye_cam={eye_cam} -> eye_head={eye_head} -> eye_back={eye_back}")

vp = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25, head_coords=hc)
proj = HandProjector(vp, hc)
result = proj.project(hand, face)
print(f"\nPROJECTION: valid={result.valid} u={result.u:.3f} v={result.v:.3f} err={result.error_message}")

# Scenario B: hand far to the right and up (like the log: u=-3.339, v=2.000)
# In camera coords, fingertip far right (+x) and far up (-y)
face2 = make_face(eye=(0.5, 0.4, -0.3), nose=(0.5, 0.45, -0.35), forehead=(0.5, 0.3, -0.3))
hand2 = make_hand(tip=(0.95, 0.05, -0.9), wrist=(0.9, 0.2, -0.7))
hc2 = HeadCoordinateSystem.from_face(face2)
vp2 = VirtualDisplayPlane(distance=0.30, width=0.40, height=0.25, head_coords=hc2)
proj2 = HandProjector(vp2, hc2)
result2 = proj2.project(hand2, face2)
print(f"\nPROJECTION (far hand): valid={result2.valid} u={result2.u:.3f} v={result2.v:.3f}")

# Inspect the head-coord ray
eye_h = hc2.camera_to_head(face2.eye_midpoint.to_numpy())
tip_h = hc2.camera_to_head(hand2.landmarks[8].to_numpy())
print(f"  eye_head={eye_h}")
print(f"  tip_head={tip_h}")
print(f"  ray_dir_head={tip_h - eye_h}")
print(f"  plane z=0.30, eye z={eye_h[2]:.3f}")