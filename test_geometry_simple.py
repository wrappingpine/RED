"""Minimal geometry test to verify projection doesn't produce out-of-bounds u/v."""
import sys
sys.path.insert(0, '.')
from airmouse.vision.head_coords import HeadCoordinateSystem
from airmouse.vision.virtual_plane import VirtualDisplayPlane
from airmouse.vision.projection import HandProjector
from airmouse.vision.face_tracker import Face, FaceLandmark
from airmouse.vision.hand_tracker import Hand, Landmark

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

# Upright head
face = make_face(eye=(0.5, 0.4, -0.3), nose=(0.5, 0.4, -0.4), forehead=(0.5, 0.3, -0.3))
hc = HeadCoordinateSystem.from_face(face)
vp = VirtualDisplayPlane(distance=0.30, width=1.0, height=1.0, head_coords=hc)
proj = HandProjector(vp, hc)

# Test cases: (tip, expected_u_range, expected_v_range, description)
tests = [
    ((0.5, 0.4, -0.5), (0.45, 0.55), (0.45, 0.55), "center"),
    ((0.0, 0.4, -0.5), (0.0, 0.1), (0.45, 0.55), "left"),
    ((1.0, 0.4, -0.5), (0.9, 1.0), (0.45, 0.55), "right"),
    ((0.5, 0.0, -0.5), (0.45, 0.55), (0.0, 0.2), "top"),
    ((0.5, 1.0, -0.5), (0.45, 0.55), (0.8, 1.0), "bottom"),
]

all_pass = True
for tip, u_range, v_range, desc in tests:
    hand = make_hand(tip=tip, wrist=(tip[0], tip[1]+0.1, tip[2]+0.1))
    r = proj.project(hand, face)
    u_ok = u_range[0] <= r.u <= u_range[1]
    v_ok = v_range[0] <= r.v <= v_range[1]
    status = "PASS" if u_ok and v_ok and r.valid else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"{desc:10s}: u={r.u:.3f} v={r.v:.3f} valid={r.valid} -> {status}")

# Edge case: parallel ray (fingertip at same depth)
hand_par = make_hand(tip=(0.5, 0.4, -0.3), wrist=(0.5, 0.5, -0.3))
r_par = proj.project(hand_par, face)
print(f"\nParallel ray: u={r_par.u:.3f} v={r_par.v:.3f} valid={r_par.valid} error={r_par.error_message}")
# Should fallback to clamped position, not produce enormous values
if r_par.valid:
    print("PASS: fallback triggered")
else:
    print("FAIL: projection invalid without fallback")

print("\n=== RESULT ===" if all_pass else "\n=== SOME TESTS FAILED ===")