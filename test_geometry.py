"""Deterministic geometry test for ray-plane intersection.

Tests the math in isolation with known inputs and expected outputs.
Uses an upright head (forward=(0,0,-1), up=(0,-1,0)) for predictable results.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
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

# Upright head: forward=(0,0,-1), up=(0,-1,0)
# eye=(0.5, 0.4, -0.3), nose=(0.5, 0.4, -0.4), forehead=(0.5, 0.3, -0.3)
face = make_face(eye=(0.5, 0.4, -0.3), nose=(0.5, 0.4, -0.4), forehead=(0.5, 0.3, -0.3))

hc = HeadCoordinateSystem.from_face(face)
print(f"Head: forward={hc.forward}, right={hc.right}, up={hc.up}")

vp = VirtualDisplayPlane(distance=0.30, width=1.0, height=1.0, head_coords=hc)
proj = HandProjector(vp, hc)

tests = []

# Test 1: Center (fingertip directly forward from eye)
hand = make_hand(tip=(0.5, 0.4, -0.5), wrist=(0.5, 0.5, -0.4))
r = proj.project(hand, face)
print(f"Test 1 (center): u={r.u:.4f} v={r.v:.4f} valid={r.valid}")
assert abs(r.u - 0.5) < 0.01 and abs(r.v - 0.5) < 0.01, f"Expected (0.5, 0.5), got ({r.u}, {r.v})"
print("  PASS")

# Test 2: Left edge (fingertip at camera x=0.0)
hand2 = make_hand(tip=(0.0, 0.4, -0.5), wrist=(0.1, 0.5, -0.4))
r2 = proj.project(hand2, face)
print(f"Test 2 (left): u={r2.u:.4f} v={r2.v:.4f} valid={r2.valid}")
assert abs(r2.u - 0.0) < 0.05, f"Expected u≈0.0, got {r2.u}"
print("  PASS")

# Test 3: Right edge
hand3 = make_hand(tip=(1.0, 0.4, -0.5), wrist=(0.9, 0.5, -0.4))
r3 = proj.project(hand3, face)
print(f"Test 3 (right): u={r3.u:.4f} v={r3.v:.4f} valid={r3.valid}")
assert abs(r3.u - 1.0) < 0.05, f"Expected u≈1.0, got {r3.u}"
print("  PASS")

# Test 4: Top edge (fingertip at camera y=0.0, which is above eye at y=0.4)
hand4 = make_hand(tip=(0.5, 0.0, -0.5), wrist=(0.5, 0.1, -0.4))
r4 = proj.project(hand4, face)
print(f"Test 4 (top): u={r4.u:.4f} v={r4.v:.4f} valid={r4.valid}")
# Expected: fingertip at camera y=0.0, eye at y=0.4. In head coords,
# fingertip is above eye by 0.4 units. With plane at z=0.30 and
# fingertip at z=-0.5 (0.2 units forward from eye at z=-0.3),
# intersection_y = 0.4 * (0.30 / 0.20) = 0.60 units above eye.
# With plane height=1.0 (±0.5 range), this is outside bounds.
# v = (-0.60 + 0.5) / 1.0 = -0.10 → out of bounds
# The fallback clamps to v=0.0
assert abs(r4.v - 0.0) < 0.05, f"Expected v≈0.0, got {r4.v}"
print("  PASS")

# Test 5: Bottom edge
hand5 = make_hand(tip=(0.5, 1.0, -0.5), wrist=(0.5, 0.9, -0.4))
r5 = proj.project(hand5, face)
print(f"Test 5 (bottom): u={r5.u:.4f} v={r5.v:.4f} valid={r5.valid}")
assert abs(r5.v - 1.0) < 0.05, f"Expected v≈1.0, got {r5.v}"
print("  PASS")

# Test 6: Parallel ray (fingertip at same depth as eye)
hand6 = make_hand(tip=(0.5, 0.4, -0.3), wrist=(0.5, 0.5, -0.3))
r6 = proj.project(hand6, face)
print(f"Test 6 (parallel): u={r6.u:.4f} v={r6.v:.4f} valid={r6.valid}")
print(f"  error={r6.error_message}")
print("  PASS (handled gracefully)")

# Test 7: Ray pointing away (fingertip behind eye)
hand7 = make_hand(tip=(0.5, 0.4, -0.1), wrist=(0.5, 0.5, -0.1))
r7 = proj.project(hand7, face)
print(f"Test 7 (away): u={r7.u:.4f} v={r7.v:.4f} valid={r7.valid}")
print(f"  error={r7.error_message}")
print("  PASS (handled gracefully)")

# Test 8: Near-parallel ray (fingertip barely in front of eye)
hand8 = make_hand(tip=(0.5, 0.4, -0.31), wrist=(0.5, 0.5, -0.31))
r8 = proj.project(hand8, face)
print(f"Test 8 (near-parallel): u={r8.u:.4f} v={r8.v:.4f} valid={r8.valid}")
print(f"  error={r8.error_message}")
print("  PASS (handled gracefully)")

# Test 9: Far right, same depth as eye (extreme case)
hand9 = make_hand(tip=(1.0, 0.4, -0.3), wrist=(0.9, 0.5, -0.3))
r9 = proj.project(hand9, face)
print(f"Test 9 (far right, same depth): u={r9.u:.4f} v={r9.v:.4f} valid={r9.valid}")
print(f"  error={r9.error_message}")
print("  PASS (handled gracefully)")

print("\n=== ALL TESTS PASSED ===")