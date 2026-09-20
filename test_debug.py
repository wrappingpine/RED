"""Debug head rotation issue - fixed coordinate computation."""
from airmouse.vision.virtual_plane import VirtualDisplayPlane
from airmouse.vision.head_coords import HeadCoordinateSystem
from airmouse.vision.projection import HandProjector
from airmouse.vision.hand_tracker import Hand, Landmark, HandLandmark
from airmouse.vision.face_tracker import Face, FaceLandmark
import numpy as np

def create_mock_face(origin_x=0.5, origin_y=0.5, origin_z=0.0, yaw=0.0, pitch=0.0):
    landmarks = [FaceLandmark(0.5, 0.5, 0.0, 1.0) for _ in range(468)]
    
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    
    left_eye_rel = np.array([-0.03, 0.0, 0.0])
    right_eye_rel = np.array([0.03, 0.0, 0.0])
    nose_rel = np.array([0.0, 0.0, -0.1])
    forehead_rel = np.array([0.0, -0.05, 0.0])
    
    def rotate(p):
        x = p[0] * cos_y - p[2] * sin_y
        y = p[1]
        z = p[0] * sin_y + p[2] * cos_y
        x2 = x
        y2 = y * cos_p - z * sin_p
        z2 = y * sin_p + z * cos_p
        return np.array([x2, y2, z2])
    
    left_eye = rotate(left_eye_rel) + np.array([origin_x, origin_y, origin_z])
    right_eye = rotate(right_eye_rel) + np.array([origin_x, origin_y, origin_z])
    nose = rotate(nose_rel) + np.array([origin_x, origin_y, origin_z])
    forehead = rotate(forehead_rel) + np.array([origin_x, origin_y, origin_z])
    
    landmarks[Face.LEFT_EYE_INNER] = FaceLandmark(left_eye[0], left_eye[1], left_eye[2], 1.0)
    landmarks[Face.LEFT_EYE_OUTER] = FaceLandmark(left_eye[0] - 0.015, left_eye[1], left_eye[2], 1.0)
    landmarks[Face.RIGHT_EYE_INNER] = FaceLandmark(right_eye[0], right_eye[1], right_eye[2], 1.0)
    landmarks[Face.RIGHT_EYE_OUTER] = FaceLandmark(right_eye[0] + 0.015, right_eye[1], right_eye[2], 1.0)
    landmarks[Face.NOSE_TIP] = FaceLandmark(nose[0], nose[1], nose[2], 1.0)
    landmarks[Face.FOREHEAD] = FaceLandmark(forehead[0], forehead[1], forehead[2], 1.0)
    
    return Face(landmarks=landmarks, confidence=1.0)

# Test with 30 degree yaw
yaw = np.radians(30)
face = create_mock_face(yaw=yaw)

head_coords = HeadCoordinateSystem.from_face(face)
print(f"Head coords:")
print(f"  origin: {head_coords.origin}")
print(f"  forward: {head_coords.forward}")
print(f"  right: {head_coords.right}")
print(f"  up: {head_coords.up}")

plane = VirtualDisplayPlane(
    distance=0.30, width=0.30, height=0.20, head_coords=head_coords
)
print(f"\nPlane center cam: {plane._plane_center_cam}")

# Hand at head center [0, 0, 0.3] -> convert to camera coords
hand_center_head = np.array([0.0, 0.0, 0.3])
hand_center_cam = head_coords.head_to_camera(hand_center_head)
print(f"Hand at head center [0,0,0.3] -> camera: {hand_center_cam}")

# The hand camera coords should be exactly hand_center_cam (not 0.5 + hand_center_cam)
# because head_coords.origin IS the eye midpoint in camera coords
projector = HandProjector(plane, head_coords, use_head_coords_for_ray=True, ray_smoothing_alpha=0.0)
hand = Hand(
    landmarks=[Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(21)],
    handedness="Right", confidence=1.0
)
hand.landmarks[HandLandmark.INDEX_TIP.value] = Landmark(
    hand_center_cam[0], hand_center_cam[1], hand_center_cam[2], 1.0
)

result = projector.project(hand, face)
print(f"\nProjection result: u={result.u:.4f}, v={result.v:.4f}")
print(f"  intersection_head: {result.intersection_head}")
print(f"  intersection_camera: {result.intersection_camera}")