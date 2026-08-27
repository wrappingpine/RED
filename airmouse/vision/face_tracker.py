"""
Face Tracking Module for Air Mouse

Uses MediaPipe FaceLandmarker (Tasks API) for real-time face landmark detection.
Provides 468 face landmarks with head pose estimation (eye midpoint, nose, forehead).
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger(__name__)


@dataclass
class FaceLandmark:
    """Normalized face landmark point (0.0 to 1.0)."""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0

    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        """Convert to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass
class Face:
    """Detected face with landmarks and derived head pose."""
    landmarks: List[FaceLandmark] = field(default_factory=list)
    confidence: float = 0.0

    # Derived properties (computed on demand)
    _eye_midpoint: Optional[FaceLandmark] = None
    _nose_tip: Optional[FaceLandmark] = None
    _forehead: Optional[FaceLandmark] = None
    _left_eye_center: Optional[FaceLandmark] = None
    _right_eye_center: Optional[FaceLandmark] = None
    _forward_vector: Optional[np.ndarray] = None
    _up_vector: Optional[np.ndarray] = None
    _right_vector: Optional[np.ndarray] = None

    # Pre-allocated temp objects to avoid allocations
    _temp_face_landmark: FaceLandmark = field(default_factory=lambda: FaceLandmark(0.0, 0.0, 0.0))
    _temp_vector: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    # MediaPipe face landmark indices (468 points)
    # Key landmarks for head pose:
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    LEFT_IRIS_CENTER = 468  # Approximate - iris landmarks start at 468
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    RIGHT_IRIS_CENTER = 473
    NOSE_TIP = 1
    FOREHEAD = 10
    CHIN = 152

    def __post_init__(self):
        if len(self.landmarks) >= 468:
            self._compute_derived()

    def _compute_derived(self):
        """Compute derived head pose properties from landmarks."""
        if len(self.landmarks) < 468:
            return

        # Get key landmarks
        left_eye_inner = self.landmarks[self.LEFT_EYE_INNER]
        left_eye_outer = self.landmarks[self.LEFT_EYE_OUTER]
        right_eye_inner = self.landmarks[self.RIGHT_EYE_INNER]
        right_eye_outer = self.landmarks[self.RIGHT_EYE_OUTER]
        nose_tip = self.landmarks[self.NOSE_TIP]
        forehead = self.landmarks[self.FOREHEAD]

        # Compute eye centers (midpoint of inner/outer corners) - reuse temp
        left_eye = self._left_eye_center or FaceLandmark(0.0, 0.0, 0.0)
        left_eye.x = (left_eye_inner.x + left_eye_outer.x) / 2
        left_eye.y = (left_eye_inner.y + left_eye_outer.y) / 2
        left_eye.z = (left_eye_inner.z + left_eye_outer.z) / 2
        self._left_eye_center = left_eye

        right_eye = self._right_eye_center or FaceLandmark(0.0, 0.0, 0.0)
        right_eye.x = (right_eye_inner.x + right_eye_outer.x) / 2
        right_eye.y = (right_eye_inner.y + right_eye_outer.y) / 2
        right_eye.z = (right_eye_inner.z + right_eye_outer.z) / 2
        self._right_eye_center = right_eye

        # Eye midpoint = origin of head coordinate system
        eye_mid = self._eye_midpoint or FaceLandmark(0.0, 0.0, 0.0)
        eye_mid.x = (left_eye.x + right_eye.x) / 2
        eye_mid.y = (left_eye.y + right_eye.y) / 2
        eye_mid.z = (left_eye.z + right_eye.z) / 2
        self._eye_midpoint = eye_mid

        self._nose_tip = nose_tip
        self._forehead = forehead

        # Compute head orientation vectors
        self._compute_head_vectors()

    def _compute_head_vectors(self):
        """Compute forward, up, right vectors for head coordinate system."""
        if not (self._eye_midpoint and self._nose_tip and self._forehead):
            return

        # Forward vector: from eye midpoint to nose tip
        eye_mid = self._eye_midpoint.to_numpy()
        nose = self._nose_tip.to_numpy()
        forehead_pt = self._forehead.to_numpy()

        forward = nose - eye_mid
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-6:
            self._forward_vector = forward / forward_norm
        else:
            self._forward_vector = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Up vector: from eye midpoint to forehead (roughly)
        up = forehead_pt - eye_mid
        up_norm = np.linalg.norm(up)
        if up_norm > 1e-6:
            up = up / up_norm
        else:
            up = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        # Right vector = forward x up (cross product)
        right = np.cross(self._forward_vector, up)
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-6:
            self._right_vector = right / right_norm
        else:
            self._right_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Recompute up to be orthogonal: up = right x forward
        self._up_vector = np.cross(self._right_vector, self._forward_vector)

    @property
    def eye_midpoint(self) -> Optional[FaceLandmark]:
        return self._eye_midpoint

    @property
    def nose_tip(self) -> Optional[FaceLandmark]:
        return self._nose_tip

    @property
    def forehead(self) -> Optional[FaceLandmark]:
        return self._forehead

    @property
    def left_eye_center(self) -> Optional[FaceLandmark]:
        return self._left_eye_center

    @property
    def right_eye_center(self) -> Optional[FaceLandmark]:
        return self._right_eye_center

    @property
    def forward_vector(self) -> Optional[np.ndarray]:
        return self._forward_vector

    @property
    def up_vector(self) -> Optional[np.ndarray]:
        return self._up_vector

    @property
    def right_vector(self) -> Optional[np.ndarray]:
        return self._right_vector

    def get_head_transform_matrix(self) -> Optional[np.ndarray]:
        """
        Get 4x4 transformation matrix from camera coordinates to head coordinates.

        Head coordinate system:
        - Origin: eye midpoint
        - X axis: right vector (pointing to user's right)
        - Y axis: up vector (pointing up)
        - Z axis: forward vector (pointing forward from face)

        Returns 4x4 matrix that transforms camera-space points to head-space points.
        """
        if not (self._forward_vector is not None and self._up_vector is not None
                and self._right_vector is not None and self._eye_midpoint is not None):
            return None

        # Rotation matrix (columns are basis vectors)
        R = np.column_stack([
            self._right_vector,
            self._up_vector,
            self._forward_vector
        ]).astype(np.float32)

        # Translation: negative eye midpoint in camera coordinates
        t = -self._eye_midpoint.to_numpy()

        # 4x4 transform matrix
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def transform_to_head_coords(self, point: np.ndarray) -> Optional[np.ndarray]:
        """Transform a 3D point from camera coordinates to head coordinates."""
        T = self.get_head_transform_matrix()
        if T is None:
            return None

        # Homogeneous coordinates
        p_homo = np.append(point, 1.0)
        p_head = T @ p_homo
        return p_head[:3]


@dataclass
class FaceTrackerSettings:
    """Face tracker configuration."""
    max_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    output_face_blendshapes: bool = False
    output_facial_transformation_matrixes: bool = True


class FaceTracker:
    """
    MediaPipe FaceLandmarker wrapper for real-time face tracking.

    Features:
    - Single face tracking (primary user)
    - 468 face landmarks
    - Head pose via facial transformation matrix
    - Eye midpoint, nose tip, forehead extraction
    """

    def __init__(self, settings: Optional[FaceTrackerSettings] = None):
        self.settings = settings or FaceTrackerSettings()
        self._landmarker = None
        # Pre-allocated landmark cache (468 landmarks max)
        self._landmark_cache = [FaceLandmark(0.0, 0.0, 0.0) for _ in range(468)]
        self._initialize()

    def _initialize(self):
        """Initialize MediaPipe FaceLandmarker."""
        model_path = self._get_model_path()

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,  # VIDEO mode for temporal coherence
            num_faces=self.settings.max_faces,
            min_face_detection_confidence=self.settings.min_detection_confidence,
            min_face_presence_confidence=self.settings.min_presence_confidence,
            min_tracking_confidence=self.settings.min_tracking_confidence,
            output_face_blendshapes=self.settings.output_face_blendshapes,
            output_facial_transformation_matrixes=self.settings.output_facial_transformation_matrixes,
        )

        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        logger.info("MediaPipe FaceLandmarker initialized (VIDEO mode)")

    def _get_model_path(self) -> str:
        """Get the path to the face landmarker model."""
        import os

        # First try local model file
        local_model = "/home/shubham/airmouse/face_landmarker.task"
        if os.path.exists(local_model):
            return local_model

        # Try to find in mediapipe package
        import mediapipe as mp
        mp_path = os.path.dirname(mp.__file__)

        possible_paths = [
            os.path.join(mp_path, "tasks", "vision", "face_landmarker", "face_landmarker.task"),
            os.path.join(mp_path, "models", "face_landmarker.task"),
            "/usr/local/lib/python3.12/dist-packages/mediapipe/tasks/vision/face_landmarker/face_landmarker.task",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Let MediaPipe download it
        return "face_landmarker.task"

    def process(self, frame: np.ndarray) -> List[Face]:
        """
        Process a frame and detect faces (VIDEO mode with timestamps).

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of detected Face objects
        """
        if frame is None:
            return []

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        h, w = rgb_frame.shape[:2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        mp_image.image_dimensions = (w, h)

        # Detect faces in VIDEO mode
        self._timestamp_ms += 33  # ~30 FPS
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        return self._convert_results(result)

    def _convert_results(self, result: mp_vision.FaceLandmarkerResult) -> List[Face]:
        """Convert MediaPipe results to our Face objects (optimized for low allocation)."""
        faces = []

        if result.face_landmarks:
            # Pre-allocate single landmark list to reuse across faces (max 1 face in our config)
            if not hasattr(self, '_face_landmarks_output'):
                self._face_landmarks_output = [FaceLandmark(0.0, 0.0, 0.0) for _ in range(468)]

            for i, face_landmarks in enumerate(result.face_landmarks):
                # Reuse pre-allocated cache
                cache = self._landmark_cache
                for j, lm in enumerate(face_landmarks):
                    if j < 468:
                        cache[j].x = lm.x
                        cache[j].y = lm.y
                        cache[j].z = lm.z
                        cache[j].visibility = getattr(lm, 'visibility', 1.0)
                        cache[j].presence = getattr(lm, 'presence', 1.0)

                # Get confidence from detection score if available
                confidence = 1.0
                if result.face_blendshapes and i < len(result.face_blendshapes):
                    # Use presence as confidence proxy
                    pass

                # Reuse output list - copy values from cache
                output_landmarks = self._face_landmarks_output
                num_landmarks = min(len(face_landmarks), 468)
                for j in range(num_landmarks):
                    output_landmarks[j].x = cache[j].x
                    output_landmarks[j].y = cache[j].y
                    output_landmarks[j].z = cache[j].z
                    output_landmarks[j].visibility = cache[j].visibility
                    output_landmarks[j].presence = cache[j].presence

                face = Face(
                    landmarks=output_landmarks[:num_landmarks],
                    confidence=confidence
                )
                faces.append(face)

        return faces

    def draw_landmarks(self, frame: np.ndarray, faces: List[Face],
                       draw_key_points: bool = True,
                       draw_connections: bool = False) -> np.ndarray:
        """
        Draw face landmarks on frame.

        Args:
            frame: BGR image to draw on
            faces: List of detected faces
            draw_key_points: Whether to draw key landmarks (eyes, nose, etc.)
            draw_connections: Whether to draw face mesh connections

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for face in faces:
            if not face.landmarks:
                continue

            h, w = frame.shape[:2]

            # Draw key landmarks
            if draw_key_points:
                key_indices = [
                    Face.LEFT_EYE_INNER, Face.LEFT_EYE_OUTER,
                    Face.RIGHT_EYE_INNER, Face.RIGHT_EYE_OUTER,
                    Face.NOSE_TIP, Face.FOREHEAD, Face.CHIN
                ]
                for idx in key_indices:
                    if idx < len(face.landmarks):
                        lm = face.landmarks[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(annotated, (x, y), 3, (0, 255, 255), -1)

                # Draw eye centers
                if face.left_eye_center:
                    x, y = face.left_eye_center.to_pixel(w, h)
                    cv2.circle(annotated, (x, y), 5, (255, 0, 255), -1)
                if face.right_eye_center:
                    x, y = face.right_eye_center.to_pixel(w, h)
                    cv2.circle(annotated, (x, y), 5, (255, 0, 255), -1)

                # Draw eye midpoint
                if face.eye_midpoint:
                    x, y = face.eye_midpoint.to_pixel(w, h)
                    cv2.circle(annotated, (x, y), 6, (0, 255, 0), 2)

            # Draw connections (simplified face mesh)
            if draw_connections:
                # Draw a few key connections for visualization
                connections = [
                    (Face.LEFT_EYE_INNER, Face.LEFT_EYE_OUTER),
                    (Face.RIGHT_EYE_INNER, Face.RIGHT_EYE_OUTER),
                    (Face.LEFT_EYE_INNER, Face.NOSE_TIP),
                    (Face.RIGHT_EYE_INNER, Face.NOSE_TIP),
                    (Face.NOSE_TIP, Face.FOREHEAD),
                ]
                for start_idx, end_idx in connections:
                    if start_idx < len(face.landmarks) and end_idx < len(face.landmarks):
                        start = face.landmarks[start_idx].to_pixel(w, h)
                        end = face.landmarks[end_idx].to_pixel(w, h)
                        cv2.line(annotated, start, end, (255, 255, 0), 1)

        return annotated

    def close(self):
        """Release resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
            logger.info("MediaPipe FaceLandmarker closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test face tracker with camera
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from camera.manager import CameraManager, CameraSettings

    logging.basicConfig(level=logging.INFO)

    print("Testing Face Tracker...")
    print("Press 'q' to quit")

    camera = CameraManager()
    if not camera.open_camera(CameraSettings(device_index=0, width=640, height=480)):
        print("Failed to open camera")
        sys.exit(1)

    tracker = FaceTracker(FaceTrackerSettings(max_faces=1))

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                break

            faces = tracker.process(frame)

            if faces:
                face = faces[0]
                print(f"\rFace detected: conf={face.confidence:.2f}, "
                      f"eye_mid=({face.eye_midpoint.x:.3f},{face.eye_midpoint.y:.3f},{face.eye_midpoint.z:.3f}) "
                      f"forward=({face.forward_vector[0]:.3f},{face.forward_vector[1]:.3f},{face.forward_vector[2]:.3f})" if face.forward_vector is not None else "forward=None", end="")

                annotated = tracker.draw_landmarks(frame, faces)
            else:
                print("\rNo face detected", end="")
                annotated = frame

            cv2.imshow("Face Tracking Test", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.close_camera()
        tracker.close()
        cv2.destroyAllWindows()