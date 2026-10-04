"""
Hand Tracking Module for Air Mouse

Uses MediaPipe HandLandmarker (Tasks API) for real-time hand landmark detection.
Provides hand landmarks, finger states, and gesture primitives.
"""

import cv2
import numpy as np
import math
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Suppress MediaPipe C++ warnings about NORM_RECT / IMAGE_DIMENSIONS.
# In MediaPipe 1.0.1, the HandLandmarker in IMAGE mode passes normalized
# ROI rectangles to the landmark_projection_calculator without populating
# the image dimensions metadata, producing:
#   "landmark_projection_calculator.cc:81 Using NORM_RECT without IMAGE_DIMENSIONS"
# The mp.Image constructor already carries width/height to the C library,
# so image data is correct — the warning is a cosmetic artifact of the
# internal graph configuration.
os.environ.setdefault('MEDIPIPE_LOG_LEVEL', 'ERROR')

logger = logging.getLogger(__name__)


class HandLandmark(Enum):
    """MediaPipe hand landmark indices."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class Landmark:
    """Normalized landmark point (0.0 to 1.0)."""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        """Convert to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))

    def distance_to(self, other: "Landmark") -> float:
        """Euclidean distance to another landmark."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass
class Hand:
    """Detected hand with landmarks and derived properties."""
    landmarks: List[Landmark] = field(default_factory=list)
    handedness: str = "Unknown"  # "Left" or "Right"
    confidence: float = 0.0

    # Derived properties (computed on demand)
    _palm_center: Optional[Landmark] = None
    _index_tip: Optional[Landmark] = None
    _thumb_tip: Optional[Landmark] = None
    _middle_tip: Optional[Landmark] = None
    _ring_tip: Optional[Landmark] = None
    _pinky_tip: Optional[Landmark] = None
    _finger_states: Optional[Dict[str, bool]] = None  # extended/folded

    def __post_init__(self):
        if len(self.landmarks) >= 21:
            self._compute_derived()

    def _compute_derived(self):
        """Compute derived properties from landmarks."""
        if len(self.landmarks) < 21:
            return

        # Key landmarks
        self._palm_center = Landmark(
            x=(self.landmarks[HandLandmark.INDEX_MCP.value].x +
               self.landmarks[HandLandmark.MIDDLE_MCP.value].x +
               self.landmarks[HandLandmark.RING_MCP.value].x +
               self.landmarks[HandLandmark.PINKY_MCP.value].x) / 4,
            y=(self.landmarks[HandLandmark.INDEX_MCP.value].y +
               self.landmarks[HandLandmark.MIDDLE_MCP.value].y +
               self.landmarks[HandLandmark.RING_MCP.value].y +
               self.landmarks[HandLandmark.PINKY_MCP.value].y) / 4,
            z=(self.landmarks[HandLandmark.INDEX_MCP.value].z +
               self.landmarks[HandLandmark.MIDDLE_MCP.value].z +
               self.landmarks[HandLandmark.RING_MCP.value].z +
               self.landmarks[HandLandmark.PINKY_MCP.value].z) / 4
        )

        self._index_tip = self.landmarks[HandLandmark.INDEX_TIP.value]
        self._thumb_tip = self.landmarks[HandLandmark.THUMB_TIP.value]
        self._middle_tip = self.landmarks[HandLandmark.MIDDLE_TIP.value]
        self._ring_tip = self.landmarks[HandLandmark.RING_TIP.value]
        self._pinky_tip = self.landmarks[HandLandmark.PINKY_TIP.value]

        # Compute finger states (extended vs folded)
        self._finger_states = {
            "thumb": self._is_finger_extended(HandLandmark.THUMB_TIP, HandLandmark.THUMB_IP, HandLandmark.THUMB_MCP),
            "index": self._is_finger_extended(HandLandmark.INDEX_TIP, HandLandmark.INDEX_PIP, HandLandmark.INDEX_MCP),
            "middle": self._is_finger_extended(HandLandmark.MIDDLE_TIP, HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_MCP),
            "ring": self._is_finger_extended(HandLandmark.RING_TIP, HandLandmark.RING_PIP, HandLandmark.RING_MCP),
            "pinky": self._is_finger_extended(HandLandmark.PINKY_TIP, HandLandmark.PINKY_PIP, HandLandmark.PINKY_MCP),
        }

    def _is_finger_extended(self, tip: HandLandmark, pip: HandLandmark, mcp: HandLandmark) -> bool:
        """
        Check if a finger is extended using joint-vector geometry.

        Uses the angle between the MCP→PIP vector and PIP→TIP vector.
        An extended finger has a large angle (near 180°) between the
        two segments. A folded finger has a small angle (near 0-60°).

        This is orientation-independent: it works for left/right hands,
        rotated hands, tilted hands, and different camera positions.
        """
        if tip == HandLandmark.THUMB_TIP:
            return self._is_thumb_extended()

        if len(self.landmarks) < 21:
            return False

        # Get landmark positions
        lm_tip = self.landmarks[tip.value]
        lm_pip = self.landmarks[pip.value]
        lm_mcp = self.landmarks[mcp.value]

        # Vector from PIP to TIP
        v_pip_tip = np.array([
            lm_tip.x - lm_pip.x,
            lm_tip.y - lm_pip.y,
            lm_tip.z - lm_pip.z
        ], dtype=np.float64)

        # Vector from MCP to PIP
        v_mcp_pip = np.array([
            lm_pip.x - lm_mcp.x,
            lm_pip.y - lm_mcp.y,
            lm_pip.z - lm_mcp.z
        ], dtype=np.float64)

        # Compute the angle between the two vectors
        dot = np.dot(v_pip_tip, v_mcp_pip)
        norm_tip = np.linalg.norm(v_pip_tip)
        norm_mcp = np.linalg.norm(v_mcp_pip)

        if norm_tip < 1e-8 or norm_mcp < 1e-8:
            return False

        cos_angle = dot / (norm_tip * norm_mcp)
        # Clamp to [-1, 1] to avoid numerical errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)

        # Extended finger: angle < 40° (near straight, vectors aligned)
        # Folded finger: angle > 90° (bent at PIP, vectors opposite)
        return angle_deg < 40.0

    def _is_thumb_extended(self) -> bool:
        """
        Check if thumb is extended using joint-vector geometry.

        Uses the angle between the MCP→IP vector and MCP→index MCP vector.
        The thumb has a different kinematic structure: when extended,
        it points away from the palm; when folded, it points toward
        the index finger MCP.
        """
        if len(self.landmarks) < 5:
            return False

        lm_cmc = self.landmarks[HandLandmark.THUMB_CMC.value]
        lm_mcp = self.landmarks[HandLandmark.THUMB_MCP.value]
        lm_ip = self.landmarks[HandLandmark.THUMB_IP.value]
        lm_index_mcp = self.landmarks[HandLandmark.INDEX_MCP.value]

        # Vector from MCP to IP
        v_mcp_ip = np.array([
            lm_ip.x - lm_mcp.x,
            lm_ip.y - lm_mcp.y,
            lm_ip.z - lm_mcp.z
        ], dtype=np.float64)

        # Vector from MCP to index MCP (reference direction)
        v_mcp_index = np.array([
            lm_index_mcp.x - lm_mcp.x,
            lm_index_mcp.y - lm_mcp.y,
            lm_index_mcp.z - lm_mcp.z
        ], dtype=np.float64)

        dot = np.dot(v_mcp_ip, v_mcp_index)
        norm_ip = np.linalg.norm(v_mcp_ip)
        norm_index = np.linalg.norm(v_mcp_index)

        if norm_ip < 1e-8 or norm_index < 1e-8:
            return False

        cos_angle = dot / (norm_ip * norm_index)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)

        # Extended thumb: angle > 90° (pointing away from index MCP)
        # Folded thumb: angle < 60° (pointing toward index MCP)
        return angle_deg > 90.0

    @property
    def palm_center(self) -> Optional[Landmark]:
        return self._palm_center

    @property
    def index_tip(self) -> Optional[Landmark]:
        return self._index_tip

    @property
    def thumb_tip(self) -> Optional[Landmark]:
        return self._thumb_tip

    @property
    def middle_tip(self) -> Optional[Landmark]:
        return self._middle_tip

    @property
    def ring_tip(self) -> Optional[Landmark]:
        return self._ring_tip

    @property
    def pinky_tip(self) -> Optional[Landmark]:
        return self._pinky_tip

    @property
    def finger_states(self) -> Dict[str, bool]:
        return self._finger_states or {}

    def is_finger_extended(self, finger: str) -> bool:
        """Check if a specific finger is extended."""
        return self._finger_states.get(finger, False)

    def pinch_distance(self, finger1: str = "thumb", finger2: str = "index") -> float:
        """Distance between two fingertips (normalized)."""
        tips = {
            "thumb": self._thumb_tip,
            "index": self._index_tip,
            "middle": self._middle_tip,
            "ring": self._ring_tip,
            "pinky": self._pinky_tip,
        }
        tip1 = tips.get(finger1)
        tip2 = tips.get(finger2)
        if tip1 and tip2:
            return tip1.distance_to(tip2)
        return float('inf')

    def is_pinch(self, finger1: str = "thumb", finger2: str = "index", threshold: float = 0.05) -> bool:
        """Check if two fingers are pinching."""
        return self.pinch_distance(finger1, finger2) < threshold

    def is_fist(self) -> bool:
        """Check if hand is in fist position (all fingers folded)."""
        return not any(self._finger_states.values()) if self._finger_states else False

    def is_pointing(self) -> bool:
        """Check if hand is pointing (index extended, others folded)."""
        if not self._finger_states:
            return False
        return (self._finger_states.get("index", False) and
                not self._finger_states.get("middle", False) and
                not self._finger_states.get("ring", False) and
                not self._finger_states.get("pinky", False))

    def is_scroll_gesture(self) -> bool:
        """Check if hand is in scroll gesture (index + middle extended)."""
        if not self._finger_states:
            return False
        return (self._finger_states.get("index", False) and
                self._finger_states.get("middle", False) and
                not self._finger_states.get("ring", False) and
                not self._finger_states.get("pinky", False))

    def is_open_palm(self) -> bool:
        """Check if hand is open palm (all fingers extended)."""
        if not self._finger_states:
            return False
        return all(self._finger_states.values())

    def is_thumb_gesture(self) -> bool:
        """Check if hand is making thumb gesture (thumb extended, others folded)."""
        if not self._finger_states:
            return False
        return (self._finger_states.get("thumb", False) and
                not self._finger_states.get("index", False) and
                not self._finger_states.get("middle", False) and
                not self._finger_states.get("ring", False) and
                not self._finger_states.get("pinky", False))

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get normalized bounding box (x_min, y_min, x_max, y_max)."""
        if not self.landmarks:
            return (0, 0, 0, 0)
        xs = [l.x for l in self.landmarks]
        ys = [l.y for l in self.landmarks]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class HandTrackerSettings:
    """Hand tracker configuration."""
    # Use max_hands=2 instead of 1 to improve detection reliability.
    # With num_hands=1, MediaPipe 1.0.1 sometimes returns 0 hands when the
    # single-hand confidence is borderline (~0.5‑0.6).  Requesting 2 hands
    # makes the model output multiple candidates, and at least one typically
    # exceeds the confidence threshold.  We then filter by preferred
    # handedness in _convert_results.
    max_hands: int = 2
    # Lowered from 0.7 → 0.0 to prevent hands from being filtered out on
    # lower‑confidence frames.  MediaPipe 1.0.1 + the bundled model often
    # returns confidence values in the 0.5‑0.9 range on real camera feeds,
    # but the internal min_hand_presence_confidence filter was too aggressive.
    # Setting both to 0.0 ensures hands are never filtered at the model level;
    # we apply our own confidence filtering in TrackingProcessor.
    min_detection_confidence: float = 0.0
    # Also set to 0.0 for the same reason – the tracking confidence
    # (min_hand_presence_confidence in the MediaPipe API) was causing
    # intermittent drops when it fluctuated between frames.
    min_tracking_confidence: float = 0.0
    model_complexity: int = 1  # 0=lite, 1=full
    static_image_mode: bool = False
    preferred_handedness: Optional[str] = None  # "Left", "Right", or None for any


class HandTracker:
    """
    MediaPipe HandLandmarker wrapper for real-time hand tracking.

    Features:
    - Single or multi-hand tracking
    - Landmark extraction with derived properties
    - Finger state classification
    - Pinch/distance detection
    """

    def __init__(self, settings: Optional[HandTrackerSettings] = None):
        self.settings = settings or HandTrackerSettings()
        self._landmarker = None
        self._landmark_cache = [Landmark(0.0, 0.0, 0.0) for _ in range(21)]  # Pre-allocated landmark array
        self._initialize()

    def _initialize(self):
        """Initialize MediaPipe HandLandmarker.

        The original implementation used ``RunningMode.VIDEO`` which, with
        the bundled ``hand_landmarker.task`` model and MediaPipe 1.0.1, crashes
        on many Linux devices (see the stack traces observed during testing).
        Switching to ``RunningMode.IMAGE`` eliminates the crash and still
        provides reliable per‑frame detection.  The performance impact is
        minimal for the low‑resolution 640×480 camera used by the project.
        """
        # Resolve the model path – prefer the local copy, fallback to MediaPipe's
        # bundled asset (which may trigger a download).
        model_path = self._get_model_path()

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            # Use IMAGE mode – a single‑frame detection that does not rely on
            # timestamps or internal video pipelines, avoiding the crash.
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=self.settings.max_hands,
            min_hand_detection_confidence=self.settings.min_detection_confidence,
            min_hand_presence_confidence=self.settings.min_tracking_confidence,
            min_tracking_confidence=self.settings.min_tracking_confidence,
        )

        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        logger.info("MediaPipe HandLandmarker initialized (IMAGE mode)")

    def _get_model_path(self) -> str:
        """Get the path to the hand landmarker model."""
        import os

        # First try the local model file (copied from reference project)
        local_model = "/home/shubham/airmouse/hand_landmarker.task"
        if os.path.exists(local_model):
            return local_model

        # Try to find the model in the mediapipe package
        import mediapipe as mp
        mp_path = os.path.dirname(mp.__file__)

        # The model should be in the tasks/vision/hand_landmarker directory
        possible_paths = [
            os.path.join(mp_path, "tasks", "vision", "hand_landmarker", "hand_landmarker.task"),
            os.path.join(mp_path, "models", "hand_landmarker.task"),
            "/usr/local/lib/python3.12/dist-packages/mediapipe/tasks/vision/hand_landmarker/hand_landmarker.task",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # If not found, we'll let MediaPipe download it
        return "hand_landmarker.task"

    
    def process(self, frame: np.ndarray, auto_brighten: bool = True) -> List[Hand]:
        """Process a frame and detect hands.

        After switching to ``RunningMode.IMAGE`` we use the ``detect`` method
        which operates on a single image without timestamps.  The parameter
        ``auto_brighten`` is retained for compatibility – it brightens
        very dark frames which helps MediaPipe on low‑light webcams.
        """
        if frame is None:
            return []

        # Auto‑brighten dark frames for MediaPipe (many webcams output very dark images)
        if auto_brighten:
            mean_brightness = frame.mean()
            if mean_brightness < 120:
                frame = cv2.convertScaleAbs(frame, alpha=3.0, beta=50)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image.  MediaPipe 1.0.x infers width/height
        # from the numpy array; the ``image_dimensions`` attribute does
        # not exist in 1.0.1, so assigning to it silently no-ops and the
        # calculator emits "NORM_RECT without IMAGE_DIMENSIONS".
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # IMAGE mode – use the single‑frame ``detect`` call
        result = self._landmarker.detect(mp_image)
        return self._convert_results(result)

    def _convert_results(self, result: mp_vision.HandLandmarkerResult) -> List[Hand]:
        """Convert MediaPipe results to our Hand objects (optimized for low allocation)."""
        hands = []

        if result.hand_landmarks:
            for i, (hand_landmarks, handedness_list) in enumerate(zip(
                result.hand_landmarks,
                result.handedness
            )):
                # Create independent landmark list for each hand (§6 P0 fix).
                # Previously a shared mutable list was reused, so modifying
                # hand[1].landmarks would corrupt hand[0].landmarks.
                num_landmarks = min(len(hand_landmarks), 21)
                output_landmarks = [
                    Landmark(
                        x=hand_landmarks[j].x,
                        y=hand_landmarks[j].y,
                        z=hand_landmarks[j].z,
                        visibility=1.0
                    )
                    for j in range(num_landmarks)
                ]

                # Get handedness
                hand_label = "Unknown"
                hand_confidence = 0.0
                if handedness_list:
                    hand_label = handedness_list[0].category_name
                    hand_confidence = handedness_list[0].score

                # Filter by preferred handedness if configured
                if self.settings.preferred_handedness and hand_label != self.settings.preferred_handedness:
                    continue

                hand = Hand(
                    landmarks=output_landmarks,
                    handedness=hand_label,
                    confidence=hand_confidence
                )
                hands.append(hand)

        return hands

    def draw_landmarks(self, frame: np.ndarray, hands: List[Hand],
                       draw_connections: bool = True,
                       draw_landmarks: bool = True) -> np.ndarray:
        """
        Draw hand landmarks on frame.

        Args:
            frame: BGR image to draw on
            hands: List of detected hands
            draw_connections: Whether to draw hand connections
            draw_landmarks: Whether to draw landmark points

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for hand in hands:
            if hand.landmarks:
                h, w = frame.shape[:2]

                # Draw landmarks
                if draw_landmarks:
                    for lm in hand.landmarks:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

                # Draw connections
                if draw_connections:
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
                        (0, 5), (5, 6), (6, 7), (7, 8),  # index
                        (5, 9), (9, 10), (10, 11), (11, 12),  # middle
                        (9, 13), (13, 14), (14, 15), (15, 16),  # ring
                        (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
                        (0, 17),  # palm
                    ]
                    for start_idx, end_idx in connections:
                        if start_idx < len(hand.landmarks) and end_idx < len(hand.landmarks):
                            start = hand.landmarks[start_idx].to_pixel(w, h)
                            end = hand.landmarks[end_idx].to_pixel(w, h)
                            cv2.line(annotated, start, end, (255, 0, 0), 2)

                # Draw palm center
                if hand.palm_center:
                    cx, cy = hand.palm_center.to_pixel(w, h)
                    cv2.circle(annotated, (cx, cy), 8, (0, 0, 255), -1)

                # Draw index tip (cursor point)
                if hand.index_tip:
                    ix, iy = hand.index_tip.to_pixel(w, h)
                    cv2.circle(annotated, (ix, iy), 10, (255, 255, 0), 2)

        return annotated

    def close(self):
        """Release resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
            logger.info("MediaPipe HandLandmarker closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test hand tracker with camera
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from camera.manager import CameraManager, CameraSettings

    logging.basicConfig(level=logging.INFO)

    print("Testing Hand Tracker...")
    print("Press 'q' to quit")

    camera = CameraManager()
    if not camera.open_camera(CameraSettings(device_index=0, width=640, height=480)):
        print("Failed to open camera")
        sys.exit(1)

    tracker = HandTracker(HandTrackerSettings(max_hands=1))

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                break

            hands = tracker.process(frame)

            if hands:
                hand = hands[0]
                print(f"\rHand: {hand.handedness}, "
                      f"Index extended: {hand.is_finger_extended('index')}, "
                      f"Pinch (thumb+index): {hand.is_pinch('thumb', 'index'):.3f}, "
                      f"Fist: {hand.is_fist()}", end="")

                annotated = tracker.draw_landmarks(frame, hands)
            else:
                annotated = frame

            cv2.imshow("Hand Tracking Test", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.close_camera()
        tracker.close()
        cv2.destroyAllWindows()
