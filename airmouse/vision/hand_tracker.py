"""
Hand Tracking Module for Air Mouse

Uses MediaPipe HandLandmarker (Tasks API) for real-time hand landmark detection.
Provides hand landmarks, finger states, and gesture primitives.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


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
        """Check if finger is extended (tip above PIP joint in y)."""
        # For thumb, use different logic (check x distance from palm)
        if tip == HandLandmark.THUMB_TIP:
            return self.landmarks[tip.value].x < self.landmarks[mcp.value].x - 0.02

        # For other fingers: tip y < pip y (higher up in image = smaller y)
        return self.landmarks[tip.value].y < self.landmarks[pip.value].y - 0.015

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
    max_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
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
        """Initialize MediaPipe HandLandmarker."""
        # Get model path - use the bundled model
        model_path = self._get_model_path()

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,  # VIDEO mode for temporal coherence (faster)
            num_hands=self.settings.max_hands,
            min_hand_detection_confidence=self.settings.min_detection_confidence,
            min_hand_presence_confidence=self.settings.min_tracking_confidence,
            min_tracking_confidence=self.settings.min_tracking_confidence,
        )

        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        logger.info("MediaPipe HandLandmarker initialized (VIDEO mode)")

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
        """
        Process a frame and detect hands (VIDEO mode with timestamps).

        Args:
            frame: BGR image from OpenCV
            auto_brighten: Whether to automatically brighten dark frames for MediaPipe

        Returns:
            List of detected Hand objects
        """
        if frame is None:
            return []

        # Auto-brighten dark frames for MediaPipe (many webcams output very dark images)
        if auto_brighten:
            # Check mean brightness - most webcams need brightening for MediaPipe
            mean_brightness = frame.mean()
            if mean_brightness < 120:  # Threshold for "too dark" - adjust based on your camera
                frame = cv2.convertScaleAbs(frame, alpha=3.0, beta=50)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image with dimensions to avoid NORM_RECT warning
        h, w = rgb_frame.shape[:2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        mp_image.image_dimensions = (w, h)

        # Use the landmarker in VIDEO mode with timestamps
        self._timestamp_ms += 33  # ~30 FPS
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        return self._convert_results(result)

    def _convert_results(self, result: mp_vision.HandLandmarkerResult) -> List[Hand]:
        """Convert MediaPipe results to our Hand objects (optimized for low allocation)."""
        hands = []

        # Pre-allocate output landmark list to reuse across hands (max 1 hand in our config)
        if not hasattr(self, '_hand_landmarks_output'):
            self._hand_landmarks_output = [Landmark(0.0, 0.0, 0.0) for _ in range(21)]

        if result.hand_landmarks:
            for i, (hand_landmarks, handedness_list) in enumerate(zip(
                result.hand_landmarks,
                result.handedness
            )):
                # Reuse pre-allocated output list - copy values directly
                output_landmarks = self._hand_landmarks_output
                num_landmarks = min(len(hand_landmarks), 21)
                for j in range(num_landmarks):
                    lm = hand_landmarks[j]
                    output_landmarks[j].x = lm.x
                    output_landmarks[j].y = lm.y
                    output_landmarks[j].z = lm.z
                    output_landmarks[j].visibility = 1.0

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
                    landmarks=output_landmarks[:num_landmarks],
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