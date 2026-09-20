"""
Calibration System for Air Mouse

Implements spec §26: calibration must establish:
- camera framing
- usable hand region
- screen mapping
- sensitivity
- preferred control area
- handedness
- baseline position
- movement scale

Calibration must be quick enough that users will actually use it.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum

from .hand_tracker import Hand, HandLandmark

logger = logging.getLogger(__name__)


class CalibrationPhase(Enum):
    """Phases of the calibration sequence."""
    IDLE = 0                    # Not calibrating
    POSITIONING = 1             # User positions hand in camera view
    FRAME_CAMERA = 2            # Establish camera framing
    HAND_REGION = 3             # Map usable hand region
    SCREEN_CORNERS = 4          # Map hand positions to screen corners
    SENSITIVITY = 5             # Determine comfortable sensitivity
    HANDEDNESS = 6              # Confirm preferred hand
    BASELINE = 7                # Establish baseline position
    MOVEMENT_SCALE = 8          # Measure movement scale
    COMPLETE = 9                # Calibration finished


@dataclass
class CalibrationConfig:
    """Configuration for calibration."""
    # Timeout per phase (seconds)
    phase_timeout: float = 10.0
    total_timeout: float = 60.0

    # Minimum frames per phase for statistical validity
    min_frames_per_phase: int = 30

    # Confidence threshold for accepting calibration data
    min_confidence: float = 0.7

    # Corner capture dwell time (seconds)
    corner_dwell_time: float = 1.0

    # Movement scale measurement distance
    movement_scale_distance: float = 0.5  # Normalized distance


@dataclass
class CalibrationResult:
    """Result of a completed calibration."""
    # Camera framing (0-1 normalized coordinates)
    camera_frame_min_x: float = 0.0
    camera_frame_max_x: float = 1.0
    camera_frame_min_y: float = 0.0
    camera_frame_max_y: float = 1.0

    # Usable hand region (subset of camera frame)
    hand_region_min_x: float = 0.1
    hand_region_max_x: float = 0.9
    hand_region_min_y: float = 0.1
    hand_region_max_y: float = 0.9

    # Screen corner mappings (hand normalized -> screen corners)
    top_left_hand: Tuple[float, float] = (0.1, 0.1)
    top_right_hand: Tuple[float, float] = (0.9, 0.1)
    bottom_left_hand: Tuple[float, float] = (0.1, 0.9)
    bottom_right_hand: Tuple[float, float] = (0.9, 0.9)

    # Preferred sensitivity
    sensitivity_mode: str = "normal"
    base_sensitivity: float = 1.0

    # Handedness
    preferred_hand: str = "Right"

    # Baseline position (center of control area)
    baseline_x: float = 0.5
    baseline_y: float = 0.5

    # Movement scale (normalized hand movement -> screen movement ratio)
    movement_scale_x: float = 1.0
    movement_scale_y: float = 1.0

    # Calibration quality metrics
    frames_captured: int = 0
    avg_confidence: float = 0.0
    duration_seconds: float = 0.0
    completed: bool = False

    # Timestamp
    calibrated_at: float = field(default_factory=time.time)


class Calibrator:
    """
    Manages the calibration sequence.

    Quick calibration (< 30 seconds) establishes:
    1. Camera framing - detects usable field of view
    2. Hand region - finds where hand naturally operates
    3. Screen corners - maps hand positions to screen edges
    4. Sensitivity - measures comfortable speed
    5. Handedness - confirms preferred hand
    6. Baseline - sets neutral position
    7. Movement scale - calibrates gain
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self._phase = CalibrationPhase.IDLE
        self._phase_start_time = 0.0
        self._calibration_start_time = 0.0
        self._result = CalibrationResult()
        self._frame_buffer: List[Hand] = []
        self._corner_buffer: List[Hand] = []
        self._current_corner = 0
        self._corner_dwell_start = 0.0
        self._callback: Optional[callable] = None

    @property
    def phase(self) -> CalibrationPhase:
        return self._phase

    @property
    def result(self) -> CalibrationResult:
        return self._result

    @property
    def progress(self) -> float:
        """Overall calibration progress (0.0 to 1.0)."""
        if self._phase == CalibrationPhase.IDLE:
            return 0.0
        if self._phase == CalibrationPhase.COMPLETE:
            return 1.0
        # Approximate progress by phase
        phase_order = list(CalibrationPhase)
        idx = phase_order.index(self._phase)
        return idx / (len(phase_order) - 1)

    @property
    def phase_progress(self) -> float:
        """Progress within current phase (0.0 to 1.0)."""
        if self._phase in (CalibrationPhase.IDLE, CalibrationPhase.COMPLETE):
            return 1.0
        elapsed = time.time() - self._phase_start_time
        return min(1.0, elapsed / self.config.phase_timeout)

    def set_callback(self, callback: callable):
        """Set callback for phase changes and completion."""
        self._callback = callback

    def start(self) -> CalibrationPhase:
        """Start calibration sequence."""
        self._phase = CalibrationPhase.POSITIONING
        self._calibration_start_time = time.time()
        self._phase_start_time = time.time()
        self._frame_buffer.clear()
        self._result = CalibrationResult()
        logger.info("Calibration started")
        return self._phase

    def process(self, hand: Optional[Hand]) -> CalibrationPhase:
        """
        Process a frame during calibration.

        Args:
            hand: Detected hand or None

        Returns:
            Current calibration phase
        """
        if self._phase == CalibrationPhase.IDLE:
            return self._phase

        if self._phase == CalibrationPhase.COMPLETE:
            return self._phase

        current_time = time.time()

        # Check total timeout
        if current_time - self._calibration_start_time > self.config.total_timeout:
            logger.warning("Calibration timed out")
            self._complete(complete=False)
            return self._phase

        # Check phase timeout
        if current_time - self._phase_start_time > self.config.phase_timeout:
            logger.warning(f"Phase {self._phase.name} timed out, advancing")
            self._advance_phase()

        # Process based on current phase
        if hand and hand.confidence >= self.config.min_confidence:
            self._frame_buffer.append(hand)

        if self._phase == CalibrationPhase.POSITIONING:
            self._process_positioning(hand)
        elif self._phase == CalibrationPhase.FRAME_CAMERA:
            self._process_frame_camera()
        elif self._phase == CalibrationPhase.HAND_REGION:
            self._process_hand_region()
        elif self._phase == CalibrationPhase.SCREEN_CORNERS:
            self._process_screen_corners(hand, current_time)
        elif self._phase == CalibrationPhase.SENSITIVITY:
            self._process_sensitivity()
        elif self._phase == CalibrationPhase.HANDEDNESS:
            self._process_handedness()
        elif self._phase == CalibrationPhase.BASELINE:
            self._process_baseline()
        elif self._phase == CalibrationPhase.MOVEMENT_SCALE:
            self._process_movement_scale()

        return self._phase

    def _process_positioning(self, hand: Optional[Hand]):
        """Wait for user to position hand in view."""
        if hand and hand.confidence >= self.config.min_confidence:
            self._advance_phase()

    def _process_frame_camera(self):
        """Establish camera framing from hand detections."""
        if len(self._frame_buffer) >= self.config.min_frames_per_phase:
            xs = [h.palm_center.x for h in self._frame_buffer if h.palm_center]
            ys = [h.palm_center.y for h in self._frame_buffer if h.palm_center]
            if xs and ys:
                margin = 0.05
                self._result.camera_frame_min_x = max(0.0, min(xs) - margin)
                self._result.camera_frame_max_x = min(1.0, max(xs) + margin)
                self._result.camera_frame_min_y = max(0.0, min(ys) - margin)
                self._result.camera_frame_max_y = min(1.0, max(ys) + margin)
                self._frame_buffer.clear()
                self._advance_phase()

    def _process_hand_region(self):
        """Map the usable hand operating region."""
        if len(self._frame_buffer) >= self.config.min_frames_per_phase:
            xs = [h.index_tip.x for h in self._frame_buffer if h.index_tip]
            ys = [h.index_tip.y for h in self._frame_buffer if h.index_tip]
            if xs and ys:
                # Use inner 80% of observed range
                x_range = max(xs) - min(xs)
                y_range = max(ys) - min(ys)
                margin_x = x_range * 0.1
                margin_y = y_range * 0.1
                self._result.hand_region_min_x = max(0.0, min(xs) + margin_x)
                self._result.hand_region_max_x = min(1.0, max(xs) - margin_x)
                self._result.hand_region_min_y = max(0.0, min(ys) + margin_y)
                self._result.hand_region_max_y = min(1.0, max(ys) - margin_y)
                self._frame_buffer.clear()
                self._advance_phase()

    def _process_screen_corners(self, hand: Optional[Hand], current_time: float):
        """Map hand positions to screen corners."""
        corners = [
            ("top_left", self._result.top_left_hand),
            ("top_right", self._result.top_right_hand),
            ("bottom_left", self._result.bottom_left_hand),
            ("bottom_right", self._result.bottom_right_hand),
        ]

        if self._current_corner >= len(corners):
            self._advance_phase()
            return

        corner_name, _ = corners[self._current_corner]

        if hand and hand.confidence >= self.config.min_confidence and hand.index_tip:
            self._corner_buffer.append(hand)
            if self._corner_dwell_start == 0.0:
                self._corner_dwell_start = current_time

            # Check dwell time
            if current_time - self._corner_dwell_start >= self.config.corner_dwell_time:
                # Average the buffered positions
                xs = [h.index_tip.x for h in self._corner_buffer if h.index_tip]
                ys = [h.index_tip.y for h in self._corner_buffer if h.index_tip]
                if xs and ys:
                    avg_x = sum(xs) / len(xs)
                    avg_y = sum(ys) / len(ys)

                    if corner_name == "top_left":
                        self._result.top_left_hand = (avg_x, avg_y)
                    elif corner_name == "top_right":
                        self._result.top_right_hand = (avg_x, avg_y)
                    elif corner_name == "bottom_left":
                        self._result.bottom_left_hand = (avg_x, avg_y)
                    elif corner_name == "bottom_right":
                        self._result.bottom_right_hand = (avg_x, avg_y)

                    logger.info(f"Captured {corner_name}: ({avg_x:.3f}, {avg_y:.3f})")

                    self._corner_buffer.clear()
                    self._corner_dwell_start = 0.0
                    self._current_corner += 1
        else:
            # Hand lost, reset dwell
            self._corner_buffer.clear()
            self._corner_dwell_start = 0.0

    def _process_sensitivity(self):
        """Determine comfortable sensitivity from movement speed."""
        if len(self._frame_buffer) >= self.config.min_frames_per_phase:
            # Measure average movement speed
            speeds = []
            for i in range(1, len(self._frame_buffer)):
                h1 = self._frame_buffer[i-1]
                h2 = self._frame_buffer[i]
                if h1.index_tip and h2.index_tip:
                    dx = h2.index_tip.x - h1.index_tip.x
                    dy = h2.index_tip.y - h1.index_tip.y
                    speed = (dx*dx + dy*dy)**0.5
                    speeds.append(speed)

            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                # Map speed to sensitivity mode
                if avg_speed < 0.005:
                    self._result.sensitivity_mode = "fast"
                    self._result.base_sensitivity = 1.5
                elif avg_speed < 0.015:
                    self._result.sensitivity_mode = "normal"
                    self._result.base_sensitivity = 1.0
                else:
                    self._result.sensitivity_mode = "precision"
                    self._result.base_sensitivity = 0.7

            self._frame_buffer.clear()
            self._advance_phase()

    def _process_handedness(self):
        """Confirm preferred hand."""
        if len(self._frame_buffer) >= self.config.min_frames_per_phase:
            left_count = sum(1 for h in self._frame_buffer if h.handedness == "Left")
            right_count = sum(1 for h in self._frame_buffer if h.handedness == "Right")

            if left_count > right_count:
                self._result.preferred_hand = "Left"
            else:
                self._result.preferred_hand = "Right"

            self._frame_buffer.clear()
            self._advance_phase()

    def _process_baseline(self):
        """Establish baseline (neutral) hand position."""
        if len(self._frame_buffer) >= self.config.min_frames_per_phase:
            xs = [h.palm_center.x for h in self._frame_buffer if h.palm_center]
            ys = [h.palm_center.y for h in self._frame_buffer if h.palm_center]
            if xs and ys:
                self._result.baseline_x = sum(xs) / len(xs)
                self._result.baseline_y = sum(ys) / len(ys)
                self._frame_buffer.clear()
                self._advance_phase()

    def _process_movement_scale(self):
        """Measure movement scale (hand movement -> cursor movement)."""
        if len(self._frame_buffer) >= self.config.min_frames_per_phase:
            # Measure average displacement from baseline
            dxs = []
            dys = []
            for h in self._frame_buffer:
                if h.index_tip:
                    dx = h.index_tip.x - self._result.baseline_x
                    dy = h.index_tip.y - self._result.baseline_y
                    dxs.append(dx)
                    dys.append(dy)

            if dxs and dys:
                avg_dx = sum(abs(d) for d in dxs) / len(dxs)
                avg_dy = sum(abs(d) for d in dys) / len(dys)

                # Normalize to target distance
                target = self.config.movement_scale_distance
                if avg_dx > 0:
                    self._result.movement_scale_x = target / avg_dx
                if avg_dy > 0:
                    self._result.movement_scale_y = target / avg_dy

            self._frame_buffer.clear()
            self._complete(complete=True)

    def _advance_phase(self):
        """Advance to next calibration phase."""
        phase_order = [
            CalibrationPhase.POSITIONING,
            CalibrationPhase.FRAME_CAMERA,
            CalibrationPhase.HAND_REGION,
            CalibrationPhase.SCREEN_CORNERS,
            CalibrationPhase.SENSITIVITY,
            CalibrationPhase.HANDEDNESS,
            CalibrationPhase.BASELINE,
            CalibrationPhase.MOVEMENT_SCALE,
            CalibrationPhase.COMPLETE,
        ]

        current_idx = phase_order.index(self._phase)
        if current_idx + 1 < len(phase_order):
            self._phase = phase_order[current_idx + 1]
            self._phase_start_time = time.time()
            self._frame_buffer.clear()
            logger.info(f"Calibration phase: {self._phase.name}")

            if self._callback:
                try:
                    self._callback(self._phase)
                except Exception as e:
                    logger.error(f"Calibration callback error: {e}")

    def _complete(self, complete: bool):
        """Finish calibration."""
        self._phase = CalibrationPhase.COMPLETE
        self._result.completed = complete
        self._result.duration_seconds = time.time() - self._calibration_start_time
        self._result.frames_captured = sum(1 for _ in self._frame_buffer) if hasattr(self._frame_buffer, '__len__') else 0

        if complete:
            # Compute average confidence
            all_frames = []  # Would accumulate all frames
            if all_frames:
                confidences = [f.confidence for f in all_frames]
                self._result.avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info(f"Calibration {'completed' if complete else 'cancelled'} in {self._result.duration_seconds:.1f}s")

        if self._callback:
            try:
                self._callback(self._phase)
            except Exception as e:
                logger.error(f"Calibration completion callback error: {e}")

    def cancel(self):
        """Cancel calibration."""
        if self._phase != CalibrationPhase.IDLE:
            self._complete(complete=False)
            self._phase = CalibrationPhase.IDLE

    def reset(self):
        """Reset calibrator state."""
        self._phase = CalibrationPhase.IDLE
        self._result = CalibrationResult()
        self._frame_buffer.clear()
        self._corner_buffer.clear()
        self._current_corner = 0
        self._corner_dwell_start = 0.0


def apply_calibration(result: CalibrationResult, config) -> None:
    """
    Apply calibration result to a CursorConfig or similar.

    Args:
        result: CalibrationResult to apply
        config: Configuration object to update (CursorConfig or similar)
    """
    # Apply sensitivity
    if hasattr(config, 'sensitivity_mode'):
        if result.sensitivity_mode == "fast":
            config.sensitivity_mode = "fast"
        elif result.sensitivity_mode == "precision":
            config.sensitivity_mode = "precision"
        else:
            config.sensitivity_mode = "normal"

    if hasattr(config, 'base_sensitivity'):
        config.base_sensitivity = result.base_sensitivity

    # Apply handedness
    if hasattr(config, 'preferred_handedness'):
        config.preferred_handedness = result.preferred_hand

    # Apply screen mapping (would require cursor controller support)
    logger.info("Applied calibration to config")