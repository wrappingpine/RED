"""
Main Control Loop for Air Mouse

Coordinates the pipeline:
Camera → Hand Tracker → Cursor Controller → Gesture Recognizer → Virtual Mouse

Handles:
- Frame processing loop
- Gesture to action mapping
- Safety features (emergency stop, corner escape)
- Coordinate mapping and smoothing
- Performance monitoring
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from enum import Enum
import queue
import numpy as np

from ..camera.manager import CameraManager, CameraSettings, CameraInfo
from ..vision.hand_tracker import HandTracker, HandTrackerSettings, Hand
from ..vision.face_tracker import FaceTracker, FaceTrackerSettings, Face
from ..vision.gestures import (
    GestureRecognizer, GestureConfig, GestureEvent, GestureType, TrackingState
)
from ..vision.tracking_processor import TrackingProcessor, TrackingConfig, TrackedHand
from ..input.uinput_mouse import VirtualMouse, UInputDeviceConfig
from .cursor import CursorController, CursorConfig, SmoothingAlgorithm, get_screen_size, SensitivityMode
from ..debug.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class AirMouseState(Enum):
    """Air mouse operational states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class AirMouseConfig:
    """Complete configuration for Air Mouse."""
    # Camera settings
    camera: CameraSettings = field(default_factory=CameraSettings)

    # Hand tracker settings
    hand_tracker: HandTrackerSettings = field(default_factory=HandTrackerSettings)

    # Cursor control settings
    cursor: CursorConfig = field(default_factory=CursorConfig)

    # Gesture recognition settings
    gestures: GestureConfig = field(default_factory=GestureConfig)

    # Tracking processor settings
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    # Virtual mouse settings
    virtual_mouse: UInputDeviceConfig = field(default_factory=UInputDeviceConfig)

    # Performance
    target_fps: int = 60
    max_frame_time: float = 0.1  # seconds

    # Safety
    emergency_stop_corner: bool = True  # Move to corner to stop
    corner_threshold: int = 10  # pixels from corner
    corner_hold_time: float = 1.0  # seconds in corner to trigger stop


@dataclass
class PerformanceStats:
    """Performance metrics."""
    fps: float = 0.0
    frame_time_ms: float = 0.0
    hand_detection_time_ms: float = 0.0
    gesture_time_ms: float = 0.0
    cursor_time_ms: float = 0.0
    mouse_time_ms: float = 0.0
    frames_processed: int = 0
    frames_dropped: int = 0
    last_update: float = field(default_factory=time.time)


class AirMouseController:
    """
    Main controller coordinating all air mouse components.
    """

    def __init__(self, config: Optional[AirMouseConfig] = None,
                 status_callback: Optional[Callable[[str, dict], None]] = None):
        self.config = config or AirMouseConfig()
        self.status_callback = status_callback

        # Components
        self.camera = CameraManager()
        self.hand_tracker: Optional[HandTracker] = None
        self.face_tracker: Optional[FaceTracker] = None
        self.cursor_controller: Optional[CursorController] = None
        self.gesture_recognizer: Optional[GestureRecognizer] = None
        self.virtual_mouse: Optional[VirtualMouse] = None
        self.tracking_processor: Optional[TrackingProcessor] = None

        # State
        self.state = AirMouseState.STOPPED
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

        # Performance
        self.stats = PerformanceStats()
        self._frame_times: List[float] = []
        self._last_frame_time = 0.0

        # Safety
        self._corner_start_time = 0.0
        self._in_corner = False
        self._last_frame = None

        # Callbacks for GUI
        self.on_hand_detected: Optional[Callable[[Hand], None]] = None
        self.on_gesture: Optional[Callable[[GestureEvent], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_stats_update: Optional[Callable[[PerformanceStats], None]] = None
        self.on_frame_processed: Optional[Callable[[np.ndarray, List[Hand]], None]] = None

        # Performance monitor
        self.performance_monitor = PerformanceMonitor()
        self._debug_overlay_enabled = False

    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing Air Mouse...")

        try:
            # Detect cameras first
            cameras = self.camera.detect_cameras()
            if not cameras:
                self._set_error("No cameras detected")
                return False

            logger.info(f"Found {len(cameras)} camera(s)")
            for cam in cameras:
                logger.info(f"  {cam}")

            # Use first available camera if not specified
            if self.config.camera.device_index < 0:
                self.config.camera.device_index = cameras[0].index
                logger.info(f"Auto-selected camera index {self.config.camera.device_index}")

            # Open camera
            if not self.camera.open_camera(self.config.camera):
                self._set_error("Failed to open camera")
                return False

            # Get actual camera resolution
            actual_width, actual_height = self.camera.get_resolution()
            logger.info(f"Camera resolution: {actual_width}x{actual_height}")

            # Initialize hand tracker
            self.hand_tracker = HandTracker(self.config.hand_tracker)

            # Initialize face tracker (required for head-relative mode)
            if self.config.tracking.use_head_relative:
                self.face_tracker = FaceTracker(FaceTrackerSettings(
                    max_faces=1,
                    min_detection_confidence=self.config.tracking.min_face_confidence,
                    min_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                ))
                logger.info("Face tracker initialized for head-relative mode")

            # Initialize cursor controller
            screen_width, screen_height = get_screen_size()
            self.config.cursor.screen_width = screen_width
            self.config.cursor.screen_height = screen_height
            self.config.cursor.camera_width = actual_width
            self.config.cursor.camera_height = actual_height
            self.cursor_controller = CursorController(self.config.cursor)

            # Initialize gesture recognizer
            self.gesture_recognizer = GestureRecognizer(
                config=self.config.gestures,
                callback=self.on_gesture
            )

            # Initialize tracking processor
            self.tracking_processor = TrackingProcessor(self.config.tracking)

            # Initialize virtual mouse
            self.virtual_mouse = VirtualMouse(self.config.virtual_mouse)
            if not self.virtual_mouse.create():
                self._set_error("Failed to create virtual mouse device")
                return False

            logger.info("All components initialized successfully")
            self._set_status("initialized", {"screen": (screen_width, screen_height)})
            return True

        except Exception as e:
            logger.exception("Initialization failed")
            self._set_error(f"Initialization failed: {e}")
            return False

    def start(self) -> bool:
        """Start the air mouse processing loop."""
        if self.state == AirMouseState.RUNNING:
            logger.warning("Already running")
            return True

        # Initialize if not already initialized (state is STOPPED on first start)
        if self.state == AirMouseState.STOPPED:
            if not self.initialize():
                return False
        elif self.state != AirMouseState.PAUSED:
            if not self.initialize():
                return False

        self._running = True
        self._stop_event.clear()
        self.state = AirMouseState.RUNNING
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Air Mouse started")
        self._set_status("started", {})
        return True

    def stop(self):
        """Stop the air mouse."""
        logger.info("Stopping Air Mouse...")
        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._cleanup()
        self.state = AirMouseState.STOPPED
        logger.info("Air Mouse stopped")
        self._set_status("stopped", {})

    def pause(self):
        """Pause tracking (keep components alive)."""
        if self.state == AirMouseState.RUNNING:
            self.state = AirMouseState.PAUSED
            if self.gesture_recognizer:
                self.gesture_recognizer.set_tracking_paused(True)
            if self.cursor_controller:
                self.cursor_controller.set_active(False)
            if self.tracking_processor:
                self.tracking_processor.reset()
            logger.info("Air Mouse paused")
            self._set_status("paused", {})

    def resume(self):
        """Resume tracking."""
        if self.state == AirMouseState.PAUSED:
            self.state = AirMouseState.RUNNING
            if self.gesture_recognizer:
                self.gesture_recognizer.set_tracking_paused(False)
            if self.cursor_controller:
                self.cursor_controller.set_active(True)
            logger.info("Air Mouse resumed")
            self._set_status("resumed", {})

    def _run_loop(self):
        """Main processing loop."""
        frame_interval = 1.0 / self.config.target_fps
        last_frame_time = 0.0

        while self._running and not self._stop_event.is_set():
            loop_start = time.time()

            # Performance monitor frame start
            self.performance_monitor.update_frame_start()

            # Read frame
            ret, frame = self.camera.read_frame()
            if not ret or frame is None:
                self.stats.frames_dropped += 1
                self.performance_monitor.update_frame_end()
                time.sleep(0.001)
                continue

            # Process frame
            self._process_frame(frame)

            # Update stats
            self._update_stats(loop_start)

            # Performance monitor frame end
            self.performance_monitor.update_frame_end()

            # Draw debug overlay if enabled
            if self._debug_overlay_enabled and self._last_frame is not None:
                self._last_frame = self.performance_monitor.draw_overlay(self._last_frame)

            # Frame rate limiting
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_frame(self, frame):
        """Process a single frame through the pipeline."""
        frame_start = time.time()
        self._last_frame = frame

        # Hand detection
        hand_start = time.time()
        hands = self.hand_tracker.process(frame)
        hand_detection_time = (time.time() - hand_start) * 1000
        self.stats.hand_detection_time_ms = hand_detection_time
        self.performance_monitor.update_hand_detection(hand_detection_time)

        # Face detection (for head-relative mode)
        faces = []
        face_detection_time = 0.0
        if self.face_tracker:
            face_start = time.time()
            faces = self.face_tracker.process(frame)
            face_detection_time = (time.time() - face_start) * 1000
            self.stats.gesture_time_ms = face_detection_time  # Reuse this stat for face detection
        self.performance_monitor.update_face_detection(face_detection_time)

        # Process hands through tracking processor
        tracking_start = time.time()
        tracking_result = self.tracking_processor.process(hands, faces) if self.tracking_processor else None
        tracking_time = (time.time() - tracking_start) * 1000
        self.stats.cursor_time_ms = tracking_time
        self.performance_monitor.update_tracking(tracking_time)

        if not tracking_result or tracking_result.tracking_state == TrackingState.NO_HAND:
            # No valid tracked hand - release all buttons
            if self.virtual_mouse:
                self.virtual_mouse.release_all()
            if self.cursor_controller:
                self.cursor_controller.reset()
            if self.gesture_recognizer:
                self.gesture_recognizer.reset()
            if self.tracking_processor:
                self.tracking_processor.reset()
            # Still emit frame to GUI for preview
            if self.on_frame_processed:
                self.on_frame_processed(frame, [])
            return

        # Get tracked hands for gesture recognition
        tracked_hands = tracking_result.tracked_hands if tracking_result.tracked_hands else []
        primary_hand = tracking_result.primary_hand

        # Notify hand detected (use primary hand)
        if self.on_hand_detected and primary_hand:
            self.on_hand_detected(primary_hand)

        # Handle PRECISION_MODE: switch to precision sensitivity when two hands tracked
        if tracking_result.tracking_state == TrackingState.PRECISION_MODE:
            self.cursor_controller.set_sensitivity_mode(SensitivityMode.PRECISION)
        elif self.cursor_controller.get_sensitivity_mode() == SensitivityMode.PRECISION:
            # Revert to normal when not in precision mode
            self.cursor_controller.set_sensitivity_mode(SensitivityMode.NORMAL)

        # Get cursor movement from tracking processor (normalized from virtual plane)
        norm_movement = self.tracking_processor.get_cursor_movement()

        # Convert normalized movement to pixel movement via cursor controller
        rel_movement = None
        if norm_movement is not None:
            rel_movement = self.cursor_controller.get_relative_movement_from_plane(norm_movement[0], norm_movement[1])

        # Gesture recognition using tracked hands
        gesture_start = time.time()
        events = self.gesture_recognizer.process(tracked_hands)
        gesture_time = (time.time() - gesture_start) * 1000
        self.stats.gesture_time_ms = gesture_time
        self.performance_monitor.update_gesture(gesture_time)

        # Process gestures and move mouse
        mouse_start = time.time()
        self._handle_gestures(events, rel_movement, tracked_hands)
        mouse_time = (time.time() - mouse_start) * 1000
        self.stats.mouse_time_ms = mouse_time
        self.performance_monitor.update_mouse(mouse_time)

        # Safety check
        self._check_safety()

        # Update performance monitor with system stats (every 10 frames)
        if self.stats.frames_processed % 10 == 0:
            self.performance_monitor.update_system_stats()

        # Update camera stats
        if self.camera:
            stats = self.camera.get_brightness_stats()
            self.performance_monitor.update_camera_stats(
                stats.get('brightness', 0),
                stats.get('exposure', -1),
                stats.get('gain', -1)
            )

        # Update tracking state
        active_gestures = [e.gesture_type.value for e in events] if events else []
        hand_state = tracking_result.tracking_state.value if hasattr(tracking_result, 'tracking_state') else "TRACKING_ONE_HAND"

        self.performance_monitor.update_tracking_state(
            hand_detected=True,
            face_detected=len(faces) > 0,
            hand_state=hand_state,
            tracking_mode="3D_HEAD_RELATIVE" if self.config.tracking.use_head_relative else "2D",
            active_gestures=active_gestures,
            cursor_pos=self.virtual_mouse.get_position() if self.virtual_mouse else (0, 0),
            cursor_velocity=rel_movement if rel_movement else (0, 0)
        )

    def _handle_gestures(self, events: List[GestureEvent],
                         rel_movement: Optional[tuple], hands: List[TrackedHand]):
        """Handle gesture events and move mouse."""
        if not self.virtual_mouse:
            return

        # Move cursor if we have relative movement
        if rel_movement and (rel_movement[0] != 0 or rel_movement[1] != 0):
            self.virtual_mouse.move(rel_movement[0], rel_movement[1])

        # Notify GUI with processed frame and hands
        if self.on_frame_processed:
            self.on_frame_processed(self._last_frame, hands)

        # Process gesture events
        for event in events:
            if self.on_gesture:
                self.on_gesture(event)

            if event.gesture_type == GestureType.LEFT_CLICK:
                self.virtual_mouse.left_click()
                logger.debug("Left click")

            elif event.gesture_type == GestureType.RIGHT_CLICK:
                self.virtual_mouse.right_click()
                logger.debug("Right click")

            elif event.gesture_type == GestureType.MIDDLE_CLICK:
                self.virtual_mouse.middle_click()
                logger.debug("Middle click")

            elif event.gesture_type == GestureType.DRAG_START:
                self.virtual_mouse.button_down(0x110)  # BTN_LEFT
                logger.debug("Drag start")

            elif event.gesture_type == GestureType.DRAG_END:
                self.virtual_mouse.button_up(0x110)  # BTN_LEFT
                logger.debug("Drag end")

            elif event.gesture_type == GestureType.SCROLL_UP:
                amount = event.data.get("amount", 3)
                self.virtual_mouse.scroll(amount)
                logger.debug(f"Scroll up: {amount}")

            elif event.gesture_type == GestureType.SCROLL_DOWN:
                amount = event.data.get("amount", 3)
                self.virtual_mouse.scroll(-amount)
                logger.debug(f"Scroll down: {amount}")

            elif event.gesture_type == GestureType.PAUSE_TRACKING:
                self.pause()
                logger.info("Tracking paused (fist)")

            elif event.gesture_type == GestureType.RESUME_TRACKING:
                self.resume()
                logger.info("Tracking resumed")

    def _check_safety(self):
        """Check safety conditions (corner escape)."""
        if not self.config.emergency_stop_corner or not self.virtual_mouse:
            return

        # This would need actual cursor position - for now we check
        # if virtual mouse is at screen corner
        # In practice, we'd need to track actual cursor position
        pass

    def _update_stats(self, loop_start: float):
        """Update performance statistics."""
        current_time = time.time()
        frame_time = current_time - loop_start

        self._frame_times.append(frame_time)
        if len(self._frame_times) > 60:
            self._frame_times.pop(0)

        self.stats.frame_time_ms = frame_time * 1000
        self.stats.frames_processed += 1

        if len(self._frame_times) > 1:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            self.stats.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        self.stats.last_update = current_time

        if self.on_stats_update:
            self.on_stats_update(self.stats)

    def _set_status(self, status: str, data: dict):
        """Update status via callback."""
        if self.status_callback:
            self.status_callback(status, data)

    def _set_error(self, message: str):
        """Set error state."""
        self.state = AirMouseState.ERROR
        logger.error(message)
        if self.on_error:
            self.on_error(message)
        self._set_status("error", {"message": message})

    def _cleanup(self):
        """Clean up all components."""
        if self.virtual_mouse:
            self.virtual_mouse.destroy()
            self.virtual_mouse = None

        if self.hand_tracker:
            self.hand_tracker.close()
            self.hand_tracker = None

        if self.face_tracker:
            self.face_tracker.close()
            self.face_tracker = None

        if self.tracking_processor:
            self.tracking_processor.reset()
            self.tracking_processor = None

        if self.camera:
            self.camera.close_camera()

        self.cursor_controller = None
        self.gesture_recognizer = None

    def get_stats(self) -> PerformanceStats:
        return self.stats

    def is_running(self) -> bool:
        return self.state == AirMouseState.RUNNING

    def get_state(self) -> AirMouseState:
        return self.state

    def toggle_debug_overlay(self) -> bool:
        """Toggle debug performance overlay. Returns new state."""
        self._debug_overlay_enabled = self.performance_monitor.toggle()
        return self._debug_overlay_enabled

    def is_debug_overlay_enabled(self) -> bool:
        """Check if debug overlay is enabled."""
        return self.performance_monitor.is_enabled()


def create_air_mouse(config: Optional[AirMouseConfig] = None,
                     status_callback: Optional[Callable[[str, dict], None]] = None) -> AirMouseController:
    """Factory function to create and initialize Air Mouse."""
    controller = AirMouseController(config, status_callback)
    if controller.initialize():
        return controller
    return None


if __name__ == "__main__":
    # Test the controller initialization
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    print("Testing AirMouseController initialization...")

    def status_cb(status, data):
        print(f"Status: {status}, Data: {data}")

    def error_cb(msg):
        print(f"Error: {msg}")

    controller = AirMouseController(status_callback=status_cb)
    controller.on_error = error_cb

    if controller.initialize():
        print("Initialization successful!")
        print(f"Screen size: {controller.config.cursor.screen_width}x{controller.config.cursor.screen_height}")
        print(f"Camera: {controller.camera.get_resolution()}")

        # Test start/stop
        print("Starting...")
        controller.start()
        time.sleep(2)
        print("Stopping...")
        controller.stop()
        print("Done!")
    else:
        print("Initialization failed")
        sys.exit(1)