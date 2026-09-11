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
- Frame synchronization and coordination (Part 9)
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from enum import Enum
import queue
import numpy as np
from collections import deque

from ..camera.manager import CameraManager, CameraSettings, CameraInfo
from ..vision.hand_tracker import HandTracker, HandTrackerSettings, Hand
from ..vision.face_tracker import FaceTracker, FaceTrackerSettings, Face
from ..vision.gestures import (
    GestureRecognizer, GestureConfig, GestureEvent, GestureType, TrackingState
)
from ..vision.tracking_processor import TrackingProcessor, TrackingConfig, TrackedHand
from ..input import LinuxInputManager, InputBackend, UInputDeviceConfig
from .cursor import CursorController, CursorConfig, SmoothingAlgorithm, get_screen_size, SensitivityMode
from ..debug.performance_monitor import PerformanceMonitor
from ..brightness import AutoBrightnessController, BrightnessConfig, BrightnessState

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Frame data with timestamps for pipeline coordination."""
    frame_id: int
    timestamp: float          # Frame capture timestamp
    camera_timestamp: float   # Camera driver timestamp
    frame: Optional[np.ndarray] = None
    hands: List[Hand] = field(default_factory=list)
    faces: List[Face] = field(default_factory=list)
    tracked_hands: List[TrackedHand] = field(default_factory=list)
    primary_hand: Optional[TrackedHand] = None
    tracking_state: TrackingState = TrackingState.NO_HAND
    cursor_position: Optional[tuple] = None
    gesture_events: List[GestureEvent] = field(default_factory=list)
    mouse_movement: Optional[tuple] = None

    # Pipeline stage timestamps (ms)
    hand_detection_time: float = 0.0
    face_detection_time: float = 0.0
    tracking_time: float = 0.0
    gesture_time: float = 0.0
    cursor_time: float = 0.0
    mouse_time: float = 0.0
    total_pipeline_time: float = 0.0


class FrameCoordination:
    """
    Frame synchronization and coordination for the vision pipeline.

    Features:
    - Frame timestamps and latency tracking
    - Drop-frame handling and recovery
    - Thread-safe frame queues with backpressure
    - Health monitoring across pipeline stages
    """

    def __init__(self, max_queue_size: int = 4, target_fps: int = 60):
        self.max_queue_size = max_queue_size
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        # Frame queues with backpressure
        self._camera_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._processing_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)

        # Frame ID counter
        self._frame_counter = 0
        self._frame_counter_lock = threading.Lock()

        # Latency tracking
        self._latency_history: deque = deque(maxlen=100)
        self._stage_latencies: Dict[str, deque] = {
            'camera': deque(maxlen=100),
            'hand_detection': deque(maxlen=100),
            'face_detection': deque(maxlen=100),
            'tracking': deque(maxlen=100),
            'gesture': deque(maxlen=100),
            'cursor': deque(maxlen=100),
            'mouse': deque(maxlen=100),
        }

        # Drop frame tracking
        self._dropped_frames = 0
        self._processed_frames = 0
        self._last_frame_id = -1

        # Health monitoring
        self._stage_health: Dict[str, float] = {
            'camera': 1.0,
            'hand_detection': 1.0,
            'face_detection': 1.0,
            'tracking': 1.0,
            'gesture': 1.0,
            'cursor': 1.0,
            'mouse': 1.0,
        }
        self._last_frame_time = time.time()

    def generate_frame_id(self) -> int:
        """Generate unique frame ID."""
        with self._frame_counter_lock:
            self._frame_counter += 1
            return self._frame_counter

    def enqueue_frame(self, frame: np.ndarray, camera_timestamp: float) -> bool:
        """
        Enqueue a frame from camera with backpressure handling.

        Returns:
            True if enqueued, False if dropped (queue full)
        """
        try:
            frame_id = self.generate_frame_id()
            frame_data = FrameData(
                frame_id=frame_id,
                timestamp=time.time(),
                camera_timestamp=camera_timestamp,
                frame=frame
            )
            self._camera_queue.put_nowait(frame_data)
            return True
        except queue.Full:
            self._dropped_frames += 1
            logger.warning(f"Camera queue full, dropping frame. Total dropped: {self._dropped_frames}")
            return False

    def get_frame_for_processing(self, timeout: float = 0.1) -> Optional[FrameData]:
        """Get next frame for processing."""
        try:
            frame_data = self._camera_queue.get(timeout=timeout)
            return frame_data
        except queue.Empty:
            return None

    def enqueue_processed_frame(self, frame_data: FrameData) -> bool:
        """Enqueue processed frame for output."""
        try:
            self._processing_queue.put_nowait(frame_data)
            return True
        except queue.Full:
            logger.warning("Processing queue full, dropping processed frame")
            return False

    def get_processed_frame(self, timeout: float = 0.01) -> Optional[FrameData]:
        """Get processed frame for output."""
        try:
            return self._processing_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def record_stage_latency(self, frame_data: FrameData, stage: str, latency_ms: float):
        """Record latency for a pipeline stage."""
        if stage in self._stage_latencies:
            self._stage_latencies[stage].append(latency_ms)
            # Update health based on latency (healthy if under 2x frame interval)
            target_ms = self.frame_interval * 1000 * 2
            health = min(1.0, target_ms / max(latency_ms, 1.0))
            self._stage_health[stage] = health * 0.9 + self._stage_health[stage] * 0.1

    def record_total_latency(self, frame_data: FrameData):
        """Record total pipeline latency."""
        total_latency = (time.time() - frame_data.timestamp) * 1000
        self._latency_history.append(total_latency)

        # Track dropped frames based on frame ID sequence
        if frame_data.frame_id != self._last_frame_id + 1 and self._last_frame_id != -1:
            dropped = frame_data.frame_id - self._last_frame_id - 1
            self._dropped_frames += dropped
            logger.warning(f"Frame gap detected: expected {self._last_frame_id + 1}, got {frame_data.frame_id}, dropped {dropped}")
        self._last_frame_id = frame_data.frame_id
        self._processed_frames += 1

    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics."""
        def stats(values: deque) -> Dict[str, float]:
            if not values:
                return {'mean': 0, 'max': 0, 'p95': 0}
            arr = np.array(list(values))
            return {
                'mean': float(np.mean(arr)),
                'max': float(np.max(arr)),
                'p95': float(np.percentile(arr, 95)) if len(arr) > 1 else float(arr[0])
            }

        return {
            'total': stats(self._latency_history),
            'stages': {k: stats(v) for k, v in self._stage_latencies.items()},
            'health': self._stage_health.copy(),
            'dropped_frames': self._dropped_frames,
            'processed_frames': self._processed_frames,
            'drop_rate': self._dropped_frames / max(1, self._processed_frames + self._dropped_frames)
        }

    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes."""
        return {
            'camera_queue': self._camera_queue.qsize(),
            'processing_queue': self._processing_queue.qsize(),
        }

    def is_healthy(self) -> bool:
        """Check if pipeline is healthy."""
        return all(h > 0.3 for h in self._stage_health.values())

    def reset(self):
        """Reset coordination state."""
        self._frame_counter = 0
        self._latency_history.clear()
        for d in self._stage_latencies.values():
            d.clear()
        self._dropped_frames = 0
        self._processed_frames = 0
        self._last_frame_id = -1
        self._stage_health = {k: 1.0 for k in self._stage_health}


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

    # Auto-brightness settings
    brightness: BrightnessConfig = field(default_factory=BrightnessConfig)

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
    # Pipeline coordination stats (Part 9)
    pipeline_latency_ms: float = 0.0
    stage_latencies: Dict[str, float] = field(default_factory=dict)
    drop_rate: float = 0.0


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
        self.brightness_controller: Optional[AutoBrightnessController] = None
        self.input_manager = None

        # State
        self.state = AirMouseState.STOPPED
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Frame coordination (Part 9)
        self._frame_coord = FrameCoordination(
            max_queue_size=4,
            target_fps=self.config.target_fps
        )

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
        self.on_brightness_update: Optional[Callable[[BrightnessState], None]] = None

        # Performance monitor
        self.performance_monitor = PerformanceMonitor()
        self._debug_overlay_enabled = False

        # Keyboard hotkeys
        self._hotkey_listener = None
        self._hotkey_manager = None
        self._hotkeys_active = True

        # Safety manager
        self._safety_manager = None

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
            if self.config.camera.device_index < 0 and not self.config.camera.device_path:
                self.config.camera.device_index = cameras[0].index
                self.config.camera.device_path = cameras[0].device_path
                logger.info(f"Auto-selected camera {cameras[0].device_path} (index {cameras[0].index})")

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

            # Initialize auto-brightness controller
            self.brightness_controller = AutoBrightnessController(
                self.config.brightness,
                on_state_change=self._on_brightness_state_change
            )

            # Initialize Linux input manager (supports Wayland, X11, uinput, ydotool)
            self.input_manager = LinuxInputManager()
            if not self.input_manager.initialize():
                self._set_error("Failed to initialize input backend")
                return False

            logger.info("All components initialized successfully")
            self._set_status("initialized", {
                "screen": (screen_width, screen_height),
                "input_backend": self.input_manager.get_backend_type().value,
                "desktop_env": self.input_manager.get_desktop_environment().value
            })
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

        # Start auto-brightness
        if self.brightness_controller:
            self.brightness_controller.start()

        # Setup hotkeys
        self._setup_hotkeys()

        # Setup safety manager
        self._setup_safety()

        self._running = True
        self._stop_event.clear()
        self.state = AirMouseState.RUNNING
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Air Mouse started")
        self._set_status("started", {})
        return True

    def _setup_hotkeys(self):
        """Setup global keyboard hotkeys using GlobalHotkeyManager."""
        try:
            from ..ui.hotkeys import GlobalHotkeyManager, Hotkey, HotkeyBackend, KeyModifier, KeyCode, create_emergency_hotkey_manager

            # Create emergency hotkey manager with default hotkeys (Super+Alt+A to disable)
            self._hotkey_manager = create_emergency_hotkey_manager()

            # Add custom hotkey for debug overlay (Ctrl+Shift+G)
            debug_hotkey = Hotkey(
                modifiers=(KeyModifier.CONTROL, KeyModifier.SHIFT),
                key=KeyCode.KEY_G,
                callback=self.toggle_debug_overlay,
                description="Toggle debug overlay"
            )
            self._hotkey_manager.register_hotkey(debug_hotkey)

            # Add pause/resume hotkey (Super+Alt+P)
            pause_hotkey = Hotkey(
                modifiers=(KeyModifier.SUPER, KeyModifier.ALT),
                key=KeyCode.KEY_P,
                callback=self._toggle_pause_resume,
                description="Pause/Resume tracking"
            )
            self._hotkey_manager.register_hotkey(pause_hotkey)

            # Start the hotkey manager
            if self._hotkey_manager.start():
                logger.info(f"Hotkeys registered: {self._hotkey_manager.get_registered_hotkeys()}")
            else:
                logger.warning("Failed to start hotkey manager")
                self._hotkey_manager = None

        except ImportError:
            logger.warning("Hotkey dependencies not available - hotkeys disabled")
            self._hotkey_manager = None
        except Exception as e:
            logger.warning(f"Failed to setup hotkeys: {e}")
            self._hotkey_manager = None

    def _toggle_pause_resume(self):
        """Toggle pause/resume state."""
        if self.state == AirMouseState.RUNNING:
            self.pause()
            logger.info("Tracking paused via hotkey")
        elif self.state == AirMouseState.PAUSED:
            self.resume()
            logger.info("Tracking resumed via hotkey")

    def _setup_safety(self):
        """Setup safety manager with emergency disable, corner escape, etc."""
        try:
            from ..ui.safety import SafetyManager, SafetyConfig, SafetyTrigger, SafetyLevel, DEFAULT_SAFETY_CONFIG

            # Create safety config - we can use defaults and customize
            safety_config = DEFAULT_SAFETY_CONFIG
            safety_config.emergency_hotkey = "Super+Alt+A"  # Handled by hotkey manager
            safety_config.corner_escape_enabled = True
            safety_config.velocity_limit_enabled = True
            safety_config.focus_loss_pause = True
            safety_config.inactivity_timeout = 300.0  # 5 minutes

            self._safety_manager = SafetyManager(safety_config)

            # Register safety callbacks
            def on_safety_triggered(trigger, level, details):
                logger.warning(f"Safety triggered: {trigger.value} at level {level.value}: {details}")
                if level in (SafetyLevel.DISABLE, SafetyLevel.EMERGENCY):
                    self.stop()
                elif level == SafetyLevel.PAUSE:
                    self.pause()

            self._safety_manager.on_safety_event = on_safety_triggered

            # Start the safety manager
            if self._safety_manager.start():
                logger.info("Safety manager started")
            else:
                logger.warning("Failed to start safety manager")
                self._safety_manager = None

        except ImportError:
            logger.warning("Safety dependencies not available - safety disabled")
            self._safety_manager = None
        except Exception as e:
            logger.warning(f"Failed to setup safety: {e}")
            self._safety_manager = None

    def stop(self):
        """Stop the air mouse."""
        logger.info("Stopping Air Mouse...")
        self._running = False
        self._stop_event.set()

        # Stop hotkey manager
        if self._hotkey_manager:
            self._hotkey_manager.stop()
            self._hotkey_manager = None

        # Stop hotkey listener (legacy)
        if self._hotkey_listener:
            self._hotkey_listener.stop()
            self._hotkey_listener = None

        # Stop auto-brightness
        if self.brightness_controller:
            self.brightness_controller.stop()

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
            # Pause auto-brightness
            if self.brightness_controller and self.brightness_controller.is_active:
                self.brightness_controller.stop()
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
            # Resume auto-brightness
            if self.brightness_controller:
                self.brightness_controller.start()
            logger.info("Air Mouse resumed")
            self._set_status("resumed", {})

    def _run_loop(self):
        """Main processing loop with frame coordination."""
        frame_interval = 1.0 / self.config.target_fps

        while self._running and not self._stop_event.is_set():
            loop_start = time.time()

            # Performance monitor frame start
            self.performance_monitor.update_frame_start()

            # Read frame from camera
            ret, frame = self.camera.read_frame()
            if not ret or frame is None:
                self.stats.frames_dropped += 1
                self.performance_monitor.update_frame_end()
                time.sleep(0.001)
                continue

            # Get camera timestamp (approximate)
            camera_timestamp = time.time()

            # Enqueue frame with coordination
            self._frame_coord.enqueue_frame(frame, camera_timestamp)

            # Get frame for processing
            frame_data = self._frame_coord.get_frame_for_processing(timeout=0.01)
            if frame_data is None:
                self.performance_monitor.update_frame_end()
                continue

            # Process frame through pipeline
            self._process_frame_with_coordination(frame_data)

            # Enqueue processed frame for output
            self._frame_coord.enqueue_processed_frame(frame_data)

            # Get processed frame for output/callbacks
            output_frame = self._frame_coord.get_processed_frame(timeout=0.01)
            if output_frame:
                self._handle_frame_output(output_frame)

            # Record total latency
            self._frame_coord.record_total_latency(frame_data)

            # Update stats
            self._update_stats_with_coordination(loop_start, frame_data)

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

    def _process_frame_with_coordination(self, frame_data: FrameData):
        """Process a single frame through the pipeline with coordination."""
        self._last_frame = frame_data.frame

        # Hand detection
        hand_start = time.time()
        hands = self.hand_tracker.process(frame_data.frame)
        hand_detection_time = (time.time() - hand_start) * 1000
        self.stats.hand_detection_time_ms = hand_detection_time
        self._frame_coord.record_stage_latency(frame_data, 'hand_detection', hand_detection_time)
        self.performance_monitor.update_hand_detection(hand_detection_time)

        frame_data.hands = hands

        # Face detection (for head-relative mode)
        faces = []
        face_detection_time = 0.0
        if self.face_tracker:
            face_start = time.time()
            faces = self.face_tracker.process(frame_data.frame)
            face_detection_time = (time.time() - face_start) * 1000
            self.stats.gesture_time_ms = face_detection_time  # Reuse this stat for face detection
            self._frame_coord.record_stage_latency(frame_data, 'face_detection', face_detection_time)
        self.performance_monitor.update_face_detection(face_detection_time)

        frame_data.faces = faces

        # Process hands through tracking processor
        tracking_start = time.time()
        tracking_result = self.tracking_processor.process(hands, faces) if self.tracking_processor else None
        tracking_time = (time.time() - tracking_start) * 1000
        self.stats.cursor_time_ms = tracking_time
        self._frame_coord.record_stage_latency(frame_data, 'tracking', tracking_time)
        self.performance_monitor.update_tracking(tracking_time)

        if not tracking_result or tracking_result.tracking_state == TrackingState.NO_HAND:
            # No valid tracked hand - release all buttons
            if self.input_manager:
                self.input_manager.release_all()
            if self.cursor_controller:
                self.cursor_controller.reset()
            if self.gesture_recognizer:
                self.gesture_recognizer.reset()
            if self.tracking_processor:
                self.tracking_processor.reset()
            frame_data.tracking_state = TrackingState.NO_HAND
            return

        # Get tracked hands for gesture recognition
        tracked_hands = tracking_result.tracked_hands if tracking_result.tracked_hands else []
        primary_hand = tracking_result.primary_hand

        frame_data.tracked_hands = tracked_hands
        frame_data.primary_hand = primary_hand
        frame_data.tracking_state = tracking_result.tracking_state

        # Notify hand detected (use primary hand)
        if self.on_hand_detected and primary_hand:
            self.on_hand_detected(primary_hand)

        # Handle PRECISION_MODE: switch to precision sensitivity when two hands tracked
        if tracking_result.tracking_state == TrackingState.PRECISION_MODE:
            self.cursor_controller.set_sensitivity_mode(SensitivityMode.PRECISION)
        elif self.cursor_controller.get_sensitivity_mode() == SensitivityMode.PRECISION:
            # Revert to normal when not in precision mode
            self.cursor_controller.set_sensitivity_mode(SensitivityMode.NORMAL)

        # Get cursor position from tracking processor (normalized from virtual plane)
        norm_position = self.tracking_processor.get_cursor_position()

        # Convert normalized position to pixel movement via cursor controller
        rel_movement = None
        if norm_position is not None:
            rel_movement = self.cursor_controller.get_relative_movement_from_plane(norm_position[0], norm_position[1])

        frame_data.cursor_position = norm_position
        frame_data.mouse_movement = rel_movement

        # Gesture recognition using tracked hands
        gesture_start = time.time()
        events = self.gesture_recognizer.process(tracked_hands)
        gesture_time = (time.time() - gesture_start) * 1000
        self.stats.gesture_time_ms = gesture_time
        self._frame_coord.record_stage_latency(frame_data, 'gesture', gesture_time)
        self.performance_monitor.update_gesture(gesture_time)

        frame_data.gesture_events = events

        # Process gestures and move mouse
        mouse_start = time.time()
        self._handle_gestures(events, rel_movement, tracked_hands)
        mouse_time = (time.time() - mouse_start) * 1000
        self.stats.mouse_time_ms = mouse_time
        self._frame_coord.record_stage_latency(frame_data, 'mouse', mouse_time)
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
            cursor_pos=(0, 0),  # LinuxInputManager doesn't expose get_position
            cursor_velocity=rel_movement if rel_movement else (0, 0)
        )

    def _handle_frame_output(self, frame_data: FrameData):
        """Handle output for processed frame (GUI callbacks)."""
        # Emit frame to GUI for preview
        if self.on_frame_processed:
            self.on_frame_processed(frame_data.frame, frame_data.hands)

    def _update_stats_with_coordination(self, loop_start: float, frame_data: FrameData):
        """Update performance statistics with coordination data."""
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

        # Add pipeline coordination stats
        latency_stats = self._frame_coord.get_latency_stats()
        self.stats.pipeline_latency_ms = latency_stats['total']['mean']
        self.stats.stage_latencies = {k: v['mean'] for k, v in latency_stats['stages'].items()}
        self.stats.drop_rate = latency_stats['drop_rate']
        self.stats.frames_dropped = latency_stats['dropped_frames']

        self.stats.last_update = current_time

        if self.on_stats_update:
            self.on_stats_update(self.stats)

    def _handle_gestures(self, events: List[GestureEvent],
                         rel_movement: Optional[tuple], hands: List[TrackedHand]):
        """Handle gesture events and move mouse."""
        if not self.input_manager:
            return

        # Move cursor if we have relative movement
        if rel_movement and (rel_movement[0] != 0 or rel_movement[1] != 0):
            self.input_manager.move(rel_movement[0], rel_movement[1])

        # Notify GUI with processed frame and hands
        if self.on_frame_processed:
            self.on_frame_processed(self._last_frame, hands)

        # Process gesture events
        for event in events:
            if self.on_gesture:
                self.on_gesture(event)

            if event.gesture_type == GestureType.LEFT_CLICK:
                self.input_manager.click(1)
                logger.debug("Left click")

            elif event.gesture_type == GestureType.RIGHT_CLICK:
                self.input_manager.click(3)
                logger.debug("Right click")

            elif event.gesture_type == GestureType.MIDDLE_CLICK:
                self.input_manager.click(2)
                logger.debug("Middle click")

            elif event.gesture_type == GestureType.DRAG_START:
                self.input_manager.button_down(1)
                logger.debug("Drag start")

            elif event.gesture_type == GestureType.DRAG_END:
                self.input_manager.button_up(1)
                logger.debug("Drag end")

            elif event.gesture_type == GestureType.SCROLL_UP:
                amount = event.data.get("amount", 3)
                self.input_manager.scroll(amount)
                logger.debug(f"Scroll up: {amount}")

            elif event.gesture_type == GestureType.SCROLL_DOWN:
                amount = event.data.get("amount", 3)
                self.input_manager.scroll(-amount)
                logger.debug(f"Scroll down: {amount}")

            elif event.gesture_type == GestureType.PINCH_END:
                # Release the mouse button for the original gesture
                original = event.data.get("original_gesture")
                if original == GestureType.LEFT_CLICK:
                    self.input_manager.button_up(1)
                    logger.debug("Left click release (pinch end)")
                elif original == GestureType.RIGHT_CLICK:
                    self.input_manager.button_up(3)
                    logger.debug("Right click release (pinch end)")
                elif original == GestureType.MIDDLE_CLICK:
                    self.input_manager.button_up(2)
                    logger.debug("Middle click release (pinch end)")

            elif event.gesture_type == GestureType.PAUSE_TRACKING:
                self.pause()
                logger.info("Tracking paused (fist)")

            elif event.gesture_type == GestureType.RESUME_TRACKING:
                self.resume()
                logger.info("Tracking resumed")

    def toggle_debug_overlay(self):
        """Toggle the performance debug overlay."""
        enabled = self.performance_monitor.toggle()
        logger.info(f"Debug overlay: {'enabled' if enabled else 'disabled'}")
        return enabled

    def _check_safety(self):
        """Check safety conditions using SafetyManager."""
        if self._safety_manager:
            # The safety manager handles its own monitoring
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

    def _on_brightness_state_change(self, state: BrightnessState):
        """Handle brightness state changes from controller."""
        # Forward to GUI callback if available
        if self.on_brightness_update:
            self.on_brightness_update(state)

    def _set_error(self, message: str):
        """Set error state."""
        self.state = AirMouseState.ERROR
        logger.error(message)
        if self.on_error:
            self.on_error(message)
        self._set_status("error", {"message": message})

    def _cleanup(self):
        """Clean up all components."""
        if self.input_manager:
            self.input_manager.cleanup()
            self.input_manager = None

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