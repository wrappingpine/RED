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
from ..vision.tracking_status import TrackingStatus, TrackingPhase, LostReason, ConfidenceState
from ..input import LinuxInputManager, InputBackend, UInputDeviceConfig
from .cursor import CursorController, CursorConfig, SmoothingAlgorithm, get_screen_size, SensitivityMode
from ..debug.performance_monitor import PerformanceMonitor
from ..brightness import AutoBrightnessController, BrightnessConfig, BrightnessState

from airmouse.utils.error_logging import get_error_logger

logger = logging.getLogger(__name__)

# Debug mode configuration
DEBUG_MODE = False
DEBUG_INTERVAL = 30  # Log every N frames
DEBUG_LOG_LEVEL = logging.DEBUG


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

    # Latency breakdown dict (populated by FrameCoordination)
    latency_data: Optional[Dict[str, float]] = None


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
        self._stopping = False
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

        # Tracking status (confidence + loss detection per spec §16-18)
        self._tracking_status = TrackingStatus()
        self.on_tracking_status_change: Optional[Callable[[dict], None]] = None

        # Rate-limited logging for tracking loss (prevents spam)
        self._last_tracking_lost_log: float = 0.0
        self._tracking_lost_log_interval: float = 3.0  # Log at most once per 3 seconds

        # §7: Action-cursor coordination.  When a click or drag fires, the
        # cursor is frozen so the hand's natural micro-twitchs don't drag
        # the selection off the target.  Reset on drag-end or after a few
        # frames of no movement.
        self._cursor_frozen: bool = False
        self._drag_active: bool = False
        self._consecutive_lost: int = 0
        self._frame_count: int = 0
        self._consecutive_projection_failures: int = 0
        self._max_projection_failures: int = 5

    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing Air Mouse...")
        
        health_results = {}
        all_ok = True

        try:
            # Detect cameras first
            cameras = self.camera.detect_cameras()
            if not cameras:
                logger.error("✗ No cameras detected")
                health_results['camera'] = 'FAIL: no devices found'
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
                logger.error("✗ Failed to open camera")
                health_results['camera'] = 'FAIL: cannot open device'
                self._set_error("Failed to open camera")
                return False

            # Get actual camera resolution
            actual_width, actual_height = self.camera.get_resolution()
            logger.info(f"✓ Camera: {actual_width}x{actual_height}")
            health_results['camera'] = f'OK: {actual_width}x{actual_height}'

            # Initialize hand tracker
            self.hand_tracker = HandTracker(self.config.hand_tracker)
            logger.info("✓ Hand tracking initialized")
            health_results['hand_tracker'] = 'OK'

            # Initialize face tracker (required for head-relative mode)
            if self.config.tracking.use_head_relative:
                self.face_tracker = FaceTracker(FaceTrackerSettings(
                    max_faces=1,
                    min_detection_confidence=self.config.tracking.min_face_confidence,
                    min_presence_confidence=0.0,
                    min_tracking_confidence=0.0,
                ))
                logger.info("✓ Face tracker initialized for head-relative mode")
                health_results['face_tracker'] = 'OK'

            # Initialize cursor controller
            screen_width, screen_height = get_screen_size()
            self.config.cursor.screen_width = screen_width
            self.config.cursor.screen_height = screen_height
            self.config.cursor.camera_width = actual_width
            self.config.cursor.camera_height = actual_height
            self.cursor_controller = CursorController(self.config.cursor)
            logger.info(f"✓ Cursor controller: {screen_width}x{screen_height} screen")
            health_results['cursor'] = f'OK: {screen_width}x{screen_height}'

            # Initialize gesture recognizer
            self.gesture_recognizer = GestureRecognizer(
                config=self.config.gestures,
                callback=self.on_gesture
            )
            logger.info("✓ Gesture recognizer initialized")
            health_results['gestures'] = 'OK'

            # Initialize tracking processor
            self.tracking_processor = TrackingProcessor(self.config.tracking)
            logger.info("✓ Tracking processor initialized")
            health_results['tracking'] = 'OK'

            # Initialize auto-brightness controller
            self.brightness_controller = AutoBrightnessController(
                self.config.brightness,
                on_state_change=self._on_brightness_state_change
            )
            logger.info("✓ Auto-brightness controller initialized")
            health_results['brightness'] = 'OK'

            # Initialize Linux input manager (supports Wayland, X11, uinput, ydotool)
            self.input_manager = LinuxInputManager()
            if not self.input_manager.initialize():
                logger.error("✗ Failed to initialize input backend")
                health_results['input'] = 'FAIL: backend unavailable'
                self._set_error("Failed to initialize input backend")
                return False
            
            backend_type = self.input_manager.get_backend_type().value
            desktop_env = self.input_manager.get_desktop_environment().value
            logger.info(f"✓ Input backend: {backend_type} (desktop: {desktop_env})")
            health_results['input'] = f'OK: {backend_type} ({desktop_env})'

            logger.info("All components initialized successfully")

            # Print calibration diagnostic at startup
            if self.tracking_processor and hasattr(self.tracking_processor, 'virtual_plane'):
                try:
                    diag = self.tracking_processor.virtual_plane.calibration_diagnostic()
                    logger.info(f"=== VIRTUAL PLANE CALIBRATION ===")
                    logger.info(f"  Plane: distance={diag['plane']['distance']}m, "
                                f"size={diag['plane']['width']}x{diag['plane']['height']} "
                                f"(aspect={diag['plane']['aspect_ratio']:.2f})")
                    logger.info(f"  Coordinate convention: {diag['coordinate_convention']['u_axis']}, "
                                f"{diag['coordinate_convention']['v_axis']}")
                    logger.info(f"  Expected mapping: center=({diag['expected_mapping']['center']['u']}, "
                                f"{diag['expected_mapping']['center']['v']}), "
                                f"left=({diag['expected_mapping']['left_edge']['u']}, "
                                f"{diag['expected_mapping']['left_edge']['v']}), "
                                f"right=({diag['expected_mapping']['right_edge']['u']}, "
                                f"{diag['expected_mapping']['right_edge']['v']}), "
                                f"top=({diag['expected_mapping']['top_edge']['u']}, "
                                f"{diag['expected_mapping']['top_edge']['v']}), "
                                f"bottom=({diag['expected_mapping']['bottom_edge']['u']}, "
                                f"{diag['expected_mapping']['bottom_edge']['v']})")
                    if diag.get('basis_valid'):
                        logger.info(f"  Basis valid: {diag['basis_orthonormal']}")
                        if not diag['basis_orthonormal']:
                            logger.warning(f"  Basis dots: {diag.get('basis_dots', {})}")
                    else:
                        logger.warning("  Basis not valid (no head coordinates)")
                    logger.info(f"=== END CALIBRATION ===")
                except Exception as e:
                    logger.debug(f"Calibration diagnostic failed: {e}")

            self._set_status("initialized", {
                "screen": (screen_width, screen_height),
                "input_backend": backend_type,
                "desktop_env": desktop_env,
                "health": health_results
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

        # Setup tracking status callbacks
        self._tracking_status.set_callbacks(
            on_lost=lambda reason: self._on_tracking_lost(reason),
            on_recovered=lambda: self._on_tracking_recovered(),
            on_phase_change=lambda old, new: self._on_tracking_phase_change(old, new)
        )

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
            self._hotkey_manager = create_emergency_hotkey_manager(emergency_callback=self._on_emergency)

            # Add custom hotkey for debug overlay (Ctrl+Shift+G)
            debug_hotkey = Hotkey(
                id="debug_overlay",
                modifiers={KeyModifier.CTRL, KeyModifier.SHIFT},
                key=KeyCode.G,
                callback=self.toggle_debug_overlay,
                description="Toggle debug overlay"
            )
            self._hotkey_manager.register_hotkey(debug_hotkey)

            # Add pause/resume hotkey (Super+Alt+P)
            pause_hotkey = Hotkey(
                id="pause_resume",
                modifiers={KeyModifier.SUPER, KeyModifier.ALT},
                key=KeyCode.P,
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

    def _on_emergency(self):
        """Emergency disable callback - freeze input and stop tracking."""
        logger.critical("EMERGENCY DISABLE TRIGGERED!")
        if self._stopping:
            return
        if self._safety_manager:
            self._safety_manager.emergency_stop()
        # Defer stop to a fresh thread — running _cleanup() (which calls
        # MediaPipe face_tracker.close()) inline on the hotkey event loop
        # thread segfaults the MediaPipe dispatcher.
        import threading as _threading
        t = _threading.Thread(target=self.stop, daemon=True)
        t.start()

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
            safety_config.enable_corner_escape = True
            safety_config.corner_size = 50
            safety_config.corner_hold_time = 0.5
            safety_config.enable_velocity_limit = True
            safety_config.max_cursor_velocity = 5000
            # Focus-loss pause is DISABLED by default (see SafetyConfig).
            # On Wayland/COSMIC the focus monitor falls back to a
            # hand-activity heuristic that cannot distinguish a stationary
            # hand in the dead zone from a user who walked away, causing
            # constant pause/resume cycles.  Only enable on X11 where a
            # real focus query is available.
            safety_config.enable_focus_loss_pause = False
            safety_config.enable_inactivity_timeout = False
            safety_config.inactivity_timeout = 300.0  # 5 minutes

            self._safety_manager = SafetyManager(safety_config)

            # Track the last cursor position we actually moved the mouse to.
            # Start the cursor at the screen center rather than (0, 0).
            # (0, 0) is the top-left corner, so the corner-escape detector
            # would fire on the very first frame. Center is safe.
            def get_screen_size():
                return (self.config.cursor.screen_width or 1920,
                        self.config.cursor.screen_height or 1080)

            screen_w, screen_h = get_screen_size()
            self._cursor_position = (screen_w // 2, screen_h // 2)

            def get_cursor_pos():
                return self._cursor_position

            # Register safety callbacks (required before start())
            # NOTE: do NOT pass disable=self.stop here — _execute_safety_action
            # runs on the safety monitor thread and would deadlock by calling
            # stop() → _cleanup() → face_tracker.close() → MediaPipe dispatcher
            # while the monitor thread is still alive. The on_safety_triggered
            # event callback (registered below) handles stop() safely on the
            # main thread instead.
            self._safety_manager.set_callbacks(
                get_cursor_pos=get_cursor_pos,
                get_screen_size=get_screen_size,
                release_all=self._release_all_input,
                pause=self.pause,
                disable=lambda: None,
                show_notification=None,
            )

            def on_safety_triggered(event):
                logger.warning(f"Safety triggered: {event.trigger.value} at level {event.level.value}: {event.details}")
                if self._stopping:
                    return
                if event.level.value >= SafetyLevel.DISABLE.value:
                    # Defer stop to a fresh thread — running _cleanup()
                    # (which calls MediaPipe face_tracker.close()) inline on
                    # the safety monitor thread segfaults the MediaPipe
                    # dispatcher.
                    import threading as _threading
                    t = _threading.Thread(target=self.stop, daemon=True)
                    t.start()
                elif event.level.value == SafetyLevel.PAUSE.value:
                    self.pause()

            self._safety_manager.add_callback(on_safety_triggered)

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

    def _release_all_input(self):
        """Release all active mouse buttons and reset cursor state."""
        if self.input_manager:
            try:
                self.input_manager.release_all()
            except Exception as e:
                logger.error(f"Failed to release input: {e}")

    def stop(self):
        """Stop the air mouse."""
        logger.info("Stopping Air Mouse...")
        self._running = False
        self._stopping = True
        self._stop_event.set()

        # Stop safety manager FIRST to prevent re-entrant callbacks
        if self._safety_manager:
            try:
                self._safety_manager.disable(join=False)
            except Exception as e:
                logger.warning(f"Safety disable error during stop: {e}")
            self._safety_manager = None

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

    def _on_tracking_lost(self, reason: LostReason):
        """Callback when tracking is lost - freeze everything per spec §17."""
        # Freeze pointer - stop all movement
        if self.cursor_controller:
            self.cursor_controller.set_active(False)
        
        # Release all buttons
        if self.input_manager:
            self.input_manager.release_all()
        
        # Stop gesture recognition
        if self.gesture_recognizer:
            self.gesture_recognizer.reset()
        
        # Update status
        self._set_status("tracking_lost", {
            "reason": reason.name,
            "confidence": self._tracking_status.confidence.confidence_value
        })
        
        # Rate-limited logging (prevents spam when hand is genuinely absent)
        now = time.time()
        if now - self._last_tracking_lost_log >= self._tracking_lost_log_interval:
            logger.warning(f"Tracking lost - all input frozen: {reason.name}")
            self._last_tracking_lost_log = now

    def _on_tracking_recovered(self):
        """Callback when tracking is recovered - reset and resume per §18."""
        # Reset frozen state
        if self.cursor_controller:
            self.cursor_controller.set_active(True)
        
        # Release any held buttons
        if self.input_manager:
            self.input_manager.release_all()
        
        # Reset gesture recognizer
        if self.gesture_recognizer:
            self.gesture_recognizer.reset()
        
        # Update status
        self._set_status("tracking_recovered", {})
        
        logger.info("Tracking recovered - input resumed")

    def _on_tracking_phase_change(self, old: TrackingPhase, new: TrackingPhase):
        """Callback when tracking phase changes."""
        self._set_status("tracking_phase", {
            "old": old.name,
            "new": new.name
        })

    def calibrate(self) -> bool:
        """
        Run calibration sequence (§26).

        Returns True if calibration completed successfully.
        """
        if self.state == AirMouseState.RUNNING:
            logger.warning("Cannot calibrate while running")
            return False

        if not self.hand_tracker:
            logger.error("Hand tracker not initialized")
            return False

        from airmouse.vision.calibration import Calibrator, CalibrationConfig, CalibrationPhase, apply_calibration

        calibrator = Calibrator(CalibrationConfig())
        calibrator.set_callback(self._on_calibration_phase)

        # Process frames through calibrator
        calibrator.start()
        max_frames = 300
        frame_count = 0

        while calibrator.phase != CalibrationPhase.COMPLETE and frame_count < max_frames:
            if self._stop_event.is_set():
                calibrator.cancel()
                break

            hand = self._get_current_hand()
            calibrator.process(hand)
            frame_count += 1
            time.sleep(0.05)  # Simulate frame processing

        result = calibrator.result
        if result.completed:
            apply_calibration(result, self.config.cursor)
            logger.info("Calibration completed successfully")
            return True
        else:
            logger.warning("Calibration did not complete")
            return False

    def _on_calibration_phase(self, phase):
        """Callback for calibration phase changes."""
        logger.info(f"Calibration phase: {phase.name}")
        if self.status_callback:
            self.status_callback("calibration", {"phase": phase.name})

    def _get_current_hand(self) -> Optional[Hand]:
        """Get the most recent detected hand."""
        if self.hand_tracker and hasattr(self.hand_tracker, 'last_hand'):
            return self.hand_tracker.last_hand
        return None

    def _run_loop(self):
        """Main processing loop with frame coordination."""
        frame_interval = 1.0 / self.config.target_fps

        # Camera recovery state (spec §48: handle camera disappearance + recovery)
        self._camera_consecutive_failures = 0
        self._camera_max_failures_before_recovery = 15  # ~0.5s at 30fps
        self._camera_recovery_backoff = 1.0  # seconds between recovery attempts
        self._camera_last_recovery_attempt = 0.0

        while self._running and not self._stop_event.is_set():
            loop_start = time.time()

            # Performance monitor frame start
            self.performance_monitor.update_frame_start()

            # Read frame from camera
            ret, frame = self.camera.read_frame()
            if not ret or frame is None:
                self.stats.frames_dropped += 1
                self._camera_consecutive_failures += 1

                # Check if camera needs recovery (spec §48)
                if self._camera_consecutive_failures >= self._camera_max_failures_before_recovery:
                    now = time.time()
                    if now - self._camera_last_recovery_attempt >= self._camera_recovery_backoff:
                        logger.warning(
                            f"Camera read failed {self._camera_consecutive_failures} times, "
                            f"attempting recovery..."
                        )
                        if self._recover_camera():
                            self._camera_consecutive_failures = 0
                            self._set_status("camera_recovered", {})
                        else:
                            self._camera_last_recovery_attempt = now
                    # Brief sleep to avoid busy-looping during recovery backoff
                    time.sleep(0.05)
                else:
                    time.sleep(0.001)

                self.performance_monitor.update_frame_end()
                continue

            # Reset failure counter on successful read
            if self._camera_consecutive_failures > 0:
                logger.info(f"Camera recovered after {self._camera_consecutive_failures} failures")
                self._camera_consecutive_failures = 0

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

        # Debug dump at intervals (spec: full pipeline observability on demand)
        if DEBUG_MODE and self._frame_count % DEBUG_INTERVAL == 0:
            self._debug_dump_pipeline(frame_data)

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

        # Update tracking confidence FIRST — before any early-return branches.
        # This ensures the confidence state machine always advances, even
        # when projection fails.  Without this, confidence stays stale and
        # the state machine gets stuck (e.g., STABILIZING forever).
        if hands:
            max_conf = max((h.confidence for h in hands), default=0.0)
            self._tracking_status.confidence.update(max_conf, frame_count=self._frame_count)

        if not tracking_result or tracking_result.tracking_state == TrackingState.NO_HAND:
            # No valid tracked hand - release all buttons and freeze
            if self.input_manager:
                self.input_manager.release_all()

            # Distinguish "no hands at all" from "hands detected but
            # confidence too low to track".  Resetting the tracking
            # processor (One Euro Filter state, gesture state) on every
            # borderline frame causes a ping-pong where the hand is
            # created and destroyed every other frame.  Only reset when
            # there are genuinely no hands, or when confidence has been
            # below the drop threshold for several consecutive frames.
            if not hands:
                if self.cursor_controller:
                    self.cursor_controller.reset()
                if self.gesture_recognizer:
                    self.gesture_recognizer.reset()
                if self.tracking_processor:
                    self.tracking_processor.reset()
                self._tracking_status.record_hand_lost(LostReason.NO_HAND_DETECTED)
            else:
                # Hands exist but confidence is too low to track.
                # Hold the last valid cursor position; do NOT reset
                # filters or gesture state — they will re-acquire
                # automatically when confidence recovers.
                max_conf = max((h.confidence for h in hands), default=0.0)
                if max_conf < 0.15:
                    # Truly lost: reset after enough consecutive bad frames
                    self._consecutive_lost = getattr(self, '_consecutive_lost', 0) + 1
                    if self._consecutive_lost >= 3:
                        if self.cursor_controller:
                            self.cursor_controller.reset()
                        if self.gesture_recognizer:
                            self.gesture_recognizer.reset()
                        if self.tracking_processor:
                            self.tracking_processor.reset()
                        self._consecutive_lost = 0
                        self._tracking_status.record_hand_lost(
                            LostReason.CONFIDENCE_DROP,
                            f"confidence={max_conf:.2f} (sustained)")
                    else:
                        self._tracking_status.record_hand_lost(
                            LostReason.CONFIDENCE_DROP,
                            f"confidence={max_conf:.2f} (holding, frame {self._consecutive_lost})")
                else:
                    # Borderline (0.15-0.40): hold position, don't trigger loss
                    self._consecutive_lost = 0

            frame_data.tracking_state = TrackingState.NO_HAND
            return

        # LOST_TRACK means hands were detected but projection/validation failed.
        # This is a PROJECTION failure, NOT a tracking loss.
        # The hand is still visible — only the fingertip ray falls outside
        # the virtual control plane.  Do NOT convert every projection failure
        # into tracking_lost.
        if tracking_result.tracking_state == TrackingState.LOST_TRACK:
            # Release buttons for safety but keep cursor state so we don't jump
            if self.input_manager:
                self.input_manager.release_all()
            if self.gesture_recognizer:
                self.gesture_recognizer.reset()

            max_conf = max((h.confidence for h in hands), default=0.0)
            proj = tracking_result.projection

            # Grace period: tolerate brief projection failures without
            # declaring tracking lost.  Only after N consecutive bad frames
            # do we transition to LOST.
            self._consecutive_projection_failures = getattr(
                self, '_consecutive_projection_failures', 0) + 1
            max_failures = getattr(self, '_max_projection_failures', 5)

            if max_conf < 0.15:
                # Hand confidence genuinely low — this IS a tracking problem
                self._tracking_status.record_hand_lost(
                    LostReason.CONFIDENCE_DROP,
                    f"confidence={max_conf:.2f}")
            elif self._consecutive_projection_failures >= max_failures:
                # Sustained projection failure — hand may be outside plane
                # Use TEMPORARY_LOSS instead of TRACKING_JUMP to avoid
                # permanent freeze.  The state machine will recover when
                # confidence returns to acceptable levels.
                self._tracking_status.phase = TrackingPhase.TEMPORARY_LOSS
                logger.info(
                    f"TRACKING_TEMPORARY_LOSS: frame={self._frame_count} "
                    f"confidence={max_conf:.2f} "
                    f"consecutive_failures={self._consecutive_projection_failures}/{max_failures} "
                    f"action=hold_position")
            else:
                # Brief projection failure — hold position, do NOT freeze
                logger.debug(
                    f"TRACKING_PROJECTION_FAILURE: frame={self._frame_count} "
                    f"confidence={max_conf:.2f} "
                    f"consecutive_failures={self._consecutive_projection_failures}/{max_failures} "
                    f"action=hold_position")

            # CRITICAL: Always call update() so the state machine advances
            # (stabilization_frames counter, reacquisition timers, etc.)
            # This prevents the permanent STABILIZING ping-pong loop.
            self._tracking_status.update()

            frame_data.tracking_state = TrackingState.LOST_TRACK
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

        # Update tracking confidence with the primary hand's confidence
        if primary_hand:
            # Reset projection failure counter when tracking succeeds
            self._consecutive_projection_failures = 0

            self._tracking_status.record_hand_detected(
                primary_hand.confidence,
                evidence=f"tracking_state={tracking_result.tracking_state.value}"
            )
            # Track phase progression from STARTING to TRACKING
            # After hand is detected with sufficient confidence, transition to active tracking
            if self._tracking_status.phase == TrackingPhase.STARTING:
                # First valid detection: transition directly to TRACKING
                # The STARTING phase should only block for a brief moment to avoid startup jumps
                logger.info(f"TRACKING_STATE: previous=STARTING current=TRACKING reason=first_valid_detection confidence={primary_hand.confidence:.2f}")
                # Record motion baseline to prevent initial jump
                self._tracking_status.record_motion_baseline_established()
                # Log transition
                logger.info(f"TRACKING_STATE_TRANSITION: STARTING -> TRACKING confidence={primary_hand.confidence:.2f}")
            # Update tracking status timers and phase transitions
            self._tracking_status.update()

        # Handle PRECISION_MODE: switch to precision sensitivity when two hands tracked
        if tracking_result.tracking_state == TrackingState.PRECISION_MODE:
            self.cursor_controller.set_sensitivity_mode(SensitivityMode.PRECISION)
        elif self.cursor_controller.get_sensitivity_mode() == SensitivityMode.PRECISION:
            # Revert to normal when not in precision mode
            self.cursor_controller.set_sensitivity_mode(SensitivityMode.NORMAL)

        # Get cursor position from tracking processor (normalized from virtual plane)
        # For relative mode, use the primary hand directly
        norm_position = self.tracking_processor.get_cursor_position()

        # Get relative movement from cursor controller
        # In relative mode, use the primary hand for frame-to-frame deltas
        rel_movement = None
        if tracking_result.primary_hand is not None:
            rel_movement = self.cursor_controller.get_relative_movement(tracking_result.primary_hand)
        elif norm_position is not None:
            # Fallback for virtual plane mode
            rel_movement = self.cursor_controller.get_relative_movement_from_plane(norm_position[0], norm_position[1])

        frame_data.cursor_position = norm_position
        frame_data.mouse_movement = rel_movement
        self._last_mouse_movement = rel_movement

        # Gesture recognition using tracked hands
        gesture_start = time.time()
        
        # Check confidence-based permissions per spec §53
        conf_state = self._tracking_status.confidence.state
        can_gesture = self._tracking_status.get_gesture_permission()
        
        if not can_gesture:
            # Per §53: high-risk actions require higher confidence
            logger.debug(f"Gesture blocked: confidence state={conf_state.name}")
            # Still call recognizer to maintain state machine, but we won't act on events
        
        events = self.gesture_recognizer.process(tracked_hands)
        gesture_time = (time.time() - gesture_start) * 1000
        self.stats.gesture_time_ms = gesture_time
        self._frame_coord.record_stage_latency(frame_data, 'gesture', gesture_time)
        self.performance_monitor.update_gesture(gesture_time)

        frame_data.gesture_events = events
        
        # Apply confidence-based filtering to events per §53
        if not can_gesture:
            # Filter out all gesture events when confidence is low
            filtered_events = []
            for event in events:
                if event.gesture_type in (GestureType.LEFT_CLICK, GestureType.RIGHT_CLICK,
                                          GestureType.DRAG_START, GestureType.DRAG_END,
                                          GestureType.SCROLL_UP, GestureType.SCROLL_DOWN,
                                          GestureType.PAUSE_TRACKING):
                    # Skip high-risk gestures when confidence is low
                    continue
                filtered_events.append(event)
            events = filtered_events
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

        # Update landmarks for debug overlay
        if self.stats.frames_processed % 1 == 0:  # Every frame for smooth overlay
            if frame_data.frame is not None:
                h, w = frame_data.frame.shape[:2]
                self.performance_monitor.update_landmarks(hands, w, h)

        # Draw projection debug overlay showing plane axes, ray, and intersection
        if (self._debug_overlay_enabled and self.tracking_processor and
            hasattr(self.tracking_processor, 'get_projection_debug_overlay') and
            frame_data.frame is not None):
            try:
                h, w = frame_data.frame.shape[:2]
                overlay = self.tracking_processor.get_projection_debug_overlay(w, h)
                if overlay is not None:
                    # Blend overlay with original frame
                    mask = (overlay.sum(axis=2) > 0)
                    frame_data.frame[mask] = overlay[mask]
            except Exception as e:
                logger.debug(f"Projection debug overlay failed: {e}")

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

    def _debug_dump_pipeline(self, frame_data: FrameData):
        """Dump full pipeline state for debugging at intervals.
        
        Spec: Full pipeline observability on demand, rate-limited to DEBUG_INTERVAL
        to avoid log spam on low-spec laptops.
        """
        logger.log(DEBUG_LOG_LEVEL, "=== PIPELINE DEBUG DUMP ===")
        logger.log(DEBUG_LOG_LEVEL, f"frame_id={self._frame_count}")
        logger.log(DEBUG_LOG_LEVEL, f"phase={self._tracking_status.phase.name}")
        logger.log(DEBUG_LOG_LEVEL, f"conf_state={self._tracking_status.confidence.state.name}")
        logger.log(DEBUG_LOG_LEVEL, f"hands={len(frame_data.hands) if frame_data.hands else 0}")
        logger.log(DEBUG_LOG_LEVEL, f"tracking_state={frame_data.tracking_state}")
        if frame_data.hands:
            for h in frame_data.hands:
                logger.log(DEBUG_LOG_LEVEL,
                    f"  hand: id={h.id} conf={h.confidence:.2f} "
                    f"pos=({h.x:.3f},{h.y:.3f},{h.z:.3f})")
        if frame_data.primary_hand:
            ph = frame_data.primary_hand
            logger.log(DEBUG_LOG_LEVEL,
                f"  primary: id={ph.id} conf={ph.confidence:.2f} "
                f"screen=({ph.screen_x:.0f},{ph.screen_y:.0f})")
        logger.log(DEBUG_LOG_LEVEL,
            f"latency_ms: hand={self.stats.hand_detection_time_ms:.1f} "
            f"tracking={self.stats.cursor_time_ms:.1f} "
            f"gesture={self.stats.gesture_time_ms:.1f} "
            f"mouse={self.stats.mouse_time_ms:.1f}")
        logger.log(DEBUG_LOG_LEVEL, "=== END DEBUG DUMP ===")

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

        # Aggregate performance report every ~2 seconds (not every frame)
        # so we can see which stage dominates without log spam.
        if not hasattr(self, '_perf_report_time'):
            self._perf_report_time = 0.0
        if current_time - self._perf_report_time >= 2.0:
            self._perf_report_time = current_time
            self._report_performance_aggregate()

        # Update performance monitor latency breakdown
        if frame_data.latency_data:
            self.performance_monitor.update_latency(
                camera_to_landmark_ms=frame_data.latency_data.get('hand_detection', 0),
                landmark_to_pointer_ms=frame_data.latency_data.get('tracking', 0),
                end_to_end_ms=frame_data.latency_data.get('total', 0),
            )

        latency_stats = self._frame_coord.get_latency_stats()
        self.stats.pipeline_latency_ms = latency_stats['total']['mean']
        self.stats.stage_latencies = {k: v['mean'] for k, v in latency_stats['stages'].items()}
        self.stats.drop_rate = latency_stats['drop_rate']
        self.stats.frames_dropped = latency_stats['dropped_frames']

        self.stats.last_update = current_time

        if self.on_stats_update:
            self.on_stats_update(self.stats)

    def _report_performance_aggregate(self):
        """Print aggregate performance report every ~2 seconds."""
        n = len(self._frame_times)
        if n < 5:
            return
        avg_frame_time = sum(self._frame_times) / n
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        hand_ms = self.stats.hand_detection_time_ms
        face_ms = self.stats.gesture_time_ms  # reused for face detection
        track_ms = self.stats.cursor_time_ms
        gesture_ms = self.stats.gesture_time_ms
        mouse_ms = self.stats.mouse_time_ms

        # Estimate end-to-end: sum of all measured stages
        e2e = hand_ms + face_ms + track_ms + gesture_ms + mouse_ms

        logger.info(
            f"AirMouse performance: "
            f"FPS={fps:.1f} "
            f"frame={avg_frame_time * 1000:.1f}ms "
            f"inference={hand_ms:.1f}ms "
            f"face={face_ms:.1f}ms "
            f"projection={track_ms:.1f}ms "
            f"smoothing={gesture_ms:.1f}ms "
            f"input={mouse_ms:.1f}ms "
            f"end_to_end={e2e:.1f}ms"
        )

    def _handle_gestures(self, events: List[GestureEvent],
                         rel_movement: Optional[tuple], hands: List[TrackedHand]):
        """Handle gesture events and move mouse.

        §48: Single safety gate before desktop input. All input goes through
        this gate which validates:
        - Tracking stability (loss tracking prevents invalid frames)
        - Confidence thresholds (per §53)
        - Gesture safety (high-risk actions need higher confidence)

        Returns:
            bool: True if any input was delivered, False otherwise
        """
        if not self.input_manager:
            logger.warning("CURSOR_PIPELINE_FAILURE: stage=input_manager reason=none")
            return False

        # Determine if we should deliver any input at all
        should_deliver_input = self._should_deliver_input(
            rel_movement, events, hands
        )

        # Structured cursor pipeline diagnostics
        # Log the full cursor pipeline at each stage so failures can be
        # traced to the exact failing layer.  Every log line carries the
        # same frame_id so stale-state bugs are visible.
        self._frame_count = getattr(self, '_frame_count', 0) + 1
        frame_id = self._frame_count

        norm_pos = None
        if self.tracking_processor:
            norm_pos = self.tracking_processor.get_cursor_position()
        proj = None
        if self.tracking_processor and hasattr(self.tracking_processor, '_last_projection'):
            proj = self.tracking_processor._last_projection

        gate_open = should_deliver_input
        backend_ok = self.input_manager.is_healthy()
        backend_type = self.input_manager.get_backend_type().value
        backend_reason = self.input_manager.get_health_reason()

        # Log gate state with tracking phase for traceability
        tracking_phase = self._tracking_status.phase.name
        tracking_valid = tracking_phase in ("TRACKING", "STARTING")
        projection_valid = proj.valid if proj else False

        if should_deliver_input:
            logger.info(
                f"CURSOR_GATE: frame={frame_id} "
                f"tracking_state={tracking_phase} "
                f"tracking_valid={tracking_valid} "
                f"projection_valid={projection_valid} "
                f"u={norm_pos[0] if norm_pos else 'N/A'} "
                f"v={norm_pos[1] if norm_pos else 'N/A'} "
                f"gate=OPEN "
                f"backend={backend_type} "
                f"backend_ok={backend_ok}"
            )

        if not should_deliver_input:
            # Even if we have events, confidence/state might block them
            # Release any held buttons from previous successful operations
            if any(event.gesture_type in (GestureType.DRAG_START, GestureType.LEFT_CLICK,
                                        GestureType.RIGHT_CLICK, GestureType.MIDDLE_CLICK)
                   for event in events):
                self.input_manager.release_all()
            # Log pipeline failure with structured reason
            reason = "gate_closed"
            if self.input_manager and not backend_ok:
                reason = f"backend_unhealthy: {backend_reason}"
            elif self._tracking_status.phase == TrackingPhase.LOST:
                reason = "tracking_lost"
            elif self._tracking_status.phase == TrackingPhase.STARTING:
                reason = "tracking_starting"
            elif self._tracking_status.phase == TrackingPhase.TEMPORARY_LOSS:
                reason = "tracking_temporary_loss"
            elif self._tracking_status.phase == TrackingPhase.REACQUIRING:
                reason = "tracking_reacquiring"
            elif self._tracking_status.phase == TrackingPhase.STABILIZING:
                reason = "tracking_stabilizing"
            logger.warning(
                f"CURSOR_GATE_BLOCKED: frame={frame_id} stage=input_gate reason={reason} "
                f"tracking_state={tracking_phase} "
                f"tracking_valid={tracking_valid} "
                f"u={norm_pos[0] if norm_pos else 'N/A'} "
                f"v={norm_pos[1] if norm_pos else 'N/A'} "
                f"proj_valid={projection_valid} "
                f"backend={backend_type} backend_ok={backend_ok} "
                f"backend_reason={backend_reason}"
            )
            return False

        # Safe to deliver input - proceed with caution
        input_delivered = False

        # §7: Action-cursor coordination.  When a drag is in progress or a
        # click has just fired, freeze cursor movement so the hand's natural
        # micro-twitchs don't drag the selection off the target.  The cursor
        # is unfrozen when the drag ends or the next frame's movement is
        # large enough to indicate a deliberate reposition.
        cursor_frozen = self._cursor_frozen
        drag_active = self._drag_active

        # Move cursor if we have relative movement
        if rel_movement and (rel_movement[0] != 0 or rel_movement[1] != 0):
            if not cursor_frozen:
                move_ok = self.input_manager.move(rel_movement[0], rel_movement[1])
                # Update tracked cursor position so the corner-escape
                # detector (and velocity limiter) see the real cursor.
                x, y = self._cursor_position
                self._cursor_position = (x + rel_movement[0], y + rel_movement[1])
                input_delivered = True
                # Structured success log
                screen_x = x + rel_movement[0]
                screen_y = y + rel_movement[1]
                logger.info(
                    f"CURSOR_PIPELINE: frame={frame_id} "
                    f"u={norm_pos[0] if norm_pos else 'N/A'} "
                    f"v={norm_pos[1] if norm_pos else 'N/A'} "
                    f"proj_valid={proj.valid if proj else 'N/A'} "
                    f"dx={rel_movement[0]} dy={rel_movement[1]} "
                    f"screen_x={int(screen_x)} screen_y={int(screen_y)} "
                    f"gate_open=True backend={backend_type} "
                    f"move_attempted=True backend_result={'success' if move_ok else 'FAIL'}"
                )
                if not move_ok:
                    logger.error(
                        f"CURSOR_PIPELINE_FAILURE: frame={frame_id} stage=input_backend "
                        f"reason=move_returned_false backend={backend_type} "
                        f"backend_reason={backend_reason}"
                    )
            else:
                # Cursor frozen - still track position for corner escape
                x, y = self._cursor_position
                self._cursor_position = (x + rel_movement[0], y + rel_movement[1])
                logger.debug(
                    f"CURSOR_PIPELINE: frame={frame_id} cursor_frozen=True "
                    f"u={norm_pos[0] if norm_pos else 'N/A'} "
                    f"v={norm_pos[1] if norm_pos else 'N/A'} "
                    f"dx={rel_movement[0]} dy={rel_movement[1]}"
                )

        # Process gesture events
        for event in events:
            if not self._is_safe_gesture_event(event):
                continue
                
            # High-risk gestures need higher confidence validation
            if event.gesture_type in (GestureType.LEFT_CLICK, GestureType.RIGHT_CLICK,
                                    GestureType.MIDDLE_CLICK):
                # Double-check confidence for clicks
                if not self._has_sufficient_confidence_for_click():
                    logger.warning(f"Click blocked: insufficient confidence for {event.gesture_type.name}")
                    continue

            if event.gesture_type == GestureType.LEFT_CLICK:
                # Freeze cursor during the click so the hand's post-click
                # micro-movement doesn't drag the selection.
                self._cursor_frozen = True
                self.input_manager.click(1)
                input_delivered = True
                logger.debug("Left click")

            elif event.gesture_type == GestureType.RIGHT_CLICK:
                self._cursor_frozen = True
                self.input_manager.click(3)
                input_delivered = True
                logger.debug("Right click")

            elif event.gesture_type == GestureType.MIDDLE_CLICK:
                self._cursor_frozen = True
                self.input_manager.click(2)
                input_delivered = True
                logger.debug("Middle click")

            elif event.gesture_type == GestureType.DRAG_START:
                self._drag_active = True
                self._cursor_frozen = True
                self.input_manager.button_down(1)
                input_delivered = True
                logger.debug("Drag start")

            elif event.gesture_type == GestureType.DRAG_END:
                self._drag_active = False
                self._cursor_frozen = False
                self.input_manager.button_up(1)
                input_delivered = True
                logger.debug("Drag end")

            elif event.gesture_type == GestureType.SCROLL_UP:
                amount = event.data.get("amount", 3)
                self.input_manager.scroll(amount)
                input_delivered = True
                logger.debug(f"Scroll up: {amount}")

            elif event.gesture_type == GestureType.SCROLL_DOWN:
                amount = event.data.get("amount", 3)
                self.input_manager.scroll(-amount)
                input_delivered = True
                logger.debug(f"Scroll down: {amount}")

            elif event.gesture_type == GestureType.PINCH_END:
                # Release the mouse button for the original gesture
                original = event.data.get("original_gesture")
                if original == GestureType.LEFT_CLICK:
                    self.input_manager.button_up(1)
                    input_delivered = True
                    logger.debug("Left click release (pinch end)")
                elif original == GestureType.RIGHT_CLICK:
                    self.input_manager.button_up(3)
                    input_delivered = True
                    logger.debug("Right click release (pinch end)")
                elif original == GestureType.MIDDLE_CLICK:
                    self.input_manager.button_up(2)
                    input_delivered = True
                    logger.debug("Middle click release (pinch end)")

            elif event.gesture_type == GestureType.PAUSE_TRACKING:
                self.pause()
                input_delivered = True
                logger.info("Tracking paused (fist)")

            elif event.gesture_type == GestureType.RESUME_TRACKING:
                self.resume()
                input_delivered = True
                logger.info("Tracking resumed")

        # Notify GUI with processed frame and hands
        if self.on_frame_processed:
            self.on_frame_processed(self._last_frame, hands)

        # §7: Auto-unfreeze the cursor after a click.  A click freezes the
        # cursor so the hand's post-click micro-movement doesn't drag the
        # selection.  If no further click/drag event fires this frame, and
        # the hand has been stationary for a couple of frames, unfreeze so
        # the user can continue moving normally.
        if self._cursor_frozen and not self._drag_active:
            self._cursor_frozen_frames = getattr(self, '_cursor_frozen_frames', 0) + 1
            if self._cursor_frozen_frames >= 2:
                self._cursor_frozen = False
                self._cursor_frozen_frames = 0
        else:
            self._cursor_frozen_frames = 0

        return input_delivered

    def _should_deliver_input(self,
                              rel_movement: Optional[tuple],
                              events: List[GestureEvent],
                              hands: List[TrackedHand]) -> bool:
        """
        Single safety gate: decide if ANY input should be delivered this frame.
        
        Returns False if:
        - Tracking is lost (hand not reliably tracked)
        - Input device is unhealthy
        - Confidence state blocks all input
        """
        # Check input device health
        if self.input_manager and not self.input_manager.is_healthy():
            reason = self.input_manager.get_health_reason()
            logger.warning(f"Input gate: device unhealthy (reason={reason})")
            return False

        # Check tracking status
        if self._tracking_status.phase == TrackingPhase.LOST:
            return False

        # TEMPORARY_LOSS: hold position, block all input
        if self._tracking_status.phase == TrackingPhase.TEMPORARY_LOSS:
            return False

        # Check if we have any valid tracking state
        if self._tracking_status.phase == TrackingPhase.STARTING:
            return False

        # Check if confidence state allows input
        if not self._tracking_status.get_gesture_permission():
            # Still allow movement if it's just cursor motion without gestures
            has_gestures = len(events) > 0
            has_movement = rel_movement and (rel_movement[0] != 0 or rel_movement[1] != 0)
            if not has_movement and not has_gestures:
                return False

        return True

    def _is_safe_gesture_event(self, event: GestureEvent) -> bool:
        """
        Check if a gesture event is safe to execute.
        
        Blocks:
        - Events with very low confidence
        - Events that don't match the tracking state
        """
        # Low confidence events are filtered earlier, but double-check
        if hasattr(event, 'confidence') and event.confidence is not None:
            if event.confidence < 0.3:
                return False

        # Don't execute PAUSE/RESUME if already in that state
        if event.gesture_type == GestureType.PAUSE_TRACKING:
            return self.state == AirMouseState.RUNNING
        if event.gesture_type == GestureType.RESUME_TRACKING:
            return self.state == AirMouseState.PAUSED

        return True

    def _has_sufficient_confidence_for_click(self) -> bool:
        """
        Check if we have sufficient confidence for click gestures.
        
        Per §53: high-risk actions (clicks, drags) require HIGH confidence.
        
        NOTE: ConfidenceState enum values are HIGH=1, MEDIUM=2, LOW=3,
        LOST=4 (from auto()), so we must check for HIGH specifically —
        a naive ``value >= 2`` would accept MEDIUM/LOW/LOST and reject
        HIGH, which is the exact opposite of the intended gate.
        """
        conf_state = self._tracking_status.confidence.state
        return conf_state == ConfidenceState.HIGH

    def toggle_debug_overlay(self):
        """Toggle the performance debug overlay."""
        enabled = self.performance_monitor.toggle()
        logger.info(f"Debug overlay: {'enabled' if enabled else 'disabled'}")
        return enabled

    def _check_safety(self):
        """Check safety conditions using SafetyManager."""
        if self._safety_manager:
            # Feed the Wayland focus monitor with hand-activity so the
            # heuristic has something to work with when X11 is unavailable.
            # Record activity whenever a hand is detected (regardless of
            # whether the cursor moved) — a present hand means the user is
            # engaged.  Only recording activity on cursor movement let the
            # idle timer expire whenever the geometry pipeline produced a
            # zero-delta frame (e.g. during projection fallback or at
            # startup), causing spurious focus_loss pauses.
            if hasattr(self._safety_manager, '_focus_monitor') and self._safety_manager._focus_monitor:
                fm = self._safety_manager._focus_monitor
                if hasattr(fm, 'record_hand_activity'):
                    tracking = (
                        hasattr(self, 'tracking_processor')
                        and self.tracking_processor
                        and self.tracking_processor.is_tracking()
                    )
                    if tracking:
                        fm.record_hand_activity()
                    # Also record on cursor movement as a fallback for
                    # legacy mode where is_tracking may lag.
                    elif hasattr(self, '_last_mouse_movement') and self._last_mouse_movement:
                        dx, dy = self._last_mouse_movement
                        if dx != 0 or dy != 0:
                            fm.record_hand_activity()

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

        # Update performance monitor with system stats (every 10 frames)
        if self.stats.frames_processed % 10 == 0:
            self.performance_monitor.update_system_stats()

        if self.on_stats_update:
            self.on_stats_update(self.stats)

    def _recover_camera(self) -> bool:
        """Attempt to recover from camera failure per spec §48.

        Closes and reopens the camera, trying alternative devices if the
        current one fails. The device index may have changed (e.g., a USB
        camera replugged) so we re-detect rather than assume the original
        index is still valid.

        Returns:
            True if recovery succeeded, False otherwise.
        """
        logger.info("Attempting camera recovery...")

        # Release frozen input before recovery attempt
        if self.input_manager:
            try:
                self.input_manager.release_all()
            except Exception:
                pass

        # Close current camera
        try:
            self.camera.close_camera()
        except Exception as e:
            logger.warning(f"Camera close error during recovery: {e}")

        # Re-detect cameras - the device may have changed (spec §48)
        try:
            cameras = self.camera.detect_cameras()
            if not cameras:
                logger.error("Camera recovery failed: no devices detected")
                return False

            # Try each available camera until one works
            available = [c for c in cameras if c.available]
            if not available:
                logger.error("Camera recovery failed: no available devices")
                return False

            for cam in available:
                logger.info(f"Trying camera {cam.device_path} (index {cam.index})...")
                self.config.camera.device_index = cam.index
                self.config.camera.device_path = cam.device_path
                if self.camera.open_camera(self.config.camera):
                    logger.info(f"Camera recovered: {cam.device_path}")
                    # Update cursor config with actual resolution
                    actual_width, actual_height = self.camera.get_resolution()
                    if self.cursor_controller:
                        self.config.cursor.camera_width = actual_width
                        self.config.cursor.camera_height = actual_height
                    return True

            logger.error("Camera recovery failed: all devices failed to open")
            return False
        except Exception as e:
            logger.error(f"Camera recovery error: {e}")
            return False

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