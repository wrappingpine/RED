"""
Performance Monitor Debug Overlay for Air Mouse

Press 'P' to toggle the debug overlay showing:
- FPS and frame time
- Hand detection time
- Gesture recognition time
- Cursor tracking time
- Mouse event time
- CPU/memory usage
- Brightness statistics
- Head tracking status
"""

import cv2
import numpy as np
import time
import psutil
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import deque


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    # Frame timing
    fps: float = 0.0
    frame_time_ms: float = 0.0

    # Pipeline stages (ms)
    hand_detection_ms: float = 0.0
    face_detection_ms: float = 0.0
    tracking_ms: float = 0.0
    gesture_ms: float = 0.0
    cursor_ms: float = 0.0
    mouse_ms: float = 0.0

    # System resources
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0

    # Camera
    brightness: float = 0.0
    exposure: int = -1
    gain: int = -1

    # Tracking state
    hand_detected: bool = False
    face_detected: bool = False
    hand_state: str = "NO_HAND"
    tracking_mode: str = "2D"  # "2D" or "3D_HEAD_RELATIVE"

    # Gestures
    active_gestures: List[str] = field(default_factory=list)

    # Cursor
    cursor_pos: tuple = (0, 0)
    cursor_velocity: tuple = (0.0, 0.0)


class PerformanceMonitor:
    """
    Performance monitor with toggleable debug overlay.

    Usage:
        monitor = PerformanceMonitor()

        # In main loop:
        monitor.update_frame_start()
        # ... process frame ...
        monitor.update_hand_detection(time_ms)
        monitor.update_face_detection(time_ms)
        monitor.update_tracking(time_ms)
        monitor.update_gesture(time_ms)
        monitor.update_cursor(time_ms)
        monitor.update_mouse(time_ms)
        monitor.update_system_stats()
        monitor.update_tracking_state(state_dict)

        # Draw overlay (call after all processing)
        frame = monitor.draw_overlay(frame)

        # Toggle with 'P' key
        if key == ord('p') or key == ord('P'):
            monitor.toggle()
    """

    def __init__(self,
                 history_size: int = 60,
                 position: tuple = (10, 30),
                 font_scale: float = 0.5,
                 font_color: tuple = (0, 255, 0),
                 bg_color: tuple = (0, 0, 0),
                 bg_alpha: float = 0.7):
        self.history_size = history_size
        self.position = position
        self.font_scale = font_scale
        self.font_color = font_color
        self.bg_color = bg_color
        self.bg_alpha = bg_alpha

        # State
        self._enabled = False
        self._lock = threading.Lock()

        # Metrics history for graphs
        self._fps_history = deque(maxlen=history_size)
        self._frame_time_history = deque(maxlen=history_size)
        self._cpu_history = deque(maxlen=history_size)
        self._memory_history = deque(maxlen=history_size)

        # Current metrics
        self._metrics = PerformanceMetrics()

        # Timing
        self._frame_start_time = 0.0
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._fps_update_time = time.time()

        # System stats process
        self._process = psutil.Process()
        self._last_cpu_time = 0.0

        # Font settings
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._line_height = 20
        self._section_spacing = 10

    def toggle(self) -> bool:
        """Toggle overlay visibility. Returns new state."""
        with self._lock:
            self._enabled = not self._enabled
            return self._enabled

    def is_enabled(self) -> bool:
        """Check if overlay is enabled."""
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool):
        """Set overlay enabled state."""
        with self._lock:
            self._enabled = enabled

    def update_frame_start(self):
        """Call at start of frame processing."""
        self._frame_start_time = time.time()

    def update_frame_end(self):
        """Call at end of frame processing."""
        current_time = time.time()
        frame_time = current_time - self._frame_start_time
        self._last_frame_time = frame_time

        # Update FPS calculation
        self._frame_count += 1
        if current_time - self._fps_update_time >= 1.0:
            with self._lock:
                self._metrics.fps = self._frame_count / (current_time - self._fps_update_time)
                self._metrics.frame_time_ms = frame_time * 1000
                self._fps_history.append(self._metrics.fps)
                self._frame_time_history.append(self._metrics.frame_time_ms)
            self._frame_count = 0
            self._fps_update_time = current_time

    def update_hand_detection(self, time_ms: float):
        """Update hand detection timing."""
        with self._lock:
            self._metrics.hand_detection_ms = time_ms

    def update_face_detection(self, time_ms: float):
        """Update face detection timing."""
        with self._lock:
            self._metrics.face_detection_ms = time_ms

    def update_tracking(self, time_ms: float):
        """Update tracking processor timing."""
        with self._lock:
            self._metrics.tracking_ms = time_ms

    def update_gesture(self, time_ms: float):
        """Update gesture recognition timing."""
        with self._lock:
            self._metrics.gesture_ms = time_ms

    def update_cursor(self, time_ms: float):
        """Update cursor controller timing."""
        with self._lock:
            self._metrics.cursor_ms = time_ms

    def update_mouse(self, time_ms: float):
        """Update virtual mouse timing."""
        with self._lock:
            self._metrics.mouse_ms = time_ms

    def update_system_stats(self):
        """Update CPU and memory usage."""
        try:
            cpu = self._process.cpu_percent(interval=0)
            mem_info = self._process.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)
            mem_percent = self._process.memory_percent()

            with self._lock:
                self._metrics.cpu_percent = cpu
                self._metrics.memory_mb = mem_mb
                self._metrics.memory_percent = mem_percent
                self._cpu_history.append(cpu)
                self._memory_history.append(mem_mb)
        except Exception:
            pass

    def update_camera_stats(self, brightness: float, exposure: int = -1, gain: int = -1):
        """Update camera statistics."""
        with self._lock:
            self._metrics.brightness = brightness
            self._metrics.exposure = exposure
            self._metrics.gain = gain

    def update_tracking_state(self,
                              hand_detected: bool = False,
                              face_detected: bool = False,
                              hand_state: str = "NO_HAND",
                              tracking_mode: str = "2D",
                              active_gestures: List[str] = None,
                              cursor_pos: tuple = (0, 0),
                              cursor_velocity: tuple = (0.0, 0.0)):
        """Update tracking state information."""
        with self._lock:
            self._metrics.hand_detected = hand_detected
            self._metrics.face_detected = face_detected
            self._metrics.hand_state = hand_state
            self._metrics.tracking_mode = tracking_mode
            self._metrics.active_gestures = active_gestures or []
            self._metrics.cursor_pos = cursor_pos
            self._metrics.cursor_velocity = cursor_velocity

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw performance overlay on frame. Returns annotated frame."""
        if not self._enabled:
            return frame

        # Create a copy for drawing
        annotated = frame.copy()

        with self._lock:
            metrics = self._metrics

        # Draw background panel
        self._draw_background_panel(annotated)

        # Draw metrics text
        self._draw_metrics_text(annotated, metrics)

        # Draw mini graphs
        self._draw_graphs(annotated)

        return annotated

    def _draw_background_panel(self, frame: np.ndarray):
        """Draw semi-transparent background panel."""
        h, w = frame.shape[:2]

        # Panel dimensions
        panel_width = 320
        panel_height = min(h - 20, 480)
        panel_x = w - panel_width - 10
        panel_y = 10

        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     self.bg_color, -1)

        # Blend
        cv2.addWeighted(overlay, self.bg_alpha, frame, 1 - self.bg_alpha, 0, frame)

        # Store panel bounds for text positioning
        self._panel_bounds = (panel_x, panel_y, panel_width, panel_height)

    def _draw_metrics_text(self, frame: np.ndarray, metrics: PerformanceMetrics):
        """Draw metrics text on frame."""
        if not hasattr(self, '_panel_bounds'):
            return

        panel_x, panel_y, panel_width, panel_height = self._panel_bounds
        x = panel_x + 10
        y = panel_y + 20

        def draw_line(text: str, color=None, bold=False):
            nonlocal y
            cv2.putText(frame, text, (x, y), self._font, self.font_scale,
                       color or self.font_color, 2 if bold else 1, cv2.LINE_AA)
            y += self._line_height

        def draw_section(title: str):
            nonlocal y
            y += self._section_spacing
            draw_line(title, color=(0, 255, 255), bold=True)

        # Title
        draw_line("AIR MOUSE PERFORMANCE", bold=True)
        draw_line("=" * 30, color=(100, 100, 100))

        # Frame timing
        draw_section("FRAME TIMING")
        draw_line(f"FPS: {metrics.fps:.1f}")
        draw_line(f"Frame Time: {metrics.frame_time_ms:.1f} ms")

        # Pipeline stages
        draw_section("PIPELINE (ms)")
        draw_line(f"  Hand Detection:  {metrics.hand_detection_ms:.1f}")
        draw_line(f"  Face Detection:  {metrics.face_detection_ms:.1f}")
        draw_line(f"  Tracking:        {metrics.tracking_ms:.1f}")
        draw_line(f"  Gestures:        {metrics.gesture_ms:.1f}")
        draw_line(f"  Cursor:          {metrics.cursor_ms:.1f}")
        draw_line(f"  Mouse:           {metrics.mouse_ms:.1f}")
        total_pipeline = (metrics.hand_detection_ms + metrics.face_detection_ms +
                         metrics.tracking_ms + metrics.gesture_ms +
                         metrics.cursor_ms + metrics.mouse_ms)
        draw_line(f"  TOTAL:           {total_pipeline:.1f}")

        # System resources
        draw_section("SYSTEM RESOURCES")
        draw_line(f"CPU: {metrics.cpu_percent:.1f}%")
        draw_line(f"RAM: {metrics.memory_mb:.0f} MB ({metrics.memory_percent:.1f}%)")

        # Camera
        draw_section("CAMERA")
        draw_line(f"Brightness: {metrics.brightness:.0f}")
        if metrics.exposure >= 0:
            draw_line(f"Exposure: {metrics.exposure}")
        if metrics.gain >= 0:
            draw_line(f"Gain: {metrics.gain}")

        # Tracking state
        draw_section("TRACKING STATE")
        hand_color = (0, 255, 0) if metrics.hand_detected else (0, 0, 255)
        face_color = (0, 255, 0) if metrics.face_detected else (0, 0, 255)
        draw_line(f"Hand: {'DETECTED' if metrics.hand_detected else 'NONE'}", color=hand_color)
        draw_line(f"Face: {'DETECTED' if metrics.face_detected else 'NONE'}", color=face_color)
        draw_line(f"State: {metrics.hand_state}")
        draw_line(f"Mode: {metrics.tracking_mode}")

        # Gestures
        if metrics.active_gestures:
            draw_section("ACTIVE GESTURES")
            for g in metrics.active_gestures:
                draw_line(f"  - {g}")

        # Cursor
        draw_section("CURSOR")
        draw_line(f"Position: ({metrics.cursor_pos[0]}, {metrics.cursor_pos[1]})")
        draw_line(f"Velocity: ({metrics.cursor_velocity[0]:.1f}, {metrics.cursor_velocity[1]:.1f})")

        # Controls hint
        y += self._section_spacing
        draw_line("Press 'P' to toggle overlay", color=(180, 180, 180))

    def _draw_graphs(self, frame: np.ndarray):
        """Draw mini performance graphs."""
        if not hasattr(self, '_panel_bounds'):
            return

        panel_x, panel_y, panel_width, panel_height = self._panel_bounds

        # Graph area below text
        graph_x = panel_x + 10
        graph_y = panel_y + 380
        graph_width = panel_width - 20
        graph_height = 80

        if graph_y + graph_height > panel_y + panel_height:
            return  # Not enough space

        # Draw FPS graph
        self._draw_mini_graph(frame, self._fps_history, graph_x, graph_y,
                             graph_width, graph_height // 2,
                             "FPS", (0, 255, 0), max_val=60)

        # Draw CPU graph
        self._draw_mini_graph(frame, self._cpu_history, graph_x, graph_y + graph_height // 2 + 5,
                             graph_width, graph_height // 2,
                             "CPU %", (255, 100, 100), max_val=100)

    def _draw_mini_graph(self, frame: np.ndarray, history: deque,
                        x: int, y: int, w: int, h: int,
                        label: str, color: tuple, max_val: float):
        """Draw a mini line graph."""
        if len(history) < 2:
            return

        # Background
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)

        # Label
        cv2.putText(frame, label, (x, y + 12), self._font, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

        # Grid lines
        cv2.line(frame, (x, y + h), (x + w, y + h), (60, 60, 60), 1)
        cv2.line(frame, (x, y + h // 2), (x + w, y + h // 2), (60, 60, 60), 1)

        # Plot data
        points = list(history)
        n = len(points)
        for i in range(1, n):
            x1 = x + int((i - 1) * w / max(1, n - 1))
            x2 = x + int(i * w / max(1, n - 1))

            y1 = y + h - int(points[i - 1] / max_val * h)
            y2 = y + h - int(points[i] / max_val * h)

            # Clamp
            y1 = max(y, min(y + h, y1))
            y2 = max(y, min(y + h, y2))

            cv2.line(frame, (x1, y1), (x2, y2), color, 1)


def create_performance_monitor(**kwargs) -> PerformanceMonitor:
    """Factory function to create a PerformanceMonitor."""
    return PerformanceMonitor(**kwargs)


# Integration example for main_loop.py:
"""
# In AirMouseController.__init__:
from ..debug.performance_monitor import PerformanceMonitor
self.performance_monitor = PerformanceMonitor()

# In _process_frame:
self.performance_monitor.update_frame_start()

# After hand detection:
self.performance_monitor.update_hand_detection(self.stats.hand_detection_time_ms)

# After face detection:
self.performance_monitor.update_face_detection(self.stats.gesture_time_ms)  # or face_time_ms

# After tracking:
self.performance_monitor.update_tracking(self.stats.cursor_time_ms)

# After gestures:
self.performance_monitor.update_gesture(self.stats.gesture_time_ms)

# After cursor:
self.performance_monitor.update_cursor(self.stats.cursor_time_ms)

# After mouse:
self.performance_monitor.update_mouse(self.stats.mouse_time_ms)

# System stats (every few frames):
if self.stats.frames_processed % 10 == 0:
    self.performance_monitor.update_system_stats()

# Camera stats:
if self.camera:
    stats = self.camera.get_brightness_stats()
    self.performance_monitor.update_camera_stats(
        stats.get('brightness', 0),
        stats.get('exposure', -1),
        stats.get('gain', -1)
    )

# Tracking state:
if tracked_hand:
    self.performance_monitor.update_tracking_state(
        hand_detected=True,
        face_detected=len(faces) > 0,
        hand_state=tracked_hand.tracking_state.value if hasattr(tracked_hand, 'tracking_state') else "TRACKING",
        tracking_mode="3D_HEAD_RELATIVE" if self.config.tracking.use_head_relative else "2D",
        active_gestures=[e.gesture_type.value for e in events],
        cursor_pos=self.virtual_mouse.get_position() if self.virtual_mouse else (0, 0),
        cursor_velocity=rel_movement if rel_movement else (0, 0)
    )

# Draw overlay at end of _process_frame:
self._last_frame = self.performance_monitor.draw_overlay(self._last_frame)
"""


if __name__ == "__main__":
    # Standalone test
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from camera.manager import CameraManager, CameraSettings

    monitor = PerformanceMonitor()
    camera = CameraManager()

    if not camera.open_camera(CameraSettings(device_index=0, width=640, height=480)):
        print("Failed to open camera")
        sys.exit(1)

    print("Performance Monitor Test - Press 'P' to toggle overlay, 'Q' to quit")

    try:
        while True:
            monitor.update_frame_start()

            ret, frame = camera.read_frame()
            if not ret:
                continue

            # Simulate some processing
            time.sleep(0.005)  # 5ms processing

            monitor.update_hand_detection(3.2)
            monitor.update_face_detection(2.1)
            monitor.update_tracking(1.5)
            monitor.update_gesture(0.8)
            monitor.update_cursor(0.5)
            monitor.update_mouse(0.2)
            monitor.update_system_stats()

            stats = camera.get_brightness_stats()
            monitor.update_camera_stats(
                stats.get('brightness', 0),
                stats.get('exposure', -1),
                stats.get('gain', -1)
            )

            monitor.update_tracking_state(
                hand_detected=True,
                face_detected=True,
                hand_state="TRACKING_ONE_HAND",
                tracking_mode="3D_HEAD_RELATIVE",
                active_gestures=["POINTING", "SCROLL"],
                cursor_pos=(960, 540),
                cursor_velocity=(10.5, -5.2)
            )

            monitor.update_frame_end()

            frame = monitor.draw_overlay(frame)

            cv2.imshow("Performance Monitor Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('p') or key == ord('P'):
                monitor.toggle()
                print(f"Overlay: {'ON' if monitor.is_enabled() else 'OFF'}")

    finally:
        camera.close_camera()
        cv2.destroyAllWindows()