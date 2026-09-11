"""
Emergency Safety Mode for Air Mouse

Provides multiple layers of safety:
1. Corner escape - move mouse to screen corner to emergency stop
2. Global hotkey - Super+Alt+A for immediate disable
3. Auto-pause on focus loss - pause when target window loses focus
4. Velocity limiting - prevent runaway cursor
5. Gesture confirmation - require stable gesture before action
6. Failsafe timer - auto-disable after inactivity
"""

import os
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, List, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SafetyTrigger(Enum):
    """What triggered the safety mode."""
    CORNER_ESCAPE = "corner_escape"
    GLOBAL_HOTKEY = "global_hotkey"
    FOCUS_LOSS = "focus_loss"
    VELOCITY_LIMIT = "velocity_limit"
    GESTURE_TIMEOUT = "gesture_timeout"
    INACTIVITY_TIMEOUT = "inactivity_timeout"
    MANUAL = "manual"
    ERROR = "error"


class SafetyLevel(Enum):
    """Safety response level."""
    NONE = 0           # No action
    PAUSE = 1          # Pause tracking
    DISABLE = 2        # Disable completely (requires manual re-enable)
    EMERGENCY = 3      # Emergency stop - release all buttons, reset state


@dataclass
class SafetyConfig:
    """Safety system configuration."""
    # Corner escape
    enable_corner_escape: bool = True
    corner_size: int = 50  # pixels from corner
    corner_hold_time: float = 0.5  # seconds in corner to trigger

    # Global hotkey (handled by hotkey manager)
    enable_global_hotkey: bool = True

    # Focus loss
    enable_focus_loss_pause: bool = True
    focus_check_interval: float = 0.5  # seconds

    # Velocity limiting
    enable_velocity_limit: bool = True
    max_cursor_velocity: float = 5000  # pixels/second
    max_acceleration: float = 20000  # pixels/second^2

    # Gesture confirmation
    enable_gesture_confirmation: bool = True
    min_gesture_duration: float = 0.15  # seconds
    min_stable_frames: int = 3

    # Inactivity timeout
    enable_inactivity_timeout: bool = False
    inactivity_timeout: float = 300  # seconds (5 minutes)

    # Emergency stop behavior
    release_all_buttons: bool = True
    reset_cursor_position: bool = False
    show_notification: bool = True


@dataclass
class SafetyEvent:
    """Safety event record."""
    timestamp: float
    trigger: SafetyTrigger
    level: SafetyLevel
    details: str = ""
    cursor_position: Tuple[int, int] = (0, 0)


class CornerEscapeDetector:
    """Detects when cursor is held in screen corner."""

    def __init__(self, config: SafetyConfig, get_cursor_pos: Callable[[], Tuple[int, int]],
                 get_screen_size: Callable[[], Tuple[int, int]]):
        self._config = config
        self._get_cursor_pos = get_cursor_pos
        self._get_screen_size = get_screen_size
        self._corner_enter_time: Optional[float] = None
        self._in_corner = False
        self._triggered = False

    def check(self) -> bool:
        """Check if corner escape is triggered. Returns True if triggered."""
        if not self._config.enable_corner_escape:
            return False

        if self._triggered:
            return True

        x, y = self._get_cursor_pos()
        width, height = self._get_screen_size()
        corner_size = self._config.corner_size

        # Check all four corners
        in_corner = (
            (x <= corner_size and y <= corner_size) or  # Top-left
            (x >= width - corner_size and y <= corner_size) or  # Top-right
            (x <= corner_size and y >= height - corner_size) or  # Bottom-left
            (x >= width - corner_size and y >= height - corner_size)  # Bottom-right
        )

        if in_corner and not self._in_corner:
            self._in_corner = True
            self._corner_enter_time = time.time()
        elif not in_corner:
            self._in_corner = False
            self._corner_enter_time = None

        if self._in_corner and self._corner_enter_time:
            if time.time() - self._corner_enter_time >= self._config.corner_hold_time:
                self._triggered = True
                logger.warning("Corner escape triggered!")
                return True

        return False

    def reset(self):
        """Reset the detector."""
        self._corner_enter_time = None
        self._in_corner = False
        self._triggered = False


class VelocityLimiter:
    """Limits cursor velocity and acceleration for safety."""

    def __init__(self, config: SafetyConfig):
        self._config = config
        self._last_position: Optional[Tuple[int, int]] = None
        self._last_time: Optional[float] = None
        self._last_velocity: Tuple[float, float] = (0.0, 0.0)

    def check_and_limit(self, dx: int, dy: int, dt: float) -> Tuple[int, int]:
        """
        Check velocity and acceleration, limit if needed.

        Args:
            dx, dy: Proposed movement delta
            dt: Time delta in seconds

        Returns:
            Limited (dx, dy)
        """
        if not self._config.enable_velocity_limit or dt <= 0:
            return dx, dy

        # Calculate current velocity
        velocity_x = dx / dt
        velocity_y = dy / dt
        speed = np.hypot(velocity_x, velocity_y)

        if speed > self._config.max_cursor_velocity:
            # Scale down to max velocity
            scale = self._config.max_cursor_velocity / speed
            dx = int(dx * scale)
            dy = int(dy * scale)
            logger.warning(f"Velocity limited: {speed:.0f} -> {self._config.max_cursor_velocity:.0f} px/s")

        # Check acceleration if we have previous velocity
        if self._last_velocity != (0.0, 0.0):
            accel_x = (velocity_x - self._last_velocity[0]) / dt
            accel_y = (velocity_y - self._last_velocity[1]) / dt
            accel = np.hypot(accel_x, accel_y)

            if accel > self._config.max_acceleration:
                scale = self._config.max_acceleration / accel
                dx = int(dx * scale)
                dy = int(dy * scale)
                logger.warning(f"Acceleration limited: {accel:.0f} -> {self._config.max_acceleration:.0f} px/s²")

        self._last_velocity = (dx / dt, dy / dt)
        return dx, dy

    def update_position(self, x: int, y: int, timestamp: float):
        """Update last known position."""
        self._last_position = (x, y)
        self._last_time = timestamp

    def reset(self):
        """Reset velocity tracking."""
        self._last_position = None
        self._last_time = None
        self._last_velocity = (0.0, 0.0)


class FocusMonitor:
    """Monitors window focus for auto-pause."""

    def __init__(self, config: SafetyConfig, on_focus_lost: Callable, on_focus_gained: Callable):
        self._config = config
        self._on_focus_lost = on_focus_lost
        self._on_focus_gained = on_focus_gained
        self._last_focus_check = 0
        self._has_focus = True
        self._monitor_thread = None
        self._running = False

    def start(self):
        """Start focus monitoring."""
        if not self._config.enable_focus_loss_pause:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Focus monitor started")

    def stop(self):
        """Stop focus monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Monitor loop."""
        while self._running:
            try:
                has_focus = self._check_focus()
                now = time.time()

                if not has_focus and self._has_focus:
                    logger.info("Focus lost - pausing")
                    self._has_focus = False
                    self._on_focus_lost()
                elif has_focus and not self._has_focus:
                    logger.info("Focus gained - resuming")
                    self._has_focus = True
                    self._on_focus_gained()

                self._last_focus_check = now
                time.sleep(self._config.focus_check_interval)
            except Exception as e:
                logger.error(f"Focus monitor error: {e}")
                time.sleep(1.0)

    def _check_focus(self) -> bool:
        """Check if Air Mouse window has focus."""
        # This is a simplified check - in practice would need to check
        # if our window or the target application has focus
        # For now, return True (would integrate with UI)
        return True


class InactivityTimer:
    """Timer for auto-disable after inactivity."""

    def __init__(self, config: SafetyConfig, on_timeout: Callable):
        self._config = config
        self._on_timeout = on_timeout
        self._last_activity = time.time()
        self._timer_thread = None
        self._running = False

    def start(self):
        """Start inactivity timer."""
        if not self._config.enable_inactivity_timeout:
            return

        self._running = True
        self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._timer_thread.start()

    def stop(self):
        """Stop inactivity timer."""
        self._running = False
        if self._timer_thread:
            self._timer_thread.join(timeout=1.0)

    def record_activity(self):
        """Record user activity."""
        self._last_activity = time.time()

    def _timer_loop(self):
        """Timer loop."""
        while self._running:
            try:
                if time.time() - self._last_activity >= self._config.inactivity_timeout:
                    logger.warning("Inactivity timeout triggered")
                    self._on_timeout(SafetyTrigger.INACTIVITY_TIMEOUT)
                    break
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Inactivity timer error: {e}")
                break


class SafetyManager:
    """
    Central safety manager coordinating all safety mechanisms.

    Integrates with:
    - Global hotkey manager (Super+Alt+A)
    - Corner escape detector
    - Focus monitor
    - Velocity limiter
    - Inactivity timer
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self._config = config or SafetyConfig()
        self._callbacks: List[Callable[[SafetyEvent], None]] = []
        self._event_history: List[SafetyEvent] = []
        self._max_history = 100

        # Components
        self._corner_detector: Optional[CornerEscapeDetector] = None
        self._velocity_limiter: Optional[VelocityLimiter] = None
        self._focus_monitor: Optional[FocusMonitor] = None
        self._inactivity_timer: Optional[InactivityTimer] = None

        # State
        self._enabled = False
        self._safety_active = False
        self._current_level = SafetyLevel.NONE
        self._get_cursor_pos: Optional[Callable] = None
        self._get_screen_size: Optional[Callable] = None
        self._release_all_callback: Optional[Callable] = None
        self._pause_callback: Optional[Callable] = None
        self._disable_callback: Optional[Callable] = None
        self._show_notification: Optional[Callable] = None

        # Monitoring thread
        self._monitor_thread = None
        self._running = False

    def set_callbacks(self,
                      get_cursor_pos: Callable[[], Tuple[int, int]],
                      get_screen_size: Callable[[], Tuple[int, int]],
                      release_all: Callable,
                      pause: Callable,
                      disable: Callable,
                      show_notification: Optional[Callable] = None):
        """Set required callbacks."""
        self._get_cursor_pos = get_cursor_pos
        self._get_screen_size = get_screen_size
        self._release_all_callback = release_all
        self._pause_callback = pause
        self._disable_callback = disable
        self._show_notification = show_notification

        # Initialize components
        self._corner_detector = CornerEscapeDetector(
            self._config, get_cursor_pos, get_screen_size
        )
        self._velocity_limiter = VelocityLimiter(self._config)
        self._focus_monitor = FocusMonitor(
            self._config,
            on_focus_lost=self._on_focus_lost,
            on_focus_gained=self._on_focus_gained
        )
        self._inactivity_timer = InactivityTimer(
            self._config,
            on_timeout=self._on_inactivity_timeout
        )

    def add_callback(self, callback: Callable[[SafetyEvent], None]):
        """Add safety event callback."""
        self._callbacks.append(callback)

    def enable(self):
        """Enable safety monitoring."""
        if self._enabled:
            return

        if not self._get_cursor_pos or not self._get_screen_size:
            logger.error("Cannot enable safety: callbacks not set")
            return

        self._enabled = True
        self._safety_active = False
        self._current_level = SafetyLevel.NONE

        # Start components
        if self._focus_monitor:
            self._focus_monitor.start()
        if self._inactivity_timer:
            self._inactivity_timer.start()

        # Start monitoring thread
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info("Safety manager enabled")

    def disable(self):
        """Disable safety monitoring."""
        self._enabled = False
        self._running = False

        if self._focus_monitor:
            self._focus_monitor.stop()
        if self._inactivity_timer:
            self._inactivity_timer.stop()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

        logger.info("Safety manager disabled")

    def _monitor_loop(self):
        """Main safety monitoring loop."""
        while self._running:
            try:
                self._check_safety()
                time.sleep(0.05)  # 20Hz check rate
            except Exception as e:
                logger.error(f"Safety monitor error: {e}")
                time.sleep(0.5)

    def _check_safety(self):
        """Check all safety conditions."""
        if not self._enabled:
            return

        # Check corner escape
        if self._corner_detector and self._corner_detector.check():
            self._trigger_safety(SafetyTrigger.CORNER_ESCAPE, SafetyLevel.EMERGENCY,
                               "Cursor held in screen corner")

        # Note: Global hotkey is handled separately by hotkey manager
        # calling trigger_emergency() directly

    def _on_focus_lost(self):
        """Handle focus lost."""
        if self._config.enable_focus_loss_pause:
            self._trigger_safety(SafetyTrigger.FOCUS_LOSS, SafetyLevel.PAUSE,
                               "Window lost focus")

    def _on_focus_gained(self):
        """Handle focus gained."""
        # Auto-resume is optional - for now just log
        logger.info("Focus regained - manual resume required")

    def _on_inactivity_timeout(self, trigger: SafetyTrigger):
        """Handle inactivity timeout."""
        self._trigger_safety(trigger, SafetyLevel.DISABLE,
                           f"Inactive for {self._config.inactivity_timeout}s")

    def check_velocity(self, dx: int, dy: int, dt: float) -> Tuple[int, int]:
        """Check and limit velocity."""
        if self._velocity_limiter:
            return self._velocity_limiter.check_and_limit(dx, dy, dt)
        return dx, dy

    def update_cursor_position(self, x: int, y: int, timestamp: float):
        """Update cursor position for velocity tracking."""
        if self._velocity_limiter:
            self._velocity_limiter.update_position(x, y, timestamp)

    def record_activity(self):
        """Record user activity for inactivity timer."""
        if self._inactivity_timer:
            self._inactivity_timer.record_activity()

    def trigger_emergency(self, trigger: SafetyTrigger = SafetyTrigger.GLOBAL_HOTKEY):
        """Trigger emergency stop (called by hotkey manager)."""
        self._trigger_safety(trigger, SafetyLevel.EMERGENCY, "Emergency stop triggered")

    def trigger_pause(self, trigger: SafetyTrigger = SafetyTrigger.MANUAL):
        """Trigger pause."""
        self._trigger_safety(trigger, SafetyLevel.PAUSE, "Pause requested")

    def trigger_disable(self, trigger: SafetyTrigger = SafetyTrigger.MANUAL):
        """Trigger disable."""
        self._trigger_safety(trigger, SafetyLevel.DISABLE, "Disable requested")

    def _trigger_safety(self, trigger: SafetyTrigger, level: SafetyLevel, details: str):
        """Trigger safety response."""
        if self._safety_active and level <= self._current_level:
            return  # Already at same or higher level

        x, y = (0, 0)
        if self._get_cursor_pos:
            x, y = self._get_cursor_pos()

        event = SafetyEvent(
            timestamp=time.time(),
            trigger=trigger,
            level=level,
            details=details,
            cursor_position=(x, y)
        )

        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # Execute safety action
        self._execute_safety_action(level)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Safety callback error: {e}")

        # Show notification
        if self._config.show_notification and self._show_notification:
            try:
                self._show_notification(f"Air Mouse Safety: {trigger.value}", details)
            except Exception:
                pass

        logger.warning(f"Safety triggered: {trigger.value} -> {level.name}: {details}")

    def _execute_safety_action(self, level: SafetyLevel):
        """Execute safety action based on level."""
        self._safety_active = True
        self._current_level = level

        if level >= SafetyLevel.EMERGENCY:
            if self._config.release_all_buttons and self._release_all_callback:
                self._release_all_callback()
            if self._disable_callback:
                self._disable_callback()
        elif level >= SafetyLevel.DISABLE:
            if self._config.release_all_buttons and self._release_all_callback:
                self._release_all_callback()
            if self._disable_callback:
                self._disable_callback()
        elif level >= SafetyLevel.PAUSE:
            if self._pause_callback:
                self._pause_callback()

        # Reset corner detector after trigger
        if self._corner_detector:
            self._corner_detector.reset()

    def reset(self):
        """Reset safety state (called when user manually re-enables)."""
        self._safety_active = False
        self._current_level = SafetyLevel.NONE

        if self._corner_detector:
            self._corner_detector.reset()
        if self._velocity_limiter:
            self._velocity_limiter.reset()
        if self._inactivity_timer:
            self._inactivity_timer.record_activity()

        logger.info("Safety manager reset")

    def is_safety_active(self) -> bool:
        return self._safety_active

    def get_current_level(self) -> SafetyLevel:
        return self._current_level

    def get_event_history(self) -> List[SafetyEvent]:
        return self._event_history.copy()

    def get_config(self) -> SafetyConfig:
        return self._config

    def update_config(self, config: SafetyConfig):
        self._config = config
        # Update component configs
        if self._corner_detector:
            self._corner_detector._config = config
        if self._velocity_limiter:
            self._velocity_limiter._config = config
        if self._focus_monitor:
            self._focus_monitor._config = config
        if self._inactivity_timer:
            self._inactivity_timer._config = config


# Default Air Mouse safety configuration
DEFAULT_SAFETY_CONFIG = SafetyConfig(
    enable_corner_escape=True,
    corner_size=50,
    corner_hold_time=0.5,
    enable_global_hotkey=True,
    enable_focus_loss_pause=True,
    focus_check_interval=0.5,
    enable_velocity_limit=True,
    max_cursor_velocity=5000,
    max_acceleration=20000,
    enable_gesture_confirmation=True,
    min_gesture_duration=0.15,
    min_stable_frames=3,
    enable_inactivity_timeout=False,
    inactivity_timeout=300,
    release_all_buttons=True,
    reset_cursor_position=False,
    show_notification=True,
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def get_pos():
        return (100, 100)

    def get_size():
        return (1920, 1080)

    def on_release():
        print("Release all buttons!")

    def on_pause():
        print("Pause tracking!")

    def on_disable():
        print("Disable Air Mouse!")

    def on_notify(title, msg):
        print(f"NOTIFICATION: {title} - {msg}")

    def on_safety(event):
        print(f"Safety event: {event.trigger.value} -> {event.level.name}: {event.details}")

    config = DEFAULT_SAFETY_CONFIG
    config.enable_inactivity_timeout = True
    config.inactivity_timeout = 5  # 5 seconds for testing

    safety = SafetyManager(config)
    safety.set_callbacks(get_pos, get_size, on_release, on_pause, on_disable, on_notify)
    safety.add_callback(on_safety)
    safety.enable()

    print("Safety manager running. Move mouse to corner or wait 5 seconds...")
    print("Press Ctrl+C to stop")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        safety.disable()