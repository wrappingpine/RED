"""
Auto-brightness controller for Air Mouse.

Manages the auto-brightness feature lifecycle:
- Starts/stops with AirMouse activation
- Polls ambient light sensor at configurable interval
- Applies smoothing and hysteresis to prevent flickering
- Controls backlight based on ambient light
- Restores original brightness on exit
"""

import time
import threading
import logging
from typing import Optional, Callable
from dataclasses import dataclass

from .config import BrightnessConfig
from .ambient_light import AmbientLightSensor, detect_all_sensors, ALSDevice
from .backlight import BacklightController, detect_all_backlights, BacklightDevice

logger = logging.getLogger(__name__)


@dataclass
class BrightnessState:
    """Current state of auto-brightness controller."""
    is_active: bool = False
    current_lux: Optional[float] = None
    smoothed_lux: Optional[float] = None
    target_brightness: Optional[float] = None
    current_brightness: Optional[float] = None
    last_adjustment_time: float = 0.0
    original_brightness: Optional[float] = None


class AutoBrightnessController:
    """
    Auto-brightness controller that runs as a background thread.

    Features:
    - Non-blocking background polling
    - EMA smoothing of ambient light readings
    - Hysteresis/deadband to prevent flickering
    - Gradual brightness transitions
    - Automatic start/stop with AirMouse
    - Original brightness restoration
    """

    def __init__(
        self,
        config: BrightnessConfig,
        on_state_change: Optional[Callable[[BrightnessState], None]] = None
    ):
        self._config = config
        self._on_state_change = on_state_change

        # Components
        self._als = AmbientLightSensor()
        self._backlight = BacklightController(config.preferred_backlight_path)

        # State
        self._state = BrightnessState()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Validate hardware
        self._als_available = self._als.is_available()
        self._backlight_available = self._backlight.is_available()

        if not self._als_available:
            logger.warning("No ambient light sensor available")
        if not self._backlight_available:
            logger.warning("Backlight control not available (no write permission)")

    @property
    def is_available(self) -> bool:
        """Check if auto-brightness can function."""
        return self._als_available and self._backlight_available

    @property
    def is_active(self) -> bool:
        """Check if auto-brightness is currently active."""
        return self._state.is_active

    @property
    def state(self) -> BrightnessState:
        """Get current state (thread-safe copy)."""
        return BrightnessState(
            is_active=self._state.is_active,
            current_lux=self._state.current_lux,
            smoothed_lux=self._state.smoothed_lux,
            target_brightness=self._state.target_brightness,
            current_brightness=self._state.current_brightness,
            last_adjustment_time=self._state.last_adjustment_time,
            original_brightness=self._state.original_brightness
        )

    def start(self) -> bool:
        """
        Start auto-brightness control.

        Returns:
            True if started successfully, False if hardware unavailable
        """
        if self._running:
            logger.debug("Auto-brightness already running")
            return True

        if not self._config.enabled:
            logger.info("Auto-brightness disabled in config")
            return False

        if not self._als_available:
            if self._config.require_als:
                logger.error("ALS required but not available")
                return False
            logger.warning("Starting without ALS - will use fixed brightness")

        if not self._backlight_available:
            logger.error("Backlight control not available")
            return False

        # Save original brightness
        if self._config.restore_on_exit:
            if not self._backlight.save_original_brightness():
                logger.warning("Failed to save original brightness")
            else:
                self._state.original_brightness = self._backlight.get_brightness()

        # Initialize state
        self._state.is_active = True
        self._state.current_brightness = self._backlight.get_brightness()

        # Initial sensor reading
        lux = self._als.read_lux()
        if lux is not None:
            self._state.current_lux = lux
            self._state.smoothed_lux = lux
            target = self._config.lux_to_brightness(lux)
            self._state.target_brightness = target
            # Apply immediately on start
            self._backlight.set_brightness(target)
            self._state.current_brightness = target
            self._state.last_adjustment_time = time.time()

        # Start background thread
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info("Auto-brightness started")
        self._notify_state_change()
        return True

    def stop(self) -> bool:
        """
        Stop auto-brightness control and restore original brightness.

        Returns:
            True if stopped successfully
        """
        if not self._running:
            return True

        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        # Restore original brightness
        if self._config.restore_on_exit and self._state.original_brightness is not None:
            self._backlight.set_brightness(self._state.original_brightness)
            logger.info(f"Restored brightness to {self._state.original_brightness:.1%}")

        self._state.is_active = False
        logger.info("Auto-brightness stopped")
        self._notify_state_change()
        return True

    def _run_loop(self):
        """Background polling loop."""
        while self._running and not self._stop_event.is_set():
            loop_start = time.time()

            try:
                self._poll_and_adjust()
            except Exception as e:
                logger.error(f"Error in brightness loop: {e}")

            # Sleep for remainder of poll interval
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, self._config.poll_interval - elapsed)
            self._stop_event.wait(sleep_time)

    def _poll_and_adjust(self):
        """Poll sensor and adjust brightness if needed."""
        # Read ambient light
        lux = self._als.read_lux()
        if lux is None:
            return

        self._state.current_lux = lux

        # Apply EMA smoothing
        if self._state.smoothed_lux is None:
            self._state.smoothed_lux = lux
        else:
            alpha = self._config.smoothing_alpha
            self._state.smoothed_lux = alpha * lux + (1 - alpha) * self._state.smoothed_lux

        # Calculate target brightness from smoothed lux
        target_brightness = self._config.lux_to_brightness(self._state.smoothed_lux)
        self._state.target_brightness = target_brightness

        # Check hysteresis - only adjust if change exceeds threshold
        current_brightness = self._backlight.get_brightness()
        if current_brightness is None:
            return

        self._state.current_brightness = current_brightness

        brightness_diff = abs(target_brightness - current_brightness)
        # Convert hysteresis from lux to brightness equivalent
        # Approximate: if lux range maps to brightness range
        lux_range = self._config.lux_at_max - self._config.lux_at_min
        brightness_range = self._config.max_brightness - self._config.min_brightness
        if lux_range > 0:
            hysteresis_brightness = (self._config.hysteresis_lux / lux_range) * brightness_range
        else:
            hysteresis_brightness = 0.02  # Default 2%

        if brightness_diff < hysteresis_brightness:
            return  # Change too small, skip

        # Apply new brightness
        if self._backlight.set_brightness(target_brightness):
            self._state.current_brightness = target_brightness
            self._state.last_adjustment_time = time.time()
            logger.debug(f"Brightness adjusted: {current_brightness:.1%} -> {target_brightness:.1%} (lux={lux:.0f}, smoothed={self._state.smoothed_lux:.0f})")
            self._notify_state_change()

    def _notify_state_change(self):
        """Notify state change callback."""
        if self._on_state_change:
            try:
                self._on_state_change(self.state)
            except Exception as e:
                logger.debug(f"State change callback error: {e}")

    def update_config(self, config: BrightnessConfig):
        """Update configuration at runtime."""
        self._config = config
        logger.info("Auto-brightness config updated")

    def force_update(self):
        """Force an immediate brightness update."""
        self._poll_and_adjust()

    def get_diagnostics(self) -> dict:
        """Get diagnostic information."""
        return {
            'available': self.is_available,
            'active': self._state.is_active,
            'als': {
                'available': self._als_available,
                'device': self._als.get_device_info(),
                'current_lux': self._state.current_lux,
                'smoothed_lux': self._state.smoothed_lux,
            },
            'backlight': {
                'available': self._backlight_available,
                'device': self._backlight.get_device_info(),
                'current_brightness': self._state.current_brightness,
                'target_brightness': self._state.target_brightness,
            },
            'config': {
                'enabled': self._config.enabled,
                'min_brightness': self._config.min_brightness,
                'max_brightness': self._config.max_brightness,
                'poll_interval': self._config.poll_interval,
                'smoothing_alpha': self._config.smoothing_alpha,
                'hysteresis_lux': self._config.hysteresis_lux,
                'lux_at_min': self._config.lux_at_min,
                'lux_at_max': self._config.lux_at_max,
            }
        }


def create_auto_brightness_controller(
    config: Optional[BrightnessConfig] = None,
    on_state_change: Optional[Callable[[BrightnessState], None]] = None
) -> AutoBrightnessController:
    """Factory function to create auto-brightness controller."""
    if config is None:
        config = BrightnessConfig()
    return AutoBrightnessController(config, on_state_change)