"""
Auto-brightness configuration for Air Mouse.

Provides configurable parameters for ambient light sensor reading,
display brightness control, smoothing, and behavior.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BrightnessConfig:
    """Configuration for auto-brightness feature."""

    # Enable/disable auto-brightness
    enabled: bool = True

    # Brightness limits (percentage 0.0 - 1.0)
    min_brightness: float = 0.1      # 10% minimum
    max_brightness: float = 1.0      # 100% maximum

    # Sensor polling interval (seconds)
    poll_interval: float = 1.0       # 1 Hz default

    # Smoothing parameters
    smoothing_alpha: float = 0.15    # EMA smoothing factor (0-1, lower = smoother)

    # Hysteresis/deadband to prevent flickering (lux)
    hysteresis_lux: float = 10.0     # Minimum change to trigger adjustment

    # Ambient light to brightness mapping
    # lux at which brightness = min_brightness
    lux_at_min: float = 10.0         # Dark environment
    # lux at which brightness = max_brightness
    lux_at_max: float = 1000.0       # Bright environment

    # Hardware interfaces (auto-detected if None)
    preferred_backlight_path: Optional[str] = None
    preferred_als_path: Optional[str] = None

    # Behavior
    restore_on_exit: bool = True     # Restore original brightness when AirMouse stops
    require_als: bool = False        # If True, disable feature if no ALS found

    def __post_init__(self):
        """Validate configuration values."""
        self.min_brightness = max(0.0, min(1.0, self.min_brightness))
        self.max_brightness = max(0.0, min(1.0, self.max_brightness))
        if self.min_brightness > self.max_brightness:
            self.min_brightness, self.max_brightness = self.max_brightness, self.min_brightness

        self.poll_interval = max(0.1, self.poll_interval)
        self.smoothing_alpha = max(0.0, min(1.0, self.smoothing_alpha))
        self.hysteresis_lux = max(0.0, self.hysteresis_lux)

        if self.lux_at_min >= self.lux_at_max:
            self.lux_at_min = 10.0
            self.lux_at_max = 1000.0

    def lux_to_brightness(self, lux: float) -> float:
        """Convert ambient light (lux) to target brightness (0.0-1.0)."""
        if lux <= self.lux_at_min:
            return self.min_brightness
        if lux >= self.lux_at_max:
            return self.max_brightness

        # Linear interpolation between min/max
        t = (lux - self.lux_at_min) / (self.lux_at_max - self.lux_at_min)
        return self.min_brightness + t * (self.max_brightness - self.min_brightness)

    def brightness_to_raw(self, brightness: float, max_raw: int) -> int:
        """Convert brightness percentage (0.0-1.0) to raw backlight value."""
        return int(round(brightness * max_raw))

    def raw_to_brightness(self, raw: int, max_raw: int) -> float:
        """Convert raw backlight value to brightness percentage (0.0-1.0)."""
        if max_raw <= 0:
            return 0.0
        return max(0.0, min(1.0, raw / max_raw))