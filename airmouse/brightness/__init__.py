"""
Auto-brightness module for Air Mouse.

Provides automatic display brightness control based on ambient light sensor readings.
Only active while AirMouse is running.
"""

from .config import BrightnessConfig
from .ambient_light import (
    AmbientLightSensor,
    ALSDevice,
    ALSSource,
    detect_all_sensors
)
from .backlight import (
    BacklightController,
    BacklightDevice,
    BacklightType,
    detect_all_backlights
)
from .controller import (
    AutoBrightnessController,
    BrightnessState,
    create_auto_brightness_controller
)

__all__ = [
    'BrightnessConfig',
    'AmbientLightSensor',
    'ALSDevice',
    'ALSSource',
    'detect_all_sensors',
    'BacklightController',
    'BacklightDevice',
    'BacklightType',
    'detect_all_backlights',
    'AutoBrightnessController',
    'BrightnessState',
    'create_auto_brightness_controller',
]