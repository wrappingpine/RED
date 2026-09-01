"""
Backlight control for Linux systems.

Interfaces with /sys/class/backlight/ to read and write display brightness.
Supports multiple backlight devices and permission handling.
"""

import os
import glob
import logging
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class BacklightType(Enum):
    """Type of backlight interface."""
    AMDGPU = "amdgpu"
    INTEL = "intel"
    NVIDIA = "nvidia"
    ACPI = "acpi"
    GENERIC = "generic"
    UNKNOWN = "unknown"


@dataclass
class BacklightDevice:
    """Represents a backlight control device."""
    path: str
    name: str
    type: BacklightType
    max_brightness: int
    current_brightness: int
    brightness_path: str
    max_brightness_path: str
    actual_brightness_path: Optional[str] = None


class BacklightController:
    """
    Controls display backlight via /sys/class/backlight/.

    Supports reading/writing brightness, detecting available devices,
    and handling permission issues gracefully.
    """

    def __init__(self, preferred_path: Optional[str] = None):
        self._device: Optional[BacklightDevice] = None
        self._original_brightness: Optional[int] = None
        self._has_write_permission: bool = False
        self._detect_backlight(preferred_path)

    def _detect_backlight(self, preferred_path: Optional[str] = None) -> bool:
        """Detect available backlight device."""
        backlight_base = "/sys/class/backlight"
        if not os.path.exists(backlight_base):
            logger.warning("No backlight class found at /sys/class/backlight")
            return False

        devices = []
        for device_dir in glob.glob(os.path.join(backlight_base, "*")):
            device = self._probe_device(device_dir)
            if device:
                devices.append(device)

        if not devices:
            logger.warning("No usable backlight devices found")
            return False

        # Prefer specific device if requested
        if preferred_path:
            for dev in devices:
                if dev.path == preferred_path or dev.name == preferred_path:
                    self._device = dev
                    break

        # Otherwise prefer amdgpu > intel > nvidia > acpi > first available
        if not self._device:
            priority_order = [
                BacklightType.AMDGPU,
                BacklightType.INTEL,
                BacklightType.NVIDIA,
                BacklightType.ACPI,
                BacklightType.GENERIC,
            ]
            for btype in priority_order:
                for dev in devices:
                    if dev.type == btype:
                        self._device = dev
                        break
                if self._device:
                    break

        # Fallback to first device
        if not self._device and devices:
            self._device = devices[0]

        if self._device:
            logger.info(f"Using backlight: {self._device.name} ({self._device.type.value}) at {self._device.path}")
            self._check_permissions()
            return True

        return False

    def _probe_device(self, device_dir: str) -> Optional[BacklightDevice]:
        """Probe a backlight device directory."""
        brightness_path = os.path.join(device_dir, "brightness")
        max_brightness_path = os.path.join(device_dir, "max_brightness")
        actual_brightness_path = os.path.join(device_dir, "actual_brightness")

        if not os.path.exists(brightness_path) or not os.path.exists(max_brightness_path):
            return None

        try:
            with open(max_brightness_path, 'r') as f:
                max_brightness = int(f.read().strip())
            with open(brightness_path, 'r') as f:
                current_brightness = int(f.read().strip())
        except Exception as e:
            logger.debug(f"Failed to read backlight {device_dir}: {e}")
            return None

        if max_brightness <= 0:
            return None

        name = os.path.basename(device_dir)
        btype = self._classify_backlight(name)

        return BacklightDevice(
            path=device_dir,
            name=name,
            type=btype,
            max_brightness=max_brightness,
            current_brightness=current_brightness,
            brightness_path=brightness_path,
            max_brightness_path=max_brightness_path,
            actual_brightness_path=actual_brightness_path if os.path.exists(actual_brightness_path) else None
        )

    def _classify_backlight(self, name: str) -> BacklightType:
        """Classify backlight type from name."""
        name_lower = name.lower()
        if 'amdgpu' in name_lower or 'amd' in name_lower:
            return BacklightType.AMDGPU
        elif 'intel' in name_lower:
            return BacklightType.INTEL
        elif 'nvidia' in name_lower or 'nv' in name_lower:
            return BacklightType.NVIDIA
        elif 'acpi' in name_lower:
            return BacklightType.ACPI
        else:
            return BacklightType.GENERIC

    def _check_permissions(self) -> bool:
        """Check if we have write permission to brightness file."""
        if not self._device:
            return False

        try:
            # Test write by writing current value
            with open(self._device.brightness_path, 'w') as f:
                f.write(str(self._device.current_brightness))
            self._has_write_permission = True
            return True
        except PermissionError:
            logger.warning(f"No write permission for {self._device.brightness_path}")
            self._has_write_permission = False
            return False
        except Exception as e:
            logger.debug(f"Permission check failed: {e}")
            self._has_write_permission = False
            return False

    def get_brightness(self) -> Optional[float]:
        """Get current brightness as percentage (0.0-1.0)."""
        if not self._device:
            return None

        try:
            if self._device.actual_brightness_path:
                with open(self._device.actual_brightness_path, 'r') as f:
                    current = int(f.read().strip())
            else:
                with open(self._device.brightness_path, 'r') as f:
                    current = int(f.read().strip())
            return current / self._device.max_brightness
        except Exception as e:
            logger.debug(f"Failed to read brightness: {e}")
            return None

    def get_raw_brightness(self) -> Optional[int]:
        """Get current raw brightness value."""
        if not self._device:
            return None

        try:
            if self._device.actual_brightness_path:
                with open(self._device.actual_brightness_path, 'r') as f:
                    return int(f.read().strip())
            else:
                with open(self._device.brightness_path, 'r') as f:
                    return int(f.read().strip())
        except Exception as e:
            logger.debug(f"Failed to read raw brightness: {e}")
            return None

    def set_brightness(self, brightness: float) -> bool:
        """
        Set brightness as percentage (0.0-1.0).

        Args:
            brightness: Target brightness 0.0-1.0

        Returns:
            True if successful, False otherwise
        """
        if not self._device or not self._has_write_permission:
            return False

        brightness = max(0.0, min(1.0, brightness))
        raw_value = int(round(brightness * self._device.max_brightness))
        raw_value = max(0, min(self._device.max_brightness, raw_value))

        try:
            with open(self._device.brightness_path, 'w') as f:
                f.write(str(raw_value))
            self._device.current_brightness = raw_value
            return True
        except Exception as e:
            logger.error(f"Failed to set brightness: {e}")
            return False

    def set_raw_brightness(self, raw_value: int) -> bool:
        """Set raw brightness value."""
        if not self._device or not self._has_write_permission:
            return False

        raw_value = max(0, min(self._device.max_brightness, raw_value))

        try:
            with open(self._device.brightness_path, 'w') as f:
                f.write(str(raw_value))
            self._device.current_brightness = raw_value
            return True
        except Exception as e:
            logger.error(f"Failed to set raw brightness: {e}")
            return False

    def save_original_brightness(self) -> bool:
        """Save current brightness as original for later restoration."""
        current = self.get_raw_brightness()
        if current is not None:
            self._original_brightness = current
            logger.info(f"Saved original brightness: {current}/{self._device.max_brightness}")
            return True
        return False

    def restore_original_brightness(self) -> bool:
        """Restore the original brightness saved at start."""
        if self._original_brightness is not None and self._device and self._has_write_permission:
            try:
                with open(self._device.brightness_path, 'w') as f:
                    f.write(str(self._original_brightness))
                logger.info(f"Restored original brightness: {self._original_brightness}")
                return True
            except Exception as e:
                logger.error(f"Failed to restore brightness: {e}")
        return False

    def is_available(self) -> bool:
        """Check if backlight control is available."""
        return self._device is not None and self._has_write_permission

    def get_device_info(self) -> Optional[dict]:
        """Get backlight device information."""
        if not self._device:
            return None
        return {
            'path': self._device.path,
            'name': self._device.name,
            'type': self._device.type.value,
            'max_brightness': self._device.max_brightness,
            'current_brightness': self._device.current_brightness,
            'brightness_percent': self.get_brightness(),
            'has_write_permission': self._has_write_permission
        }

    def get_max_brightness(self) -> int:
        """Get maximum raw brightness value."""
        return self._device.max_brightness if self._device else 0


def detect_all_backlights() -> List[BacklightDevice]:
    """Detect all available backlight devices."""
    devices = []
    backlight_base = "/sys/class/backlight"
    if not os.path.exists(backlight_base):
        return devices

    for device_dir in glob.glob(os.path.join(backlight_base, "*")):
        device = BacklightController._probe_device_static(device_dir)
        if device:
            devices.append(device)

    return devices


# Static method for detection without instance
BacklightController._probe_device_static = staticmethod(BacklightController._probe_device)