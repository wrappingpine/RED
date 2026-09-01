"""
Ambient Light Sensor (ALS) detection and reading for Linux.

Supports multiple hardware interfaces:
- IIO (Industrial I/O) subsystem: /sys/bus/iio/devices/
- hwmon sensors: /sys/class/hwmon/
- Direct sensor files in /sys/
"""

import os
import glob
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ALSSource(Enum):
    """Source of ambient light sensor."""
    IIO = "iio"
    HWMON = "hwmon"
    UNKNOWN = "unknown"


@dataclass
class ALSDevice:
    """Represents a detected ambient light sensor device."""
    path: str
    source: ALSSource
    name: str
    max_lux: Optional[float] = None
    scale: float = 1.0  # Raw value multiplier to get lux


class AmbientLightSensor:
    """
    Detects and reads from ambient light sensors on Linux.

    Searches for sensors in order of preference:
    1. IIO devices with in_illuminance input
    2. hwmon devices with illuminance input
    3. Any other light sensor interfaces
    """

    def __init__(self):
        self._device: Optional[ALSDevice] = None
        self._last_lux: Optional[float] = None
        self._detect_sensor()

    def _detect_sensor(self) -> bool:
        """Detect available ambient light sensor."""
        # Priority 1: IIO devices
        device = self._find_iio_sensor()
        if device:
            self._device = device
            logger.info(f"Found ALS via IIO: {device.name} at {device.path}")
            return True

        # Priority 2: hwmon devices
        device = self._find_hwmon_sensor()
        if device:
            self._device = device
            logger.info(f"Found ALS via hwmon: {device.name} at {device.path}")
            return True

        # Priority 3: Generic search
        device = self._find_generic_sensor()
        if device:
            self._device = device
            logger.info(f"Found ALS via generic search: {device.name} at {device.path}")
            return True

        logger.warning("No ambient light sensor found")
        return False

    def _find_iio_sensor(self) -> Optional[ALSDevice]:
        """Find ALS in IIO subsystem."""
        iio_base = "/sys/bus/iio/devices"
        if not os.path.exists(iio_base):
            return None

        for device_dir in glob.glob(os.path.join(iio_base, "iio:device*")):
            # Check for illuminance input
            illuminance_files = glob.glob(os.path.join(device_dir, "in_illuminance*_input"))
            if not illuminance_files:
                # Also check for raw + scale
                raw_files = glob.glob(os.path.join(device_dir, "in_illuminance*_raw"))
                scale_files = glob.glob(os.path.join(device_dir, "in_illuminance*_scale"))
                if raw_files and scale_files:
                    illuminance_files = raw_files

            if illuminance_files:
                # Read name
                name_path = os.path.join(device_dir, "name")
                try:
                    with open(name_path, 'r') as f:
                        name = f.read().strip()
                except Exception:
                    name = os.path.basename(device_dir)

                # Get scale if available
                scale = 1.0
                scale_file = os.path.join(device_dir, "in_illuminance_scale")
                if os.path.exists(scale_file):
                    try:
                        with open(scale_file, 'r') as f:
                            scale = float(f.read().strip())
                    except Exception:
                        pass

                return ALSDevice(
                    path=illuminance_files[0],
                    source=ALSSource.IIO,
                    name=name,
                    scale=scale
                )

        return None

    def _find_hwmon_sensor(self) -> Optional[ALSDevice]:
        """Find ALS in hwmon subsystem."""
        hwmon_base = "/sys/class/hwmon"
        if not os.path.exists(hwmon_base):
            return None

        for hwmon_dir in glob.glob(os.path.join(hwmon_base, "hwmon*")):
            # Check for illuminance input
            illuminance_files = glob.glob(os.path.join(hwmon_dir, "*illuminance*_input"))
            if not illuminance_files:
                # Check for light sensor
                illuminance_files = glob.glob(os.path.join(hwmon_dir, "*light*_input"))

            if illuminance_files:
                name_path = os.path.join(hwmon_dir, "name")
                try:
                    with open(name_path, 'r') as f:
                        name = f.read().strip()
                except Exception:
                    name = os.path.basename(hwmon_dir)

                return ALSDevice(
                    path=illuminance_files[0],
                    source=ALSSource.HWMON,
                    name=name
                )

        return None

    def _find_generic_sensor(self) -> Optional[ALSDevice]:
        """Generic search for light sensors."""
        # Search common locations
        search_paths = [
            "/sys/devices/platform/*/iio:device*/in_illuminance*_input",
            "/sys/devices/virtual/hwmon/hwmon*/in_illuminance*_input",
            "/sys/bus/iio/devices/iio:device*/in_illuminance*_input",
        ]

        for pattern in search_paths:
            for path in glob.glob(pattern):
                if os.path.exists(path):
                    return ALSDevice(
                        path=path,
                        source=ALSSource.UNKNOWN,
                        name=os.path.basename(os.path.dirname(path))
                    )

        return None

    def read_lux(self) -> Optional[float]:
        """Read current ambient light level in lux."""
        if not self._device:
            return None

        try:
            with open(self._device.path, 'r') as f:
                raw = float(f.read().strip())
            lux = raw * self._device.scale
            self._last_lux = lux
            return lux
        except Exception as e:
            logger.debug(f"Failed to read ALS: {e}")
            return None

    def is_available(self) -> bool:
        """Check if sensor is available."""
        return self._device is not None

    def get_device_info(self) -> Optional[dict]:
        """Get sensor device information."""
        if not self._device:
            return None
        return {
            'path': self._device.path,
            'source': self._device.source.value,
            'name': self._device.name,
            'scale': self._device.scale
        }

    def get_last_reading(self) -> Optional[float]:
        """Get last successful reading."""
        return self._last_lux


def detect_all_sensors() -> List[ALSDevice]:
    """Detect all available ambient light sensors."""
    sensors = []

    # IIO sensors
    iio_base = "/sys/bus/iio/devices"
    if os.path.exists(iio_base):
        for device_dir in glob.glob(os.path.join(iio_base, "iio:device*")):
            illuminance_files = glob.glob(os.path.join(device_dir, "in_illuminance*_input"))
            if not illuminance_files:
                raw_files = glob.glob(os.path.join(device_dir, "in_illuminance*_raw"))
                scale_files = glob.glob(os.path.join(device_dir, "in_illuminance*_scale"))
                if raw_files and scale_files:
                    illuminance_files = raw_files

            if illuminance_files:
                name_path = os.path.join(device_dir, "name")
                try:
                    with open(name_path, 'r') as f:
                        name = f.read().strip()
                except Exception:
                    name = os.path.basename(device_dir)

                scale = 1.0
                scale_file = os.path.join(device_dir, "in_illuminance_scale")
                if os.path.exists(scale_file):
                    try:
                        with open(scale_file, 'r') as f:
                            scale = float(f.read().strip())
                    except Exception:
                        pass

                sensors.append(ALSDevice(
                    path=illuminance_files[0],
                    source=ALSSource.IIO,
                    name=name,
                    scale=scale
                ))

    # hwmon sensors
    hwmon_base = "/sys/class/hwmon"
    if os.path.exists(hwmon_base):
        for hwmon_dir in glob.glob(os.path.join(hwmon_base, "hwmon*")):
            illuminance_files = glob.glob(os.path.join(hwmon_dir, "*illuminance*_input"))
            if not illuminance_files:
                illuminance_files = glob.glob(os.path.join(hwmon_dir, "*light*_input"))

            if illuminance_files:
                name_path = os.path.join(hwmon_dir, "name")
                try:
                    with open(name_path, 'r') as f:
                        name = f.read().strip()
                except Exception:
                    name = os.path.basename(hwmon_dir)

                sensors.append(ALSDevice(
                    path=illuminance_files[0],
                    source=ALSSource.HWMON,
                    name=name
                ))

    return sensors