"""Air Mouse Config Data Models and Manager

Provides robust, typed configuration mapping directly to the TOML schema:
- CameraConfig
- CursorConfig
- GestureConfig
- TrackingConfig
- HotkeyConfig
- PrivacyConfig
- StartupConfig
- DiagnosticsConfig
- Config (top-level config container)
- ConfigManager: handles I/O, validation, paths, and migrations.
"""

import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Union

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "config.toml"
PROFILES_DIR_NAME = "profiles"


@dataclass
class CameraConfig:
    device_index: int = 0
    device_path: str = ""
    width: int = 1280
    height: int = 720
    fps: int = 30
    exposure: int = -1
    gain: int = -1
    brightness: int = -1


@dataclass
class CursorConfig:
    sensitivity: float = 0.8
    smoothing: str = "adaptive"  # adaptive (one euro), ema, none
    acceleration: bool = True
    dead_zone: float = 0.02
    invert_x: bool = False
    invert_y: bool = False


@dataclass
class GestureConfig:
    left_click: str = "pinch"
    right_click: str = "custom"
    drag: str = "pinch_hold"
    scroll: str = "two_finger"
    pinch_enter_threshold: float = 0.045
    pinch_confirm_threshold: float = 0.040
    pinch_release_threshold: float = 0.070
    fist_hold_time: float = 0.5
    drag_hold_time: float = 0.2


@dataclass
class TrackingConfig:
    confidence_threshold: float = 0.75
    min_face_confidence: float = 0.5
    use_head_relative: bool = True
    virtual_plane_distance: float = 0.30  # positive Z in head coords (30cm in front)
    virtual_plane_width: float = 0.40
    virtual_plane_height: float = 0.25


@dataclass
class HotkeyConfig:
    emergency_stop: str = "Super+Alt+A"
    pause_resume: str = "Super+Alt+P"
    debug_overlay: str = "Ctrl+Shift+G"


@dataclass
class PrivacyConfig:
    local_only: bool = True
    telemetry_enabled: bool = False
    allow_remote_camera: bool = False


@dataclass
class StartupConfig:
    start_on_boot: bool = False
    background_mode: bool = True


@dataclass
class DiagnosticsConfig:
    log_level: str = "INFO"
    log_file: str = ""
    save_observations: bool = False


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    startup: StartupConfig = field(default_factory=StartupConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


DEFAULT_CONFIG = Config()


def get_default_config() -> Config:
    """Return a fresh default config instance (avoids mutation of module-level DEFAULT_CONFIG)."""
    return Config()


class ConfigManager:
    """Manages application config with robust validation and profiles support."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        if config_dir is None:
            # Match standard XDG spec and user home
            self.config_dir = Path.home() / ".config" / "airmouse"
        else:
            self.config_dir = Path(config_dir)

        self.config_file = self.config_dir / CONFIG_FILE_NAME
        self.profiles_dir = self.config_dir / PROFILES_DIR_NAME

    def ensure_directories(self):
        """Create necessary config directories."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def get_config_path(self) -> Path:
        """Get path to the active config file."""
        return self.config_file

    def get_profiles_dir(self) -> Path:
        """Get path to profiles directory."""
        return self.profiles_dir

    def load(self) -> Config:
        """Load and validate the config file, fallback to defaults."""
        self.ensure_directories()
        if not self.config_file.exists():
            logger.info(f"Config file not found, creating default at {self.config_file}")
            default_cfg = get_default_config()
            self.save(default_cfg)
            return default_cfg

        try:
            with open(self.config_file, "rb") as f:
                data = tomllib.load(f)

            # Map the nested dictionary back to strongly typed classes with safe fallbacks
            return self.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading config file {self.config_file}: {e}. Resetting to default.")
            return get_default_config()

    def save(self, config: Config) -> bool:
        """Save configuration back to persistent TOML file."""
        self.ensure_directories()
        try:
            data = self.to_dict(config)
            with open(self.config_file, "wb") as f:
                tomli_w.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Error saving config file {self.config_file}: {e}")
            return False

    def validate(self, config: Config) -> bool:
        """Check the configuration parameters for validity according to product spec bounds."""
        try:
            # Check camera boundaries
            if config.camera.device_index < 0:
                return False
            if config.camera.width <= 0 or config.camera.height <= 0:
                return False
            if config.camera.fps not in [15, 30, 60, 90, 120]:
                return False

            # Check cursor parameters
            if not (0.0 <= config.cursor.sensitivity <= 5.0):
                return False
            if config.cursor.smoothing not in ["adaptive", "ema", "none"]:
                return False
            if not (0.0 <= config.cursor.dead_zone <= 0.5):
                return False

            # Check thresholds
            if not (0.0 < config.gestures.pinch_enter_threshold < 1.0):
                return False
            if not (0.0 < config.gestures.pinch_confirm_threshold < 1.0):
                return False
            if not (0.0 < config.gestures.pinch_release_threshold < 1.0):
                return False
            # Verify hysteresis order: pinch enter should be looser than confirm, and release looser than enter
            # enter < release and confirm <= enter
            if config.gestures.pinch_confirm_threshold > config.gestures.pinch_enter_threshold:
                return False
            if config.gestures.pinch_enter_threshold > config.gestures.pinch_release_threshold:
                return False

            if config.gestures.fist_hold_time < 0.1 or config.gestures.drag_hold_time < 0.1:
                return False

            # Check tracking
            if not (0.0 <= config.tracking.confidence_threshold <= 1.0):
                return False
            if config.tracking.virtual_plane_distance <= 0.05:
                return False

            return True
        except Exception as e:
            logger.error(f"Exception during config validation: {e}")
            return False

    def to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert Config dataclass to dictionary."""
        return asdict(config)

    def from_dict(self, data: Dict[str, Any]) -> Config:
        """Create a Config object from raw dictionary safely."""
        conf = Config()

        def update_dataclass(target, updates):
            if not updates or not isinstance(updates, dict):
                return
            for key, val in updates.items():
                if hasattr(target, key):
                    setattr(target, key, val)

        update_dataclass(conf.camera, data.get("camera"))
        update_dataclass(conf.cursor, data.get("cursor"))
        update_dataclass(conf.gestures, data.get("gestures"))
        update_dataclass(conf.tracking, data.get("tracking"))
        update_dataclass(conf.hotkeys, data.get("hotkeys"))
        update_dataclass(conf.privacy, data.get("privacy"))
        update_dataclass(conf.startup, data.get("startup"))
        update_dataclass(conf.diagnostics, data.get("diagnostics"))

        return conf
