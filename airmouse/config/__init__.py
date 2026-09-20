"""Air Mouse Configuration Module

Provides persistent, human-readable TOML configuration with:
- ConfigManager: load / save / validate / migrate TOML config files
- ProfileManager: create / edit / delete / import / export profiles
- Config: top-level configuration dataclass mirroring the product spec

The config is intentionally decoupled from the runtime so that the
background daemon, the CLI, and the settings UI all read the same source
of truth.
"""

from .config import (
    Config,
    CameraConfig,
    CursorConfig as ConfigCursorConfig,
    GestureConfig as ConfigGestureConfig,
    TrackingConfig as ConfigTrackingConfig,
    HotkeyConfig,
    PrivacyConfig,
    StartupConfig,
    DiagnosticsConfig,
    ConfigManager,
    DEFAULT_CONFIG,
    CONFIG_FILE_NAME,
    PROFILES_DIR_NAME,
)
from .profiles import ProfileManager, ProfileInfo

__all__ = [
    "Config",
    "CameraConfig",
    "ConfigCursorConfig",
    "ConfigGestureConfig",
    "ConfigTrackingConfig",
    "HotkeyConfig",
    "PrivacyConfig",
    "StartupConfig",
    "DiagnosticsConfig",
    "ConfigManager",
    "DEFAULT_CONFIG",
    "CONFIG_FILE_NAME",
    "PROFILES_DIR_NAME",
    "ProfileManager",
    "ProfileInfo",
]