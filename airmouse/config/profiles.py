"""
Application Profiles System for AirMouse (§36-38).

Handles creation, editing, deletion, import and export of named profiles.
"""

import logging
import shutil
import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import fnmatch

logger = logging.getLogger(__name__)


class ProfileSource(Enum):
    """Source of profile configuration."""
    GLOBAL = "global"
    APP_SPECIFIC = "app_specific"
    USER_OVERRIDE = "user_override"
    AUTO_DETECTED = "auto_detected"


@dataclass
class ProfileConfig:
    """Configuration for a single application profile."""
    name: str
    description: str = ""
    
    # Cursor modifications
    cursor_sensitivity_mode: Optional[str] = None  # "precision", "normal", "fast"
    cursor_base_sensitivity: Optional[float] = None
    cursor_acceleration: Optional[float] = None
    cursor_smoothing: Optional[str] = None
    cursor_dead_zone_radius: Optional[float] = None
    cursor_invert_x: Optional[bool] = None
    cursor_invert_y: Optional[bool] = None
    cursor_use_index_tip: Optional[bool] = None
    cursor_max_velocity: Optional[int] = None
    cursor_max_velocity_precision: Optional[int] = None
    
    # Gesture modifications
    gesture_pinch_enter_threshold: Optional[float] = None
    gesture_pinch_confirm_threshold: Optional[float] = None
    gesture_pinch_release_threshold: Optional[float] = None
    gesture_scroll_sensitivity: Optional[float] = None
    gesture_scroll_cooldown: Optional[float] = None
    gesture_fist_hold_time: Optional[float] = None
    gesture_drag_hold_time: Optional[float] = None
    gesture_click_max_duration: Optional[float] = None
    gesture_click_max_movement: Optional[float] = None
    gesture_gesture_cooldown: Optional[float] = None
    gesture_enable_two_hand: Optional[bool] = None
    gesture_clutch_enabled: Optional[bool] = None
    gesture_clutch_trigger_gesture: Optional[str] = None
    gesture_clutch_timeout: Optional[float] = None
    gesture_conflict_resolution: Optional[bool] = None
    
    # Input mapping modifications
    input_left_click_action: Optional[str] = None  # "left_click", "right_click", "middle_click", "custom"
    input_right_click_action: Optional[str] = None
    input_middle_click_action: Optional[str] = None
    input_scroll_up_action: Optional[str] = None
    input_scroll_down_action: Optional[str] = None
    input_scroll_horizontal_action: Optional[str] = None
    input_drag_action: Optional[str] = None
    input_pinch_confirm_action: Optional[str] = None
    input_open_palm_action: Optional[str] = None
    input_fist_action: Optional[str] = None
    input_thumb_gesture_action: Optional[str] = None
    input_two_hand_action: Optional[str] = None
    
    # Custom actions (app-specific shortcuts)
    custom_actions: Dict[str, str] = field(default_factory=dict)
    
    # Profile metadata
    source: ProfileSource = ProfileSource.APP_SPECIFIC
    priority: int = 50  # Higher = more specific
    enabled: bool = True
    auto_detect_patterns: List[str] = field(default_factory=list)  # Window title/class patterns
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['source'] = self.source.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfileConfig":
        """Create from dictionary."""
        data = data.copy()
        if 'source' in data and isinstance(data['source'], str):
            data['source'] = ProfileSource(data['source'])
        return cls(**data)


# Built-in default profiles
DEFAULT_PROFILES = {
    "global": ProfileConfig(
        name="global",
        description="Default global profile - applies to all applications",
        source=ProfileSource.GLOBAL,
        priority=0,
    ),
    "chrome": ProfileConfig(
        name="chrome",
        description="Optimized for web browsing",
        cursor_sensitivity_mode="normal",
        cursor_acceleration=1.1,
        gesture_scroll_sensitivity=0.003,
        gesture_drag_hold_time=0.15,
        auto_detect_patterns=["*Chrome*", "*Chromium*", "*Firefox*", "*Browser*"],
        source=ProfileSource.APP_SPECIFIC,
        priority=100,
    ),
    "code_editor": ProfileConfig(
        name="code_editor",
        description="Optimized for code editors (VS Code, Vim, etc.)",
        cursor_sensitivity_mode="precision",
        cursor_acceleration=1.0,
        cursor_dead_zone_radius=0.015,
        gesture_pinch_enter_threshold=0.04,
        gesture_drag_hold_time=0.25,
        auto_detect_patterns=["*VS Code*", "*Vim*", "*Neovim*", "*Sublime*", "*IntelliJ*", "*PyCharm*", "*Code*"],
        source=ProfileSource.APP_SPECIFIC,
        priority=100,
    ),
    "blender": ProfileConfig(
        name="blender",
        description="Optimized for Blender 3D",
        cursor_sensitivity_mode="fast",
        cursor_acceleration=1.5,
        gesture_scroll_sensitivity=0.004,
        gesture_enable_two_hand=True,
        # Map gestures to Blender actions
        input_left_click_action="select",
        input_right_click_action="context_menu",
        input_middle_click_action="pan",
        input_scroll_up_action="zoom_in",
        input_scroll_down_action="zoom_out",
        input_drag_action="drag_select",
        auto_detect_patterns=["*Blender*"],
        source=ProfileSource.APP_SPECIFIC,
        priority=100,
    ),
    "kicad": ProfileConfig(
        name="kicad",
        description="Optimized for KiCad PCB design",
        cursor_sensitivity_mode="precision",
        cursor_acceleration=1.0,
        cursor_dead_zone_radius=0.01,
        gesture_pinch_enter_threshold=0.035,
        gesture_drag_hold_time=0.3,
        auto_detect_patterns=["*KiCad*", "*pcbnew*", "*eeschema*"],
        source=ProfileSource.APP_SPECIFIC,
        priority=100,
    ),
    "media": ProfileConfig(
        name="media",
        description="Optimized for media players",
        cursor_sensitivity_mode="normal",
        gesture_scroll_sensitivity=0.005,
        gesture_drag_hold_time=0.5,
        # Map gestures to media controls
        input_left_click_action="play_pause",
        input_right_click_action="fullscreen",
        input_scroll_up_action="volume_up",
        input_scroll_down_action="volume_down",
        input_scroll_horizontal_action="seek",
        auto_detect_patterns=["*VLC*", "*MPV*", "*Spotify*", "*YouTube*", "*Netflix*", "*Plex*"],
        source=ProfileSource.APP_SPECIFIC,
        priority=100,
    ),
    "presentation": ProfileConfig(
        name="presentation",
        description="Optimized for presentations",
        cursor_sensitivity_mode="precision",
        gesture_drag_hold_time=0.5,
        gesture_fist_hold_time=1.0,
        # Map gestures to presentation controls
        input_left_click_action="next_slide",
        input_right_click_action="previous_slide",
        input_open_palm_action="pause_presentation",
        input_thumb_gesture_action="laser_pointer",
        auto_detect_patterns=["*PowerPoint*", "*LibreOffice Impress*", "*Keynote*", "*Slides*"],
        source=ProfileSource.APP_SPECIFIC,
        priority=100,
    ),
    "gaming": ProfileConfig(
        name="gaming",
        description="Optimized for gaming (mouse-like control)",
        cursor_sensitivity_mode="fast",
        cursor_acceleration=2.0,
        cursor_max_velocity=5000,
        cursor_max_velocity_precision=1000,
        gesture_pinch_enter_threshold=0.04,
        gesture_drag_hold_time=0.1,
        gesture_enable_two_hand=False,
        auto_detect_patterns=["*Steam*", "*Lutris*", "*Heroic*", "*Game*"],
        source=ProfileSource.APP_SPECIFIC,
        priority=100,
    ),
}


class ProfileManager:
    """
    Manages application profiles with auto-detection and manual switching.
    
    Per §36-38: Universal input + optional application profiles.
    Core system never depends on profile detection.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        # Support both old API (ConfigManager) and new API (Path)
        if config_dir is None:
            from ..config import ConfigManager
            config_dir = ConfigManager().profiles_dir
        elif hasattr(config_dir, 'profiles_dir'):
            # Old API: ConfigManager object
            self.config_manager = config_dir
            self.config_dir = config_dir.profiles_dir
        else:
            # New API: Path object
            self.config_manager = None
            self.config_dir = config_dir
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._profiles: Dict[str, ProfileConfig] = {}
        self._current_profile: Optional[ProfileConfig] = None
        self._current_app_name: str = "unknown"
        self._detection_callback: Optional[Callable[[], str]] = None
        self._profile_change_callback: Optional[Callable[[ProfileConfig], None]] = None
        
        # Load profiles
        self._load_default_profiles()
        self._load_user_profiles()
        
        # Start with global profile
        self._current_profile = self._profiles.get("global")
        if not self._current_profile:
            self._current_profile = DEFAULT_PROFILES["global"]
    
    def get_profiles_dir(self) -> Path:
        """Return path to profiles directory."""
        return self.config_dir

    def _load_default_profiles(self):
        """Load built-in default profiles."""
        for name, profile in DEFAULT_PROFILES.items():
            if name not in self._profiles:
                self._profiles[name] = profile
    
    def _load_user_profiles(self):
        """Load user-defined profiles from config directory."""
        for profile_file in self.config_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                profile = ProfileConfig.from_dict(data)
                self._profiles[profile.name] = profile
                logger.info(f"Loaded user profile: {profile.name}")
            except Exception as e:
                logger.warning(f"Failed to load profile {profile_file}: {e}")
    
    def get_profiles_dir(self) -> Path:
        """Return path to profiles directory."""
        return self.config_dir

    def list_profiles(self):
        """List all saved profiles."""
        profiles = []
        for name, profile in self._profiles.items():
            # Create ProfileInfo for backward compatibility
            profiles.append(type('ProfileInfo', (), {
                'name': name,
                'source_config_path': str(self.config_dir / f"{name}.json"),
                'created_at': '',
                'description': profile.description
            })())
        return profiles

    def create(self, name: str, description: str = ""):
        """Create a new profile named <name> with current config values."""
        filename = self.config_dir / f"{name}.json"
        if filename.exists():
            logger.warning(f"Profile '{name}' already exists.")
            return None

        # Create profile with default values
        profile = ProfileConfig(name=name, description=description)
        self._profiles[name] = profile
        
        # Save to file
        self.save_profile(profile)
        
        return filename

    def save_profile(self, profile: ProfileConfig):
        """Save a profile to config directory."""
        profile_file = self.config_dir / f"{profile.name}.json"
        try:
            with open(profile_file, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            logger.info(f"Saved profile: {profile.name}")
        except Exception as e:
            logger.error(f"Failed to save profile {profile.name}: {e}")

    def load_profile(self, name: str):
        """Load a profile returns ProfileConfig."""
        profile = self._profiles.get(name)
        if profile is None:
            # Try to load from file
            filename = self.config_dir / f"{name}.json"
            if filename.exists():
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    profile = ProfileConfig.from_dict(data)
                    self._profiles[name] = profile
                except Exception as e:
                    logger.error(f"Failed to load profile {name}: {e}")
                    return None
            else:
                return None
        return profile

    def export_profile(self, name: str, export_path: Union[str, Path]):
        """Export profile to external path."""
        profile = self._profiles.get(name)
        if profile is None:
            logger.warning(f"Profile '{name}' not found for export.")
            return False
        
        try:
            export_path = Path(export_path)
            with open(export_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            logger.info(f"Exported profile '{name}' to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export profile {name}: {e}")
            return False

    def import_profile(self, import_path: Union[str, Path]):
        """Import profile from external path."""
        try:
            import_path = Path(import_path)
            with open(import_path, 'r') as f:
                data = json.load(f)
            profile = ProfileConfig.from_dict(data)
            
            # Avoid overwriting existing profiles
            if profile.name in self._profiles:
                # Add suffix to make unique
                base_name = profile.name
                counter = 1
                while f"{base_name}_{counter}" in self._profiles:
                    counter += 1
                profile.name = f"{base_name}_{counter}"
            
            self._profiles[profile.name] = profile
            self.save_profile(profile)
            
            return self.config_dir / f"{profile.name}.json"
        except Exception as e:
            logger.error(f"Failed to import profile from {import_path}: {e}")
            return None

    def delete(self, name: str):
        """Delete a profile."""
        if name in DEFAULT_PROFILES:
            logger.warning(f"Cannot delete built-in profile: {name}")
            return False
        
        profile_file = self.config_dir / f"{name}.json"
        if profile_file.exists():
            profile_file.unlink()
            self._profiles.pop(name, None)
            logger.info(f"Deleted profile: {name}")
            return True
        else:
            logger.warning(f"Profile '{name}' not found for deletion.")
            return False

    def get_profile(self, name: str) -> Optional[ProfileConfig]:
        """Get a profile by name."""
        return self._profiles.get(name)
    
    def list_profiles_new(self) -> List[ProfileConfig]:
        """List all available profiles (new API)."""
        return list(self._profiles.values())
    
    def set_detection_callback(self, callback):
        """Set callback for application detection."""
        self._detection_callback = callback
    
    def set_profile_change_callback(self, callback):
        """Set callback when profile changes."""
        self._profile_change_callback = callback
    
    def detect_active_app(self) -> str:
        """Detect the currently active application."""
        if self._detection_callback:
            try:
                return self._detection_callback()
            except Exception as e:
                logger.warning(f"App detection callback failed: {e}")
        return "unknown"
    
    def auto_switch_profile(self):
        """Automatically switch profile based on active application."""
        app_name = self.detect_active_app()
        self._current_app_name = app_name
        
        # Find matching profile
        best_match = None
        best_priority = -1
        
        for profile in self._profiles.values():
            if not profile.enabled:
                continue
            if profile.source == ProfileSource.GLOBAL:
                continue  # Global is fallback
            
            for pattern in profile.auto_detect_patterns:
                if fnmatch.fnmatch(app_name, pattern):
                    if profile.priority > best_priority:
                        best_match = profile
                        best_priority = profile.priority
        
        if best_match and best_match != self._current_profile:
            self._switch_to_profile(best_match)
        elif not best_match and self._current_profile != self._profiles.get("global"):
            # Fall back to global
            self._switch_to_profile(self._profiles.get("global"))
    
    def _switch_to_profile(self, profile: ProfileConfig):
        """Switch to a new profile."""
        if profile == self._current_profile:
            return
        
        logger.info(f"Switching profile: {self._current_profile.name} -> {profile.name}")
        self._current_profile = profile
        
        if self._profile_change_callback:
            try:
                self._profile_change_callback(profile)
            except Exception as e:
                logger.error(f"Profile change callback failed: {e}")
    
    def manual_switch(self, name: str) -> bool:
        """Manually switch to a profile by name."""
        profile = self._profiles.get(name)
        if not profile:
            logger.warning(f"Profile not found: {name}")
            return False
        self._switch_to_profile(profile)
        return True
    
    def get_current_profile(self) -> ProfileConfig:
        """Get the currently active profile."""
        return self._current_profile
    
    def get_current_app_name(self) -> str:
        """Get the currently detected application name."""
        return self._current_app_name
    
    def apply_to_gesture_config(self, gesture_config):
        """Apply current profile to a GestureConfig object."""
        if not self._current_profile:
            return gesture_config
        
        profile = self._current_profile
        # Apply gesture modifications
        if profile.gesture_pinch_enter_threshold is not None:
            gesture_config.pinch_enter_threshold = profile.gesture_pinch_enter_threshold
        if profile.gesture_pinch_confirm_threshold is not None:
            gesture_config.pinch_confirm_threshold = profile.gesture_pinch_confirm_threshold
        if profile.gesture_pinch_release_threshold is not None:
            gesture_config.pinch_release_threshold = profile.gesture_pinch_release_threshold
        if profile.gesture_scroll_sensitivity is not None:
            gesture_config.scroll_sensitivity = profile.gesture_scroll_sensitivity
        if profile.gesture_scroll_cooldown is not None:
            gesture_config.scroll_cooldown = profile.gesture_scroll_cooldown
        if profile.gesture_fist_hold_time is not None:
            gesture_config.fist_hold_time = profile.gesture_fist_hold_time
        if profile.gesture_drag_hold_time is not None:
            gesture_config.drag_hold_time = profile.gesture_drag_hold_time
        if profile.gesture_click_max_duration is not None:
            gesture_config.click_max_duration = profile.gesture_click_max_duration
        if profile.gesture_click_max_movement is not None:
            gesture_config.click_max_movement = profile.gesture_click_max_movement
        if profile.gesture_gesture_cooldown is not None:
            gesture_config.gesture_cooldown = profile.gesture_gesture_cooldown
        if profile.gesture_enable_two_hand is not None:
            gesture_config.enable_two_hand = profile.gesture_enable_two_hand
        if profile.gesture_clutch_enabled is not None:
            gesture_config.clutch_enabled = profile.gesture_clutch_enabled
        if profile.gesture_clutch_trigger_gesture is not None:
            gesture_config.clutch_trigger_gesture = profile.gesture_clutch_trigger_gesture
        if profile.gesture_clutch_timeout is not None:
            gesture_config.clutch_timeout = profile.gesture_clutch_timeout
        if profile.gesture_conflict_resolution is not None:
            gesture_config.conflict_resolution = profile.gesture_conflict_resolution
        
        return gesture_config
    
    def apply_to_cursor_config(self, cursor_config):
        """Apply current profile to a CursorConfig object."""
        if not self._current_profile:
            return cursor_config
        
        profile = self._current_profile
        # Apply cursor modifications
        if profile.cursor_sensitivity_mode is not None:
            from ..control.cursor import SensitivityMode
            cursor_config.sensitivity_mode = SensitivityMode(profile.cursor_sensitivity_mode)
        if profile.cursor_base_sensitivity is not None:
            cursor_config.base_sensitivity = profile.cursor_base_sensitivity
        if profile.cursor_acceleration is not None:
            cursor_config.acceleration = profile.cursor_acceleration
        if profile.cursor_smoothing is not None:
            from ..control.cursor import SmoothingAlgorithm
            cursor_config.smoothing = SmoothingAlgorithm(profile.cursor_smoothing)
        if profile.cursor_dead_zone_radius is not None:
            cursor_config.dead_zone_radius = profile.cursor_dead_zone_radius
        if profile.cursor_invert_x is not None:
            cursor_config.invert_x = profile.cursor_invert_x
        if profile.cursor_invert_y is not None:
            cursor_config.invert_y = profile.cursor_invert_y
        if profile.cursor_use_index_tip is not None:
            cursor_config.use_index_tip = profile.cursor_use_index_tip
        if profile.cursor_max_velocity is not None:
            cursor_config.max_velocity = profile.cursor_max_velocity
        if profile.cursor_max_velocity_precision is not None:
            cursor_config.max_velocity_precision = profile.cursor_max_velocity_precision
        
        return cursor_config
    
    def map_gesture_to_action(self, gesture_type) -> str:
        """Map a gesture type to an action based on current profile."""
        if not self._current_profile:
            return gesture_type.name.lower()
        
        profile = self._current_profile
        mapping = {
            "LEFT_CLICK": profile.input_left_click_action,
            "RIGHT_CLICK": profile.input_right_click_action,
            "MIDDLE_CLICK": profile.input_middle_click_action,
            "SCROLL_UP": profile.input_scroll_up_action,
            "SCROLL_DOWN": profile.input_scroll_down_action,
            "SCROLL_HORIZONTAL": profile.input_scroll_horizontal_action,
            "DRAG_START": profile.input_drag_action,
            "PINCH_CONFIRM": profile.input_pinch_confirm_action,
            "OPEN_PALM": profile.input_open_palm_action,
            "FIST": profile.input_fist_action,
            "THUMB_GESTURE": profile.input_thumb_gesture_action,
            "TWO_HAND_GESTURE": profile.input_two_hand_action,
        }
        
        action = mapping.get(gesture_type.name)
        if action:
            return action
        
        # Check custom actions
        if gesture_type.name in profile.custom_actions:
            return profile.custom_actions[gesture_type.name]
        
        return gesture_type.name.lower()


# Global profile manager instance
_profile_manager: Optional[ProfileManager] = None


def get_profile_manager(config_dir: Optional[Path] = None) -> ProfileManager:
    """Get or create the global profile manager."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager(config_dir)
    return _profile_manager


def reset_profile_manager():
    """Reset the global profile manager (for testing)."""
    global _profile_manager
    _profile_manager = None


# Backward compatibility
ProfileInfo = ProfileConfig