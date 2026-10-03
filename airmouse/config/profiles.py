"""
Configuration Profile Manager for AirMouse

Handles profile loading, saving, and validation for airmouse configuration profiles.
Simple JSON-based profile system for P1.10 Configuration Profiles.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from airmouse.config.schema import (
    CONFIG_SCHEMA, CONFIG_SCHEMA_VERSION, 
    SensitivityMode, SmoothingAlgorithm, PreferredHandedness
)

logger = logging.getLogger(__name__)

# Default configuration directory
DEFAULT_CONFIG_DIR = Path.home() / ".airmouse"
DEFAULT_PROFILES_DIR = DEFAULT_CONFIG_DIR / "profiles"

# Built-in default profiles
DEFAULT_PROFILES = {
    "precision": {
        "name": "Precision Mode",
        "description": "High-precision, stable configuration for detailed work",
        "version": CONFIG_SCHEMA_VERSION,
        "author": "Air Mouse Team",
        "created": datetime.now(timezone.utc).isoformat(),
        "modified": datetime.now(timezone.utc).isoformat(),
        
        "cursor": {
            "screen_width": 1920,
            "screen_height": 1080,
            "camera_width": 1280,
            "camera_height": 720,
            "dead_zone_radius": 0.01,
            "sensitivity_mode": SensitivityMode.PRECISION.value,
            "base_sensitivity": 1.0,
            "sensitivity_precision": 0.12,
            "sensitivity_normal": 0.32,
            "sensitivity_fast": 0.48,
            "acceleration": 1.2,
            "max_velocity": 500,
            "smoothing": SmoothingAlgorithm.ONE_EURO.value,
            "ema_alpha": 0.3,
            "one_euro_min_cutoff": 2.0,
            "one_euro_beta": 0.01,
            "one_euro_d_cutoff": 2.0,
            "invert_x": False,
            "invert_y": False,
            "use_index_tip": True,
            "monitor_count": 1,
            "monitor_arrangement": "horizontal",
            "primary_monitor": 0
        },
        
        "tracking": {
            "min_hand_confidence": 0.8,
            "min_landmark_visibility": 0.7,
            "min_face_confidence": 0.7,
            "preferred_handedness": PreferredHandedness.RIGHT.value,
            "max_landmark_jump": 0.15,
            "max_wrist_jump": 0.6,
            "max_hand_center_jump": 0.25,
            "one_euro_min_cutoff": 2.0,
            "one_euro_beta": 0.01,
            "one_euro_d_cutoff": 2.0,
            "dead_zone_radius": 0.015,
            "max_velocity": 0.3,
            "velocity_smoothing": 0.2,
            "max_lost_frames": 20,
            "reset_on_large_jump": True,
            "stabilization_frames": 10,
            "use_head_relative": True,
            "virtual_plane_distance": 0.35,
            "virtual_plane_width": 0.50,
            "virtual_plane_height": 0.30,
            "use_head_coords_for_ray": True,
            "head_coords_smoothing_alpha": 0.4,
            "head_confidence_threshold": 0.7,
            "reference_point_update_mode": "every_frame",
            "enable_two_hand": False,
            "track_primary_only": True
        },
        
        "gestures": {
            "pinch_enter_threshold": 0.03,
            "pinch_confirm_threshold": 0.025,
            "pinch_release_threshold": 0.06,
            "click_max_duration": 0.3,
            "scroll_sensitivity": 0.8,
            "scroll_dead_zone": 0.03,
            "dwell_time": 0.3,
            "gesture_hysteresis_frames": 5,
            "phase_dwell_frames": 5,
            "enable_drag": True,
            "enable_scroll": True,
            "enable_palm_control": True,
            "enable_fist_gesture": True
        },
        
        "hotkeys": {
            "emergency_disable": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "A",
                "callback": "emergency_disable"
            },
            "pause_resume": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "P"
            },
            "calibrate": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "C"
            },
            "settings": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "S"
            },
            "precision_toggle": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "M"
            }
        },
        
        "safety": {
            "emergency_disable": {
                "enabled": True,
                "confirmation_required": True,
                "cooldown_seconds": 10
            },
            "focus_safety": {
                "enabled": True,
                "check_interval_ms": 200
            },
            "input_health_monitoring": {
                "enabled": True,
                "health_check_interval_ms": 500,
                "max_write_errors": 5,
                "auto_recovery": True
            },
            "safety_gate": {
                "enabled": True,
                "require_stable_tracking": True,
                "require_min_confidence": True,
                "confidence_threshold": 0.7,
                "max_latency_ms": 500
            }
        },
        
        "performance": {
            "target_fps": 60,
            "pipeline_latency_target_ms": 80,
            "memory_limit_mb": 2048,
            "enable_profiling": False,
            "debug_logging": False
        },
        
        "debug": {
            "enable_visualization": False,
            "landmark_visualization": False,
            "projection_debug": False,
            "gesture_debug": False,
            "show_fps": False,
            "show_latency": False,
            "log_level": "INFO"
        }
    },
    
    "normal": {
        "name": "Normal Mode",
        "description": "Balanced configuration for general use",
        "version": CONFIG_SCHEMA_VERSION,
        "author": "Air Mouse Team",
        "created": datetime.now(timezone.utc).isoformat(),
        "modified": datetime.now(timezone.utc).isoformat(),
        
        "cursor": {
            "screen_width": 1920,
            "screen_height": 1080,
            "camera_width": 1280,
            "camera_height": 720,
            "dead_zone_radius": 0.02,
            "sensitivity_mode": SensitivityMode.NORMAL.value,
            "base_sensitivity": 1.0,
            "smoothing": SmoothingAlgorithm.ONE_EURO.value,
            "ema_alpha": 0.3,
            "one_euro_min_cutoff": 1.5,
            "one_euro_beta": 0.007,
            "one_euro_d_cutoff": 1.0,
            "invert_x": False,
            "invert_y": False,
            "use_index_tip": True,
            "monitor_count": 1,
            "monitor_arrangement": "horizontal",
            "primary_monitor": 0
        },
        
        "tracking": {
            "min_hand_confidence": 0.6,
            "min_landmark_visibility": 0.5,
            "min_face_confidence": 0.5,
            "preferred_handedness": PreferredHandedness.RIGHT.value,
            "max_landmark_jump": 0.15,
            "max_wrist_jump": 0.5,
            "max_hand_center_jump": 0.25,
            "one_euro_min_cutoff": 1.0,
            "one_euro_beta": 0.007,
            "one_euro_d_cutoff": 1.0,
            "dead_zone_radius": 0.015,
            "max_velocity": 0.5,
            "velocity_smoothing": 0.3,
            "max_lost_frames": 15,
            "reset_on_large_jump": True,
            "stabilization_frames": 5,
            "use_head_relative": True,
            "virtual_plane_distance": 0.30,
            "virtual_plane_width": 0.40,
            "virtual_plane_height": 0.25,
            "use_head_coords_for_ray": True,
            "head_coords_smoothing_alpha": 0.3,
            "head_confidence_threshold": 0.5,
            "reference_point_update_mode": "every_frame",
            "enable_two_hand": True,
            "track_primary_only": False
        },
        
        "gestures": {
            "pinch_enter_threshold": 0.045,
            "pinch_confirm_threshold": 0.040,
            "pinch_release_threshold": 0.070,
            "click_max_duration": 0.5,
            "scroll_sensitivity": 1.0,
            "scroll_dead_zone": 0.05,
            "dwell_time": 0.5,
            "gesture_hysteresis_frames": 3,
            "phase_dwell_frames": 3,
            "enable_drag": True,
            "enable_scroll": True,
            "enable_palm_control": True,
            "enable_fist_gesture": True
        },
        
        "hotkeys": {
            "emergency_disable": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "A",
                "callback": "emergency_disable"
            },
            "pause_resume": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "P"
            },
            "calibrate": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "C"
            },
            "settings": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "S"
            },
            "precision_toggle": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "M"
            }
        },
        
        "safety": {
            "emergency_disable": {
                "enabled": True,
                "confirmation_required": False,
                "cooldown_seconds": 5
            },
            "focus_safety": {
                "enabled": True,
                "check_interval_ms": 500
            },
            "input_health_monitoring": {
                "enabled": True,
                "health_check_interval_ms": 1000,
                "max_write_errors": 10,
                "auto_recovery": True
            },
            "safety_gate": {
                "enabled": True,
                "require_stable_tracking": True,
                "require_min_confidence": True,
                "confidence_threshold": 0.6,
                "max_latency_ms": 1000
            }
        },
        
        "performance": {
            "target_fps": 60,
            "pipeline_latency_target_ms": 100,
            "memory_limit_mb": 2048,
            "enable_profiling": False,
            "debug_logging": False
        },
        
        "debug": {
            "enable_visualization": False,
            "landmark_visualization": False,
            "projection_debug": False,
            "gesture_debug": False,
            "show_fps": False,
            "show_latency": False,
            "log_level": "INFO"
        }
    },
    
    "fast": {
        "name": "Fast Mode",
        "description": "High-speed configuration for quick navigation",
        "version": CONFIG_SCHEMA_VERSION,
        "author": "Air Mouse Team",
        "created": datetime.now(timezone.utc).isoformat(),
        "modified": datetime.now(timezone.utc).isoformat(),
        
        "cursor": {
            "screen_width": 1920,
            "screen_height": 1080,
            "camera_width": 1280,
            "camera_height": 720,
            "dead_zone_radius": 0.03,
            "sensitivity_mode": SensitivityMode.FAST.value,
            "base_sensitivity": 1.0,
            "smoothing": SmoothingAlgorithm.EMA.value,
            "ema_alpha": 0.4,
            "one_euro_min_cutoff": 3.0,
            "one_euro_beta": 0.01,
            "one_euro_d_cutoff": 3.0,
            "invert_x": False,
            "invert_y": False,
            "use_index_tip": True,
            "monitor_count": 1,
            "monitor_arrangement": "horizontal",
            "primary_monitor": 0
        },
        
        "tracking": {
            "min_hand_confidence": 0.4,
            "min_landmark_visibility": 0.4,
            "min_face_confidence": 0.4,
            "preferred_handedness": PreferredHandedness.RIGHT.value,
            "max_landmark_jump": 0.20,
            "max_wrist_jump": 0.4,
            "max_hand_center_jump": 0.30,
            "one_euro_min_cutoff": 5.0,
            "one_euro_beta": 0.01,
            "one_euro_d_cutoff": 5.0,
            "dead_zone_radius": 0.025,
            "max_velocity": 1.0,
            "velocity_smoothing": 0.1,
            "max_lost_frames": 10,
            "reset_on_large_jump": False,
            "stabilization_frames": 3,
            "use_head_relative": True,
            "virtual_plane_distance": 0.25,
            "virtual_plane_width": 0.60,
            "virtual_plane_height": 0.20,
            "use_head_coords_for_ray": True,
            "head_coords_smoothing_alpha": 0.2,
            "head_confidence_threshold": 0.4,
            "reference_point_update_mode": "every_frame",
            "enable_two_hand": True,
            "track_primary_only": False
        },
        
        "gestures": {
            "pinch_enter_threshold": 0.06,
            "pinch_confirm_threshold": 0.055,
            "pinch_release_threshold": 0.080,
            "click_max_duration": 0.8,
            "scroll_sensitivity": 1.5,
            "scroll_dead_zone": 0.07,
            "dwell_time": 0.2,
            "gesture_hysteresis_frames": 2,
            "phase_dwell_frames": 2,
            "enable_drag": True,
            "enable_scroll": True,
            "enable_palm_control": False,
            "enable_fist_gesture": False
        },
        
        "hotkeys": {
            "emergency_disable": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "A",
                "callback": "emergency_disable"
            },
            "pause_resume": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "P"
            },
            "calibrate": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "C"
            },
            "settings": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "S"
            },
            "precision_toggle": {
                "enabled": True,
                "modifiers": ["super", "alt"],
                "key": "M"
            }
        },
        
        "safety": {
            "emergency_disable": {
                "enabled": True,
                "confirmation_required": False,
                "cooldown_seconds": 3
            },
            "focus_safety": {
                "enabled": True,
                "check_interval_ms": 1000
            },
            "input_health_monitoring": {
                "enabled": True,
                "health_check_interval_ms": 2000,
                "max_write_errors": 20,
                "auto_recovery": True
            },
            "safety_gate": {
                "enabled": True,
                "require_stable_tracking": False,
                "require_min_confidence": True,
                "confidence_threshold": 0.4,
                "max_latency_ms": 2000
            }
        },
        
        "performance": {
            "target_fps": 60,
            "pipeline_latency_target_ms": 200,
            "memory_limit_mb": 2048,
            "enable_profiling": False,
            "debug_logging": False
        },
        
        "debug": {
            "enable_visualization": False,
            "landmark_visualization": False,
            "projection_debug": False,
            "gesture_debug": False,
            "show_fps": False,
            "show_latency": False,
            "log_level": "INFO"
        }
    }
}
class ProfileManager:
    """
    Simple configuration profile manager for AirMouse.
    
    Handles loading, saving, and validation of configuration profiles.
    Provides built-in profiles (precision, normal, fast) and file-based profile management.
    """
    
    def __init__(self, profiles_dir: Optional[str] = None):
        """
        Initialize profile manager.
        
        Args:
            profiles_dir: Directory for user profiles. Defaults to ~/.airmouse/profiles
        """
        self.profiles_dir = Path(profiles_dir) if profiles_dir else DEFAULT_PROFILES_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        self._default_profiles = DEFAULT_PROFILES.copy()
        
    def load_profile(self, name: str) -> Dict[str, Any]:
        """
        Load a configuration profile by name.
        
        Args:
            name: Profile name ('precision', 'normal', 'fast', or custom profile filename)
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If profile not found or invalid
        """
        if not name:
            raise ValueError("Profile name cannot be empty")
            
        # Check built-in profiles
        if name in self._default_profiles:
            return self._default_profiles[name].copy()
            
        # Check custom profile file
        profile_path = self.profiles_dir / f"{name}.json"
        
        if profile_path.exists():
            return self._load_profile_from_file(profile_path)
        
        # Profile not found
        available = list(self._default_profiles.keys()) + [
            f.stem for f in self.profiles_dir.glob("*.json")
        ]
        raise ValueError(f"Profile '{name}' not found. Available: {', '.join(available)}")
    
    def save_profile(self, name: str, config: Dict[str, Any], make_backup: bool = True) -> bool:
        """
        Save a configuration profile.
        
        Args:
            name: Profile name
            config: Configuration dictionary
            make_backup: Whether to create a backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate configuration
            self._validate_config(config)
            
            # Update metadata
            config = config.copy()
            config["name"] = name
            config["version"] = CONFIG_SCHEMA_VERSION
            config["modified"] = datetime.now(timezone.utc).isoformat()
            
            # Save to profiles directory
            profile_path = self.profiles_dir / f"{name}.json"
            
            # Create backup if requested
            if make_backup:
                self._create_backup(name, config)
            
            # Write profile to file
            with open(profile_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Profile '{name}' saved successfully to {profile_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save profile '{name}': {e}")
            return False
    
    def list_profiles(self) -> List[str]:
        """
        List all available profiles.
        
        Returns:
            List of profile names
        """
        profiles = list(self._default_profiles.keys())
        
        # Add custom profiles
        for profile_file in self.profiles_dir.glob("*.json"):
            profiles.append(profile_file.stem)
        
        return sorted(list(set(profiles)))
    
    def delete_profile(self, name: str) -> bool:
        """
        Delete a profile.
        
        Args:
            name: Profile name
            
        Returns:
            True if successful, False otherwise
        """
        # Prevent deletion of default profiles
        if name in self._default_profiles:
            logger.warning(f"Cannot delete default profile '{name}'")
            return False
        
        # Check profile file
        profile_path = self.profiles_dir / f"{name}.json"
        if profile_path.exists():
            profile_path.unlink()
            logger.info(f"Profile '{name}' deleted")
            return True
        
        logger.warning(f"Profile '{name}' not found")
        return False
    
    def duplicate_profile(self, source: str, destination: str) -> bool:
        """
        Duplicate a profile.
        
        Args:
            source: Source profile name
            destination: Destination profile name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = self.load_profile(source)
            return self.save_profile(destination, config)
        except Exception as e:
            logger.error(f"Failed to duplicate profile '{source}' to '{destination}': {e}")
            return False
    
    def export_profile(self, name: str, export_dir: Optional[str] = None) -> bool:
        """
        Export a profile to file.
        
        Args:
            name: Profile name
            export_dir: Export directory (uses profiles_dir if None)
            
        Returns:
            True if successful, False otherwise
        """
        if export_dir is None:
            export_dir = self.profiles_dir
        
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        try:
            config = self.load_profile(name)
            
            export_file = export_path / f"{name}.json"
            with open(export_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Profile '{name}' exported to {export_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export profile '{name}': {e}")
            return False
    
    def import_profile(self, import_path: str) -> bool:
        """
        Import a profile from file.
        
        Args:
            import_path: Path to profile file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import_path = Path(import_path)
            if not import_path.exists():
                logger.error(f"Import path does not exist: {import_path}")
                return False
            
            # Load configuration
            with open(import_path, 'r') as f:
                config = json.load(f)
            
            # Extract profile name from filename
            name = import_path.stem
            
            # Save profile
            return self.save_profile(name, config)
            
        except Exception as e:
            logger.error(f"Failed to import profile from {import_path}: {e}")
            return False
    
    def get_default_profiles(self) -> List[str]:
        """
        Get list of default profile names.
        
        Returns:
            List of default profile names
        """
        return list(self._default_profiles.keys())
    
    def get_profile_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a profile.
        
        Args:
            name: Profile name
            
        Returns:
            Profile information dictionary
        """
        config = self.load_profile(name)
        
        return {
            "name": name,
            "version": config.get("version", "unknown"),
            "author": config.get("author", "unknown"),
            "created": config.get("created", "unknown"),
            "modified": config.get("modified", "unknown"),
            "description": config.get("description", ""),
            "cursor_sensitivity": config.get("cursor", {}).get("sensitivity_mode", "unknown"),
            "tracking_mode": "head-relative" if config.get("tracking", {}).get("use_head_relative", False) else "2d",
            "gestures_enabled": sum([
                config.get("gestures", {}).get("enable_drag", False),
                config.get("gestures", {}).get("enable_scroll", False),
                config.get("gestures", {}).get("enable_palm_control", False),
                config.get("gestures", {}).get("enable_fist_gesture", False)
            ])
        }
    
    def validate_profile(self, name: str) -> List[str]:
        """
        Validate a profile configuration.
        
        Args:
            name: Profile name
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            config = self.load_profile(name)
            self._validate_config(config)
            
        except Exception as e:
            errors.append(str(e))
        
        return errors
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required keys
        required_keys = ["version", "cursor", "tracking"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: '{key}'")
        
        # Validate schema
        try:
            import jsonschema
            jsonschema.validate(config, CONFIG_SCHEMA)
        except ImportError:
            logger.warning("jsonschema not available, skipping schema validation")
        except Exception as e:
            raise ValueError(f"Configuration validation error: {e}")
        
        # Application-specific validation
        self._validate_application_rules(config)
    
    def _validate_application_rules(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration against application rules.
        
        Args:
            config: Configuration dictionary
        """
        # Check for incompatible combinations
        cursor_config = config.get("cursor", {})
        tracking_config = config.get("tracking", {})
        gestures_config = config.get("gestures", {})
        
        # Check sensitivity mode consistency
        sensitivity_mode = cursor_config.get("sensitivity_mode", "normal")
        max_velocity = cursor_config.get("max_velocity", 2000)
        
        # Precision mode should have lower max velocity
        if sensitivity_mode == "precision" and max_velocity > 1000:
            logger.warning("Precision mode with high max_velocity may cause unstable tracking")
        
        # Fast mode should have higher max velocity  
        if sensitivity_mode == "fast" and max_velocity < 2000:
            logger.warning("Fast mode with low max_velocity may limit navigation speed")
        
        # Check tracking confidence thresholds
        min_hand_confidence = tracking_config.get("min_hand_confidence", 0.0)
        min_face_confidence = tracking_config.get("min_face_confidence", 0.0)
        
        if min_hand_confidence > 0.8 and min_face_confidence < 0.5:
            logger.warning("High hand confidence with low face confidence may cause tracking issues")
        
        # Check gesture consistency
        enable_drag = gestures_config.get("enable_drag", False)
        enable_scroll = gestures_config.get("enable_scroll", False)
        enable_palm = gestures_config.get("enable_palm_control", False)
        
        if not any([enable_drag, enable_scroll, enable_palm]):
            logger.warning("No gesture types enabled - consider enabling at least one")
    
    def _load_profile_from_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to profile file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If file cannot be loaded or is invalid
        """
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Validate configuration
            self._validate_config(config)
            
            logger.info(f"Profile '{file_path.stem}' loaded successfully from {file_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load profile from {file_path}: {e}")
            raise ValueError(f"Failed to load profile from {file_path}: {e}")
    
    def _create_backup(self, name: str, config: Dict[str, Any]) -> None:
        """
        Create backup of configuration profile.
        
        Args:
            name: Profile name
            config: Configuration dictionary
        """
        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{name}_{timestamp}.json"
            
            backup_path = self.profiles_dir / backup_filename
            
            # Add backup metadata
            config_with_metadata = config.copy()
            config_with_metadata["backup_timestamp"] = timestamp
            config_with_metadata["backup_reason"] = "automatic_backup"
            
            with open(backup_path, 'w') as f:
                json.dump(config_with_metadata, f, indent=2)
            
            logger.debug(f"Profile '{name}' backed up to {backup_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create backup for profile '{name}': {e}")
    
    def __str__(self) -> str:
        """String representation."""
        profiles = self.list_profiles()
        return f"ProfileManager(profiles={len(profiles)}, default={len(self._default_profiles)})"
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return self.__str__()
class ProfileInfo:
    """
    Profile information data class.
    
    Simple container for profile metadata.
    """
    
    def __init__(self, name: str, version: str = "", author: str = "",
                 description: str = "", created: str = "", modified: str = ""):
        self.name = name
        self.version = version
        self.author = author
        self.description = description
        self.created = created
        self.modified = modified

    def __str__(self) -> str:
        """String representation."""
        return f"ProfileInfo(name={self.name}, version={self.version}, author={self.author})"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return self.__str__()