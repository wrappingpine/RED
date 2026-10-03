"""
Configuration Schema for Air Mouse

Comprehensive JSON schema validation and profile management
for airmouse hand-tracking application configuration.
"""

from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Version of the configuration schema
CONFIG_SCHEMA_VERSION = "1.0.0"

# =============================================================================
# ENUM DEFINITIONS
# =============================================================================

class SensitivityMode(str, Enum):
    """Cursor sensitivity modes."""
    PRECISION = "precision"
    NORMAL = "normal"
    FAST = "fast"

class SmoothingAlgorithm(str, Enum):
    """Available smoothing algorithms."""
    NONE = "none"
    EMA = "ema"
    ONE_EURO = "one_euro"

class TrackingState(str, Enum):
    """Hand tracking states."""
    NO_HAND = "no_hand"
    TRACKING_ONE_HAND = "tracking_one_hand"
    TRACKING_TWO_HANDS = "tracking_two_hands"
    LOST_TRACK = "lost_track"
    FROZEN = "frozen"

class PreferredHandedness(str, Enum):
    """Preferred hand selection."""
    RIGHT = "Right"
    LEFT = "Left"
    ANY = "Any"

class GestureType(str, Enum):
    """Gesture types."""
    NONE = "none"
    POINT = "point"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    DRAG_START = "drag_start"
    DRAG_END = "drag_end"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_horizontal"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    PAUSE_TRACKING = "pause_tracking"
    RESUME_TRACKING = "resume_tracking"
    THUMB_GESTURE = "thumb_gesture"
    TWO_HAND_GESTURE = "two_hand_gesture"

# =============================================================================
# CONFIGURATION SCHEMA
# =============================================================================

CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Air Mouse Configuration",
    "description": "Configuration schema for airmouse hand-tracking application",
    "version": CONFIG_SCHEMA_VERSION,
    "type": "object",
    
    "required": ["version", "cursor", "tracking"],
    
    "properties": {
        "version": {
            "type": "string",
            "pattern": "^\\d+\\.\\d+\\.\\d+$",
            "description": "Configuration schema version"
        },
        
        "name": {
            "type": "string",
            "description": "Profile name"
        },
        
        "description": {
            "type": "string",
            "description": "Profile description"
        },
        
        "author": {
            "type": "string",
            "description": "Profile author"
        },
        
        "created": {
            "type": "string",
            "format": "date-time",
            "description": "Creation date"
        },
        
        "modified": {
            "type": "string",
            "format": "date-time",
            "description": "Last modified date"
        },
        
        "cursor": {
            "type": "object",
            "description": "Cursor control configuration",
            
            "properties": {
                "screen_width": {
                    "type": "integer",
                    "minimum": 800,
                    "maximum": 8000,
                    "default": 1920,
                    "description": "Screen width in pixels"
                },
                
                "screen_height": {
                    "type": "integer",
                    "minimum": 600,
                    "maximum": 6000,
                    "default": 1080,
                    "description": "Screen height in pixels"
                },
                
                "camera_width": {
                    "type": "integer",
                    "minimum": 320,
                    "maximum": 1920,
                    "default": 640,
                    "description": "Camera frame width"
                },
                
                "camera_height": {
                    "type": "integer",
                    "minimum": 240,
                    "maximum": 1080,
                    "default": 480,
                    "description": "Camera frame height"
                },
                
                "dead_zone_radius": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.1,
                    "default": 0.02,
                    "description": "Dead zone radius as fraction of frame"
                },
                
                "sensitivity_mode": {
                    "type": "string",
                    "enum": [mode.value for mode in SensitivityMode],
                    "default": SensitivityMode.NORMAL.value,
                    "description": "Cursor sensitivity mode"
                },
                
                "base_sensitivity": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "default": 1.0,
                    "description": "Base sensitivity multiplier"
                },
                
                "sensitivity_precision": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.12,
                    "description": "Sensitivity multiplier for precision mode"
                },
                
                "sensitivity_normal": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.32,
                    "description": "Sensitivity multiplier for normal mode"
                },
                
                "sensitivity_fast": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.48,
                    "description": "Sensitivity multiplier for fast mode"
                },
                
                "acceleration": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 3.0,
                    "default": 1.2,
                    "description": "Acceleration curve multiplier"
                },
                
                "max_velocity": {
                    "type": "integer",
                    "minimum": 100,
                    "maximum": 5000,
                    "default": 2000,
                    "description": "Maximum cursor velocity (pixels/sec)"
                },
                
                "max_velocity_precision": {
                    "type": "integer",
                    "minimum": 100,
                    "maximum": 2000,
                    "default": 500,
                    "description": "Max velocity in precision mode"
                },
                
                "smoothing": {
                    "type": "string",
                    "enum": [alg.value for alg in SmoothingAlgorithm],
                    "default": SmoothingAlgorithm.ONE_EURO.value,
                    "description": "Smoothing algorithm"
                },
                
                "ema_alpha": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.3,
                    "description": "EMA smoothing alpha (0=more smoothing)"
                },
                
                "one_euro_min_cutoff": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "default": 1.0,
                    "description": "One Euro filter minimum cutoff frequency"
                },
                
                "one_euro_beta": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                    "description": "One Euro filter beta parameter"
                },
                
                "one_euro_d_cutoff": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "default": 1.0,
                    "description": "One Euro filter derivative cutoff"
                },
                
                "invert_x": {
                    "type": "boolean",
                    "default": False,
                    "description": "Invert horizontal movement"
                },
                
                "invert_y": {
                    "type": "boolean",
                    "default": False,
                    "description": "Invert vertical movement"
                },
                
                "use_index_tip": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use index finger tip (True) or palm center (False)"
                },
                
                "monitor_count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 1,
                    "description": "Number of active monitors"
                },
                
                "monitor_arrangement": {
                    "type": "string",
                    "enum": ["horizontal", "grid"],
                    "default": "horizontal",
                    "description": "Monitor arrangement layout"
                },
                
                "primary_monitor": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 9,
                    "default": 0,
                    "description": "Primary monitor index"
                }
            },
            
            "additionalProperties": False
        },
        
        "tracking": {
            "type": "object",
            "description": "Tracking processor configuration",
            
            "properties": {
                "min_hand_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                    "description": "Minimum hand confidence threshold"
                },
                
                "min_landmark_visibility": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Minimum landmark visibility threshold"
                },
                
                "min_face_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Minimum face confidence threshold"
                },
                
                "preferred_handedness": {
                    "type": "string",
                    "enum": [hand.value for hand in PreferredHandedness],
                    "default": PreferredHandedness.RIGHT.value,
                    "description": "Preferred hand for primary tracking"
                },
                
                "max_landmark_jump": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.15,
                    "description": "Maximum normalized landmark jump per frame"
                },
                
                "max_wrist_jump": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Maximum wrist landmark jump (more lenient)"
                },
                
                "max_hand_center_jump": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.25,
                    "description": "Maximum hand center jump per frame"
                },
                
                "one_euro_min_cutoff": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "default": 1.0,
                    "description": "One Euro filter parameters for tracking"
                },
                
                "one_euro_beta": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.007,
                    "description": "One Euro filter beta"
                },
                
                "one_euro_d_cutoff": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "default": 1.0,
                    "description": "One Euro filter derivative cutoff"
                },
                
                "dead_zone_radius": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.1,
                    "default": 0.015,
                    "description": "Normalized dead zone radius"
                },
                
                "max_velocity": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Maximum normalized velocity per frame"
                },
                
                "velocity_smoothing": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.3,
                    "description": "Velocity smoothing factor"
                },
                
                "max_lost_frames": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60,
                    "default": 15,
                    "description": "Frames to hold position before resetting"
                },
                
                "reset_on_large_jump": {
                    "type": "boolean",
                    "default": True,
                    "description": "Reset tracking when landmarks jump too far"
                },
                
                "stabilization_frames": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 30,
                    "default": 5,
                    "description": "Frames to skip jump check during stabilization"
                },
                
                "use_head_relative": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable head-relative tracking"
                },
                
                "virtual_plane_distance": {
                    "type": "number",
                    "minimum": 0.2,
                    "maximum": 2.0,
                    "default": 0.30,
                    "description": "Distance to virtual plane (meters)"
                },
                
                "virtual_plane_width": {
                    "type": "number",
                    "minimum": 0.2,
                    "maximum": 2.0,
                    "default": 0.40,
                    "description": "Width of virtual plane (meters)"
                },
                
                "virtual_plane_height": {
                    "type": "number",
                    "minimum": 0.2,
                    "maximum": 2.0,
                    "default": 0.25,
                    "description": "Height of virtual plane (meters)"
                },
                
                "use_head_coords_for_ray": {
                    "type": "boolean",
                    "default": True,
                    "description": "Compute ray in head coordinates for accuracy"
                },
                
                "head_coords_smoothing_alpha": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.3,
                    "description": "Temporal smoothing for head coordinate system"
                },
                
                "head_confidence_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Minimum face confidence for head-relative mode (alias for min_face_confidence)"
                },
                
                "reference_point_update_mode": {
                    "type": "string",
                    "enum": ["every_frame", "dead_zone_exit"],
                    "default": "every_frame",
                    "description": "Reference point update strategy: every_frame for head-relative, dead_zone_exit for legacy"
                },
                
                "enable_two_hand": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable stable two-hand tracking"
                },
                
                "track_primary_only": {
                    "type": "boolean",
                    "default": False,
                    "description": "If True, only track primary hand for cursor"
                }
            },
            
            "additionalProperties": False
        },
        
        "gestures": {
            "type": "object",
            "description": "Gesture recognition configuration",
            
            "properties": {
                "pinch_enter_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.2,
                    "default": 0.045,
                    "description": "Enter pinch state threshold"
                },
                
                "pinch_confirm_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.2,
                    "default": 0.040,
                    "description": "Confirm pinch threshold (stricter)"
                },
                
                "pinch_release_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.3,
                    "default": 0.070,
                    "description": "Release pinch threshold"
                },
                
                "click_max_duration": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "default": 0.5,
                    "description": "Maximum duration for click (seconds)"
                },
                
                "scroll_sensitivity": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 2.0,
                    "default": 1.0,
                    "description": "Scroll sensitivity multiplier"
                },
                
                "scroll_dead_zone": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.1,
                    "default": 0.05,
                    "description": "Scroll dead zone (normalized)"
                },
                
                "dwell_time": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "default": 0.5,
                    "description": "Dwell time threshold (seconds)"
                },
                
                "gesture_hysteresis_frames": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                    "description": "Frames in candidate state before confirmation"
                },
                
                "phase_dwell_frames": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                    "description": "Frames in CANDIDATE before STABLE"
                },
                
                "confidence_thresholds": {
                    "type": "object",
                    "description": "Gesture confidence thresholds by type",
                    
                    "properties": {
                        "high_risk": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.7,
                            "description": "Confidence threshold for high-risk gestures"
                        },
                        
                        "medium_risk": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.5,
                            "description": "Confidence threshold for medium-risk gestures"
                        },
                        
                        "low_risk": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.3,
                            "description": "Confidence threshold for low-risk gestures"
                        }
                    },
                    
                    "additionalProperties": False
                },
                
                "enable_drag": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable drag gestures"
                },
                
                "enable_scroll": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable scroll gestures"
                },
                
                "enable_palm_control": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable palm-based controls"
                },
                
                "enable_fist_gesture": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable fist gesture for pause/resume"
                }
            },
            
            "additionalProperties": False
        },
        
        "hotkeys": {
            "type": "object",
            "description": "Hotkey configuration",
            
            "properties": {
                "emergency_disable": {
                    "type": "object",
                    "description": "Emergency disable hotkey",
                    
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "modifiers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["super", "alt"],
                            "description": "Keyboard modifiers"
                        },
                        "key": {"type": "string", "default": "A", "description": "Key code"},
                        "callback": {"type": "string", "description": "Callback function name"}
                    },
                    
                    "additionalProperties": False
                },
                
                "pause_resume": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "modifiers": {"type": "array", "items": {"type": "string"}, "default": ["super", "alt"]},
                        "key": {"type": "string", "default": "P"}
                    },
                    "additionalProperties": False
                },
                
                "calibrate": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "modifiers": {"type": "array", "items": {"type": "string"}, "default": ["super", "alt"]},
                        "key": {"type": "string", "default": "C"}
                    },
                    "additionalProperties": False
                },
                
                "settings": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "modifiers": {"type": "array", "items": {"type": "string"}, "default": ["super", "alt"]},
                        "key": {"type": "string", "default": "S"}
                    },
                    "additionalProperties": False
                },
                
                "precision_toggle": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "modifiers": {"type": "array", "items": {"type": "string"}, "default": ["super", "alt"]},
                        "key": {"type": "string", "default": "M"}
                    },
                    "additionalProperties": False
                }
            },
            
            "additionalProperties": False
        },
        
        "safety": {
            "type": "object",
            "description": "Safety system configuration",
            
            "properties": {
                "emergency_disable": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "confirmation_required": {"type": "boolean", "default": False},
                        "cooldown_seconds": {"type": "integer", "minimum": 0, "default": 5}
                    },
                    "additionalProperties": False
                },
                
                "focus_safety": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "check_interval_ms": {"type": "integer", "minimum": 100, "default": 500}
                    },
                    "additionalProperties": False
                },
                
                "input_health_monitoring": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "health_check_interval_ms": {"type": "integer", "minimum": 100, "default": 1000},
                        "max_write_errors": {"type": "integer", "minimum": 1, "default": 10},
                        "auto_recovery": {"type": "boolean", "default": True}
                    },
                    "additionalProperties": False
                },
                
                "safety_gate": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "require_stable_tracking": {"type": "boolean", "default": True},
                        "require_min_confidence": {"type": "boolean", "default": True},
                        "confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.6},
                        "max_latency_ms": {"type": "integer", "minimum": 100, "default": 1000}
                    },
                    "additionalProperties": False
                }
            },
            
            "additionalProperties": False
        },
        
        "performance": {
            "type": "object",
            "description": "Performance tuning configuration",
            
            "properties": {
                "target_fps": {"type": "number", "minimum": 10, "maximum": 120, "default": 60},
                "pipeline_latency_target_ms": {"type": "integer", "minimum": 10, "default": 100},
                "memory_limit_mb": {"type": "integer", "minimum": 512, "default": 2048},
                "enable_profiling": {"type": "boolean", "default": False},
                "debug_logging": {"type": "boolean", "default": False}
            },
            
            "additionalProperties": False
        },
        
        "debug": {
            "type": "object",
            "description": "Debug and development settings",
            
            "properties": {
                "enable_visualization": {"type": "boolean", "default": False},
                "landmark_visualization": {"type": "boolean", "default": False},
                "projection_debug": {"type": "boolean", "default": False},
                "gesture_debug": {"type": "boolean", "default": False},
                "show_fps": {"type": "boolean", "default": False},
                "show_latency": {"type": "boolean", "default": False},
                "log_level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                    "default": "INFO"
                }
            },
            
            "additionalProperties": False
        }
    },
    
    "additionalProperties": False
}