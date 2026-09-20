"""
Diagnostics module for AirMouse RED.

Provides deep, structured observability into the entire airmouse system per spec §19:
- Camera Diagnostics: frame rates, capture latency, frame drops, exposure/brightness metrics.
- Tracking Diagnostics: landmark confidence levels, tracking state, face tracking status.
- Gesture Diagnostics: gesture recognition accuracy, active gestures, event rate.
- Input Backend Diagnostics: active backend (uinput, ydotool), latency, event buffer status.
- System Diagnostics: CPU, memory, thread pool health, device paths, and platform info.
- Performance Metrics: pipeline stage-by-stage latency profile.
"""

import os
import sys
import time
import psutil
import logging
import platform
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticSnapshot:
    """Consolidated snapshot containing all sub-system metrics."""
    timestamp: float
    system: Dict[str, Any]
    camera: Dict[str, Any]
    tracking: Dict[str, Any]
    gestures: Dict[str, Any]
    input: Dict[str, Any]
    performance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to plain dictionary."""
        return asdict(self)


class DiagnosticsCollector:
    """
    Active monitor that pulls metrics from running components or probes the local environment.
    """

    def __init__(self, controller=None):
        self.controller = controller

    def collect_system_info(self) -> Dict[str, Any]:
        """Gather platform and system resource usage metrics."""
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=0)
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()

        # Check /dev/uinput access
        uinput_exists = os.path.exists("/dev/uinput")
        uinput_writable = os.access("/dev/uinput", os.W_OK) if uinput_exists else False

        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_percent": cpu_percent,
            "memory_rss_mb": mem_info.rss / (1024 * 1024),
            "memory_percent": mem_percent,
            "uinput_available": uinput_exists,
            "uinput_writable": uinput_writable,
            "threads_count": len(process.threads()),
            "load_average": os.getloadavg() if hasattr(os, "getloadavg") else (0, 0, 0)
        }

    def collect_camera_info(self) -> Dict[str, Any]:
        """Gather camera metrics from the active controller or probe locally."""
        if self.controller and self.controller.camera:
            cam = self.controller.camera
            stats = cam.get_brightness_stats()
            resolution = cam.get_resolution()
            controls = cam.get_camera_controls()
            return {
                "active": cam.is_running(),
                "width": resolution[0],
                "height": resolution[1],
                "brightness": stats.get("current_brightness", 0.0),
                "avg_brightness": stats.get("avg_brightness", 0.0),
                "exposure": stats.get("current_exposure", -1),
                "gain": stats.get("current_gain", -1),
                "auto_exposure": controls.get("auto_exposure", False),
                "v4l2_controls": controls
            }
        
        # Probe locally if controller is not present/running
        from ..camera.manager import diagnose_camera_issues
        issues = diagnose_camera_issues()
        return {
            "active": False,
            "probed_devices": issues.get("video_devices", []),
            "v4l2_available": issues.get("v4l2_available", False),
            "user_in_video_group": issues.get("in_video_group", False)
        }

    def collect_tracking_info(self) -> Dict[str, Any]:
        """Gather landmarker tracking confidence and state metrics."""
        if self.controller and self.controller.tracking_processor:
            proc = self.controller.tracking_processor
            has_hand = self.controller.stats.hand_detection_time_ms > 0
            return {
                "tracking_mode": "3D_HEAD_RELATIVE" if self.controller.config.tracking.use_head_relative else "2D",
                "hand_detected": has_hand,
                "face_detected": self.controller.face_tracker is not None and self.controller.stats.gesture_time_ms > 0, # face detection reuse
                "confidence_threshold": proc.config.confidence_threshold,
                "min_face_confidence": proc.config.min_face_confidence
            }
        return {
            "tracking_mode": "N/A",
            "hand_detected": False,
            "face_detected": False
        }

    def collect_gestures_info(self) -> Dict[str, Any]:
        """Gather gesture recognition and hysteresis events."""
        if self.controller and self.controller.gesture_recognizer:
            rec = self.controller.gesture_recognizer
            return {
                "pinch_enter_threshold": rec.config.pinch_enter_threshold,
                "pinch_confirm_threshold": rec.config.pinch_confirm_threshold,
                "pinch_release_threshold": rec.config.pinch_release_threshold,
                "tracking_paused": rec.is_tracking_paused() if hasattr(rec, "is_tracking_paused") else False
            }
        return {}

    def collect_input_info(self) -> Dict[str, Any]:
        """Gather input subsystem and active driver backend info."""
        if self.controller and self.controller.input_manager:
            mgr = self.controller.input_manager
            return {
                "backend": mgr.get_backend_type().value if hasattr(mgr, "get_backend_type") else "unknown",
                "desktop_env": mgr.get_desktop_environment().value if hasattr(mgr, "get_desktop_environment") else "unknown",
                "initialized": True
            }
        return {
            "backend": "none",
            "initialized": False
        }

    def collect_performance_info(self) -> Dict[str, Any]:
        """Gather latency profiles across pipeline stages."""
        if self.controller:
            stats = self.controller.get_stats()
            return {
                "fps": stats.fps,
                "frame_time_ms": stats.frame_time_ms,
                "pipeline_latency_ms": stats.pipeline_latency_ms,
                "stage_latencies_ms": stats.stage_latencies,
                "frames_processed": stats.frames_processed,
                "frames_dropped": stats.frames_dropped,
                "drop_rate": stats.drop_rate
            }
        return {}

    def collect_all(self) -> DiagnosticSnapshot:
        """Acquire a synchronized full system observability snapshot."""
        return DiagnosticSnapshot(
            timestamp=time.time(),
            system=self.collect_system_info(),
            camera=self.collect_camera_info(),
            tracking=self.collect_tracking_info(),
            gestures=self.collect_gestures_info(),
            input=self.collect_input_info(),
            performance=self.collect_performance_info()
        )
