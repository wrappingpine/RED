"""
Camera Manager Module for Air Mouse

Provides robust camera detection, selection, and management with diagnostic capabilities.
Includes auto-exposure/brightness monitoring for consistent hand tracking.
"""

import cv2
import os
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


logger = logging.getLogger(__name__)


class CameraErrorType(Enum):
    """Types of camera errors for diagnostic purposes."""
    NO_CAMERA = "no_camera"
    PERMISSION_DENIED = "permission_denied"
    CAMERA_IN_USE = "camera_in_use"
    CANNOT_OPEN = "cannot_open"
    UNKNOWN = "unknown"


@dataclass
class CameraInfo:
    """Information about a detected camera device."""
    index: int
    device_path: str
    name: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    available: bool = True
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.available:
            return f"{self.device_path} ({self.name}) - {self.width}x{self.height} @ {self.fps:.0f}fps"
        return f"{self.device_path} - UNAVAILABLE: {self.error}"


@dataclass
class CameraSettings:
    """Camera configuration settings."""
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30

    # Auto-exposure/brightness settings
    auto_exposure: bool = True          # Enable auto-exposure monitoring
    target_brightness: int = 130        # Target mean brightness (0-255)
    brightness_tolerance: int = 15      # Acceptable deviation from target
    exposure_min: int = 1               # Minimum exposure value (V4L2)
    exposure_max: int = 10000           # Maximum exposure value (V4L2)
    gain_min: int = 0                   # Minimum gain value (V4L2)
    gain_max: int = 255                 # Maximum gain value (V4L2)
    adjustment_interval: float = 1.0    # Seconds between adjustments
    brightness_history_size: int = 10   # Frames to average for brightness


class CameraManager:
    """
    Manages camera detection, selection, and frame capture.

    Features:
    - Automatic detection of /dev/video* devices
    - Detailed diagnostic information
    - Clean resource management
    - Configurable resolution and FPS
    - Auto-exposure/brightness monitoring for consistent tracking
    - Frame pooling to reduce allocations
    """

    def __init__(self):
        self._capture: Optional[cv2.VideoCapture] = None
        self._settings = CameraSettings()
        self._is_running = False

        # Auto-exposure state
        self._brightness_history: List[float] = []
        self._last_adjustment_time: float = 0.0
        self._current_exposure: int = -1
        self._current_gain: int = -1
        self._exposure_supported: bool = False
        self._gain_supported: bool = False

        # Frame pooling - pre-allocate frame buffers to avoid allocations
        self._frame_pool: List[np.ndarray] = []
        self._pool_size = 3
        self._frame_shape = None

    def detect_cameras(self) -> List[CameraInfo]:
        """
        Detect all available camera devices.

        Returns:
            List of CameraInfo objects with diagnostic details
        """
        cameras = []

        # Check /dev/video* devices
        video_devices = []
        for i in range(20):  # Check first 20 video devices
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                video_devices.append((i, device_path))

        if not video_devices:
            logger.warning("No /dev/video* devices found")
            return cameras

        for index, device_path in video_devices:
            info = self._probe_camera(index, device_path)
            cameras.append(info)

        logger.info(f"Detected {len(cameras)} camera(s)")
        for cam in cameras:
            logger.info(f"  {cam}")

        return cameras

    def _probe_camera(self, index: int, device_path: str) -> CameraInfo:
        """
        Probe a specific camera device for capabilities.

        Args:
            index: Camera index
            device_path: Device path (e.g., /dev/video0)

        Returns:
            CameraInfo with probe results
        """
        info = CameraInfo(index=index, device_path=device_path)

        # Try to open with V4L2 backend (preferred on Linux)
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

        if not cap.isOpened():
            # Try default backend
            cap = cv2.VideoCapture(index)

        if not cap.isOpened():
            info.available = False
            info.error = "Cannot open camera (permission denied or device busy)"
            # Check specific error type
            if not os.access(device_path, os.R_OK | os.W_OK):
                info.error = "Permission denied on device"
            return info

        # Get camera properties
        info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info.fps = cap.get(cv2.CAP_PROP_FPS)

        # Try to get camera name via v4l2-ctl if available
        info.name = self._get_camera_name(device_path)

        cap.release()
        return info

    def _get_camera_name(self, device_path: str) -> str:
        """Try to get human-readable camera name."""
        try:
            import subprocess
            result = subprocess.run(
                ["v4l2-ctl", "--device", device_path, "--info"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Card type' in line:
                        return line.split(':', 1)[1].strip()
        except Exception:
            pass
        return f"Camera {device_path}"

    def open_camera(self, settings: Optional[CameraSettings] = None) -> bool:
        """
        Open the camera with specified settings.

        Args:
            settings: Camera settings (uses current settings if None)

        Returns:
            True if camera opened successfully
        """
        if settings:
            self._settings = settings

        if self._capture is not None:
            self.close_camera()

        logger.info(f"Opening camera {self._settings.device_index} "
                   f"({self._settings.width}x{self._settings.height} @ {self._settings.fps}fps)")

        # Try V4L2 first for better Linux compatibility
        self._capture = cv2.VideoCapture(self._settings.device_index, cv2.CAP_V4L2)

        if not self._capture.isOpened():
            logger.warning("V4L2 backend failed, trying default backend")
            self._capture = cv2.VideoCapture(self._settings.device_index)

        if not self._capture.isOpened():
            logger.error(f"Failed to open camera {self._settings.device_index}")
            self._capture = None
            return False

        # Set camera properties
        # Use YUYV format for better MediaPipe compatibility (MJPG can cause issues)
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

        # Optimal settings for Ryzen 3 3250U: 640x480 @ 30fps
        # (can be overridden by settings if needed)
        target_width = self._settings.width if self._settings.width <= 640 else 640
        target_height = self._settings.height if self._settings.height <= 480 else 480
        target_fps = min(self._settings.fps, 30)

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        self._capture.set(cv2.CAP_PROP_FPS, target_fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        # Initialize frame pool with actual frame dimensions
        self._frame_shape = (target_height, target_width, 2)  # YUYV is 2 channels
        self._frame_pool = [np.zeros(self._frame_shape, dtype=np.uint8) for _ in range(self._pool_size)]
        self._pool_index = 0

        # Warmup frames to let camera auto-exposure stabilize
        for _ in range(5):
            self._capture.read()

        # Verify actual settings
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

        logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")

        # Check V4L2 controls for auto-exposure
        self._check_v4l2_controls()

        self._is_running = True
        return True

    def read_frame(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        """
        Read a single frame from the camera.

        Returns:
            Tuple of (success, frame) where frame is None on failure
        """
        if self._capture is None or not self._is_running:
            return False, None

        # Get next frame from pool
        pool_idx = self._pool_index
        self._pool_index = (self._pool_index + 1) % self._pool_size
        frame = self._frame_pool[pool_idx]

        ret, frame_data = self._capture.read(frame)
        if not ret:
            logger.warning("Failed to read frame from camera")
            return False, None

        # Convert YUYV to BGR for MediaPipe compatibility
        # YUYV is 2 channels (YUYV 4:2:2), BGR is 3 channels
        if frame.shape[2] == 2:  # YUYV format
            # Create BGR frame in pool buffer
            bgr_frame = self._frame_pool[(pool_idx + 1) % self._pool_size]
            if bgr_frame.shape[2] != 3:
                # Reallocate if needed
                bgr_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV, dst=bgr_frame)
            frame = bgr_frame

        # Auto-exposure/brightness monitoring (use original YUYV for brightness)
        if self._settings.auto_exposure:
            # Use Y channel from YUYV for brightness
            y_channel = frame_data[:, :, 0] if frame_data.shape[2] == 2 else frame
            brightness = np.mean(y_channel)
            self._adjust_exposure_gain(brightness)

        return True, frame

    def close_camera(self) -> None:
        """Release camera resources cleanly."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            self._is_running = False
            logger.info("Camera closed")

    def is_running(self) -> bool:
        """Check if camera is currently running."""
        return self._is_running and self._capture is not None and self._capture.isOpened()

    def get_settings(self) -> CameraSettings:
        """Get current camera settings."""
        return self._settings

    def get_resolution(self) -> Tuple[int, int]:
        """Get actual camera resolution."""
        if self._capture is not None and self._capture.isOpened():
            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return self._settings.width, self._settings.height

    def update_settings(self, settings: CameraSettings) -> bool:
        """
        Update camera settings (requires reopening camera).

        Args:
            settings: New camera settings

        Returns:
            True if reopened successfully
        """
        was_running = self.is_running()
        self.close_camera()
        self._settings = settings
        if was_running:
            return self.open_camera()
        return True

    def _check_v4l2_controls(self) -> None:
        """Check if V4L2 exposure/gain controls are supported."""
        if self._capture is None or not self._capture.isOpened():
            return

        try:
            import subprocess
            device_path = f"/dev/video{self._settings.device_index}"
            result = subprocess.run(
                ["v4l2-ctl", "--device", device_path, "--list-ctrls"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                controls = result.stdout.lower()
                self._exposure_supported = "exposure_absolute" in controls or "exposure_auto" in controls
                self._gain_supported = "gain" in controls
                if self._exposure_supported or self._gain_supported:
                    logger.info(f"V4L2 controls: exposure={self._exposure_supported}, gain={self._gain_supported}")
        except Exception as e:
            logger.debug(f"Could not check V4L2 controls: {e}")
            self._exposure_supported = False
            self._gain_supported = False

    def _compute_brightness(self, frame: np.ndarray) -> float:
        """Compute mean brightness of frame (handles YUYV, BGR, grayscale)."""
        if frame is None or frame.size == 0:
            return 0.0

        # YUYV format (2 bytes per pixel, interleaved YUV)
        if len(frame.shape) == 2 and frame.shape[1] == self._settings.width * 2:
            # Extract Y channel from YUYV (every other byte starting from 0)
            y_channel = frame[:, 0::2].astype(np.float32)
            return float(np.mean(y_channel))

        # BGR format - convert to grayscale
        if len(frame.shape) == 3:
            if frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return float(np.mean(gray))
            elif frame.shape[2] == 1:
                return float(np.mean(frame))

        # Fallback
        return float(np.mean(frame))

    def _adjust_exposure_gain(self, brightness: float) -> None:
        """Adjust exposure/gain via V4L2 to maintain target brightness."""
        if not self._settings.auto_exposure:
            return

        if self._capture is None or not self._capture.isOpened():
            return

        # Throttle adjustments
        current_time = time.time()
        if current_time - self._last_adjustment_time < self._settings.adjustment_interval:
            return

        # Update brightness history
        self._brightness_history.append(brightness)
        if len(self._brightness_history) > self._settings.brightness_history_size:
            self._brightness_history.pop(0)

        # Need enough history for stable adjustment
        if len(self._brightness_history) < 3:
            return

        avg_brightness = float(np.mean(self._brightness_history))
        target = float(self._settings.target_brightness)
        tolerance = float(self._settings.brightness_tolerance)

        # Check if adjustment needed
        if abs(avg_brightness - target) <= tolerance:
            return

        # Calculate adjustment
        error = target - avg_brightness
        error_ratio = error / target  # Normalized error

        device_path = f"/dev/video{self._settings.device_index}"

        try:
            import subprocess

            # Try adjusting exposure first (more natural than gain)
            if self._exposure_supported:
                # Get current exposure
                result = subprocess.run(
                    ["v4l2-ctl", "--device", device_path, "--get-ctrl=exposure_absolute"],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    try:
                        current_exp = int(result.stdout.strip().split(':')[-1].strip())
                        # Adjust exposure based on error (inverse relationship)
                        new_exp = int(current_exp * (1.0 - error_ratio * 0.3))  # 30% max change
                        new_exp = max(self._settings.exposure_min,
                                     min(self._settings.exposure_max, new_exp))

                        subprocess.run(
                            ["v4l2-ctl", "--device", device_path,
                             f"--set-ctrl=exposure_absolute={new_exp}"],
                            capture_output=True, timeout=1
                        )
                        self._current_exposure = new_exp
                        self._last_adjustment_time = current_time
                        logger.debug(f"Adjusted exposure: {current_exp} -> {new_exp} "
                                   f"(brightness: {avg_brightness:.1f}, target: {target})")
                        return
                    except (ValueError, IndexError):
                        pass

            # Fallback to gain adjustment
            if self._gain_supported:
                result = subprocess.run(
                    ["v4l2-ctl", "--device", device_path, "--get-ctrl=gain"],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    try:
                        current_gain = int(result.stdout.strip().split(':')[-1].strip())
                        new_gain = int(current_gain + error_ratio * 10)  # Small gain adjustment
                        new_gain = max(self._settings.gain_min,
                                      min(self._settings.gain_max, new_gain))

                        subprocess.run(
                            ["v4l2-ctl", "--device", device_path,
                             f"--set-ctrl=gain={new_gain}"],
                            capture_output=True, timeout=1
                        )
                        self._current_gain = new_gain
                        self._last_adjustment_time = current_time
                        logger.debug(f"Adjusted gain: {current_gain} -> {new_gain} "
                                   f"(brightness: {avg_brightness:.1f}, target: {target})")
                        return
                    except (ValueError, IndexError):
                        pass

        except Exception as e:
            logger.debug(f"Exposure/gain adjustment failed: {e}")

    def get_brightness_stats(self) -> dict:
        """Get current brightness monitoring statistics."""
        return {
            "current_brightness": self._brightness_history[-1] if self._brightness_history else 0.0,
            "avg_brightness": float(np.mean(self._brightness_history)) if self._brightness_history else 0.0,
            "target_brightness": self._settings.target_brightness,
            "history_size": len(self._brightness_history),
            "exposure_supported": self._exposure_supported,
            "gain_supported": self._gain_supported,
            "current_exposure": self._current_exposure,
            "current_gain": self._current_gain,
            "auto_exposure_enabled": self._settings.auto_exposure,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_camera()


def diagnose_camera_issues() -> dict:
    """
    Run comprehensive camera diagnostics.

    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {
        "video_devices": [],
        "permissions": {},
        "v4l2_available": False,
        "opencv_version": cv2.__version__,
        "errors": []
    }

    # Check video devices
    for i in range(20):
        path = f"/dev/video{i}"
        if os.path.exists(path):
            stat = os.stat(path)
            diagnostics["video_devices"].append({
                "path": path,
                "readable": os.access(path, os.R_OK),
                "writable": os.access(path, os.W_OK),
                "mode": oct(stat.st_mode)
            })

    # Check V4L2 tools
    try:
        import subprocess
        result = subprocess.run(["v4l2-ctl", "--list-devices"],
                              capture_output=True, text=True, timeout=5)
        diagnostics["v4l2_available"] = result.returncode == 0
        if result.returncode == 0:
            diagnostics["v4l2_devices"] = result.stdout
    except Exception as e:
        diagnostics["errors"].append(f"v4l2-ctl check failed: {e}")

    # Check user groups
    try:
        import subprocess
        result = subprocess.run(["groups"], capture_output=True, text=True)
        diagnostics["user_groups"] = result.stdout.strip().split()
        diagnostics["in_video_group"] = "video" in diagnostics["user_groups"]
    except Exception:
        pass

    return diagnostics


if __name__ == "__main__":
    # Test camera detection
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("=== Camera Diagnostics ===")
    diag = diagnose_camera_issues()
    print(f"OpenCV version: {diag['opencv_version']}")
    print(f"V4L2 available: {diag['v4l2_available']}")
    print(f"User in video group: {diag.get('in_video_group', 'unknown')}")
    print(f"Video devices found: {len(diag['video_devices'])}")
    for dev in diag['video_devices']:
        print(f"  {dev['path']}: readable={dev['readable']}, writable={dev['writable']}")

    print("\n=== Camera Detection ===")
    manager = CameraManager()
    cameras = manager.detect_cameras()

    if cameras:
        print(f"\nFound {len(cameras)} camera(s):")
        for cam in cameras:
            print(f"  [{cam.index}] {cam}")

        # Test opening first available camera
        available_cams = [c for c in cameras if c.available]
        if available_cams:
            print(f"\nTesting camera {available_cams[0].index}...")
            if manager.open_camera(CameraSettings(device_index=available_cams[0].index)):
                print("Camera opened successfully!")
                ret, frame = manager.read_frame()
                if ret and frame is not None:
                    print(f"Frame captured: {frame.shape}")
                manager.close_camera()
            else:
                print("Failed to open camera")
    else:
        print("No cameras detected")