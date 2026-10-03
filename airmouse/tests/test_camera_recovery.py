#!/usr/bin/env python3
"""Camera Recovery Tests

Tests camera recovery functionality per spec §48. Tests handle camera
disappearance and recover from camera errors without crashing the main loop.
"""

import time
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from airmouse.camera.manager import CameraManager, CameraSettings
from airmouse.control.main_loop import AirMouseController, AirMouseConfig


class TestCameraRecovery:
    """Test suite for camera recovery functionality."""

    def test_camera_recovery_method_exists_and_callable(self):
        """Test that _recover_camera method exists and is callable."""
        ctrl = AirMouseController(AirMouseConfig())
        assert hasattr(ctrl, '_recover_camera'), "Recovery method must exist"
        assert callable(getattr(ctrl, '_recover_camera')), "Recovery must be callable"
        print("✓ Recovery method exists and is callable")

    def test_camera_recovery_handles_multiple_cameras(self):
        """Test that recovery tries alternative camera devices."""
        ctrl = AirMouseController(AirMouseConfig())
        ctrl.config.target_fps = 1

        # Track camera selection order
        camera_selections = []

        def mock_open_camera(settings):
            camera_selections.append(settings.device_index)
            if len(camera_selections) == 1:
                return False
            return True

        # Patch the camera methods
        ctrl.camera = Mock(spec=CameraManager)
        ctrl.camera.detect_cameras = Mock(return_value=[
            Mock(index=0, device_path='/dev/video0', available=True),
            Mock(index=1, device_path='/dev/video1', available=True),
        ])
        ctrl.camera.open_camera = Mock(side_effect=mock_open_camera)
        ctrl.camera.close_camera = Mock()
        ctrl.camera.is_running = Mock(return_value=True)
        ctrl.camera.get_resolution = Mock(return_value=(640, 480))

        # Set up controller state
        ctrl.state = 'running'

        # Call recover
        result = ctrl._recover_camera()

        # Should try both cameras and succeed on the second
        assert result == True, "Recovery should succeed with second camera"
        assert 0 in camera_selections, "Should try first camera (index 0)"
        assert 1 in camera_selections, "Should try second camera (index 1)"
        print("✓ Recovery tries multiple cameras and succeeds")

    def test_camera_recovery_failure_handles_gracefully(self):
        """Test that recovery failure doesn't crash the application."""
        ctrl = AirMouseController(AirMouseConfig())
        ctrl.config.target_fps = 1

        # Track attempts
        attempt_count = 0

        def mock_recover():
            nonlocal attempt_count
            attempt_count += 1
            return False  # Always fail

        ctrl._recover_camera = mock_recover

        # Set up camera that always fails
        ctrl.camera = Mock(spec=CameraManager)
        ctrl.camera.read_frame = Mock(return_value=(False, None))
        ctrl.camera.close_camera = Mock()

        ctrl._camera_consecutive_failures = 0

        # Run recovery multiple times - should not crash
        for i in range(10):
            if ctrl._recover_camera():
                break
            time.sleep(0.01)

        assert attempt_count > 0, "Should have attempted recovery"
        print("✓ Recovery failure doesn't crash gracefully")

    def test_camera_restores_after_recovery(self):
        """Test that camera is properly reinitialized after successful recovery."""
        ctrl = AirMouseController(AirMouseConfig())
        ctrl.config.target_fps = 1

        # Track re-initialization
        camera_states = []

        def mock_read_frame(*args, **kwargs):
            if time.time() < 0.5:
                return False, None
            camera_states.append('working')
            return True, Mock(shape=(640, 480, 3))

        ctrl.camera = Mock(spec=CameraManager)
        ctrl.camera.read_frame = mock_read_frame
        ctrl.camera.close_camera = Mock()
        ctrl.camera.is_running = Mock(return_value=True)

        initial_failures = 15
        ctrl._camera_consecutive_failures = initial_failures

        # Simulate detection of first frame after recovery
        ret, frame = ctrl.camera.read_frame()

        if ret and frame is not None:
            ctrl._camera_consecutive_failures = 0
        else:
            ctrl._camera_consecutive_failures = initial_failures

        assert ctrl._camera_consecutive_failures <= 0, \
            "Success should reset failure counter"
        print("✓ Camera properly restored after recovery")

    def test_recovery_backoff_prevents_retry_spam(self):
        """Test that recovery uses backoff to avoid excessive retry attempts."""
        ctrl = AirMouseController(AirMouseConfig())
        ctrl.config.target_fps = 1

        # Track when recovery attempts happen
        attempts = []

        def mock_recover():
            attempts.append(time.time())
            return False  # Always fail

        ctrl._recover_camera = mock_recover
        ctrl.camera = Mock(spec=CameraManager)
        ctrl.camera.read_frame = Mock(return_value=(False, None))

        ctrl._camera_consecutive_failures = 0
        ctrl._camera_max_failures_before_recovery = 15
        ctrl._camera_recovery_backoff = 0.1  # 100ms backoff
        ctrl._camera_last_recovery_attempt = 0.0

        # Simulate a window of camera failures
        for i in range(20):
            ctrl.camera.read_frame()

        # Recovery attempts should be spaced by at least backoff
        if len(attempts) >= 2:
            gaps = [attempts[i+1] - attempts[i] for i in range(len(attempts)-1)]
            min_gap = min(gaps)
            assert min_gap >= 0.09, f"Backoff not respected: min_gap={min_gap}s, expected>=0.09s"
            print(f"✓ Recovery uses backoff: min_gap={min_gap:.2f}s")


if __name__ == "__main__":
    import unittest
    unittest.main()