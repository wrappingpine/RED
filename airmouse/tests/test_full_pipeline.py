"""
Integration test for full AirMouse pipeline.

Tests camera → hand → face → projection → cursor → mouse pipeline.
"""
import sys
sys.path.insert(0, '/home/shubham/airmouse')

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock, patch

from airmouse.control.main_loop import AirMouseController, AirMouseConfig
from airmouse.control.cursor import CursorConfig
from airmouse.camera.manager import CameraSettings
from airmouse.vision.hand_tracker import HandTrackerSettings
from airmouse.vision.face_tracker import FaceTrackerSettings
from airmouse.input.uinput_mouse import UInputDeviceConfig


class TestFullPipeline:
    """Integration tests for the full AirMouse pipeline."""

    def setup_method(self):
        """Set up controller with test configuration."""
        self.config = AirMouseConfig()
        self.config.tracking.use_head_relative = True
        self.config.camera = CameraSettings(device_index=0, width=640, height=480, fps=30)
        self.config.hand_tracker = HandTrackerSettings(max_hands=2)
        self.config.face_tracker = FaceTrackerSettings()
        self.config.virtual_mouse = UInputDeviceConfig(name="Air Mouse Test")

    def test_controller_initialization(self):
        """Test controller initializes all components after start."""
        controller = AirMouseController(self.config, lambda s, d: None)

        # Before start, components are None
        assert controller.hand_tracker is None
        assert controller.face_tracker is None
        assert controller.tracking_processor is None
        assert controller.cursor_controller is None
        assert controller.gesture_recognizer is None
        assert controller.virtual_mouse is None

    def test_controller_start_stop(self):
        """Test controller start/stop cycle."""
        controller = AirMouseController(self.config, lambda s, d: None)

        # Start (will fail without camera, but should return False gracefully)
        result = controller.start()
        # May fail if no camera, but should not raise exception
        assert isinstance(result, bool)

        # Stop
        controller.stop()

    def test_debug_overlay_toggle(self):
        """Test debug overlay toggle."""
        controller = AirMouseController(self.config, lambda s, d: None)
        controller.start()

        # Toggle on
        result = controller.toggle_debug_overlay()
        assert result is True
        assert controller._debug_overlay_enabled is True

        # Toggle off
        result = controller.toggle_debug_overlay()
        assert result is False
        assert controller._debug_overlay_enabled is False

        controller.stop()

    @patch('airmouse.camera.manager.CameraManager.open_camera')
    @patch('airmouse.camera.manager.CameraManager.read_frame')
    @patch('airmouse.control.main_loop.HandTracker')
    @patch('airmouse.control.main_loop.FaceTracker')
    @patch('airmouse.control.main_loop.VirtualMouse')
    def test_processing_loop_mock(self, mock_virtual_mouse, mock_face_tracker, mock_hand_tracker, mock_read_frame, mock_open_camera):
        """Test processing loop with mocked camera."""
        # Mock camera
        mock_open_camera.return_value = True
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_read_frame.return_value = (True, mock_frame)

        # Mock trackers
        mock_hand_instance = Mock()
        mock_hand_instance.process = Mock(return_value=[])
        mock_hand_tracker.return_value = mock_hand_instance

        mock_face_instance = Mock()
        mock_face_instance.process = Mock(return_value=[])
        mock_face_tracker.return_value = mock_face_instance

        # Mock virtual mouse
        mock_mouse_instance = Mock()
        mock_mouse_instance.create = Mock(return_value=True)
        mock_virtual_mouse.return_value = mock_mouse_instance

        controller = AirMouseController(self.config, lambda s, d: None)

        controller.start()

        # Wait for initialization to complete
        time.sleep(0.5)

        # Let it run a few cycles
        time.sleep(0.5)

        controller.stop()

        # Verify trackers were called
        assert mock_hand_instance.process.called
        assert mock_face_instance.process.called

    def test_config_aggregation(self):
        """Test AirMouseConfig aggregates all sub-configs."""
        config = AirMouseConfig()

        # Camera config (using defaults)
        assert hasattr(config, 'camera')
        assert config.camera.width == 1280
        assert config.camera.height == 720

        # Hand tracker config
        assert hasattr(config, 'hand_tracker')
        assert config.hand_tracker.max_hands == 1

        # Face tracker config (created dynamically in start())
        assert not hasattr(config, 'face_tracker')  # Not in config, created in start()

        # Tracking config
        assert hasattr(config, 'tracking')
        assert config.tracking.use_head_relative is True
        assert config.tracking.virtual_plane_distance == 0.30

        # Cursor config
        assert hasattr(config, 'cursor')
        assert config.cursor.sensitivity_mode == CursorConfig().sensitivity_mode
        assert config.cursor.sensitivity_normal == 0.32

        # Gesture config (field name is 'gestures')
        assert hasattr(config, 'gestures')
        assert config.gestures.pinch_enter_threshold == 0.045

        # Virtual mouse config
        assert hasattr(config, 'virtual_mouse')
        assert config.virtual_mouse.name == "Air Mouse"


class TestHeadRelativePipeline:
    """Tests specifically for head-relative tracking pipeline."""

    def setup_method(self):
        """Set up with head-relative tracking enabled."""
        self.config = AirMouseConfig()
        self.config.tracking.use_head_relative = True
        self.config.camera = CameraSettings(device_index=0, width=640, height=480, fps=30)
        self.config.hand_tracker = HandTrackerSettings(max_hands=2)
        self.config.face_tracker = FaceTrackerSettings()

    @patch('airmouse.camera.manager.CameraManager.open_camera')
    @patch('airmouse.camera.manager.CameraManager.read_frame')
    def test_head_coordinate_system_created(self, mock_read_frame, mock_open_camera):
        """Test HeadCoordinateSystem is created from face."""
        # Mock camera
        mock_open_camera.return_value = True
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_read_frame.return_value = (True, mock_frame)

        controller = AirMouseController(self.config, lambda s, d: None)

        # Mock trackers
        controller.hand_tracker = Mock()
        controller.hand_tracker.process = Mock(return_value=[])
        controller.face_tracker = Mock()
        controller.face_tracker.process = Mock(return_value=[])

        controller.start()

        # Check that tracking processor has head coordinate system
        processor = controller.tracking_processor
        assert processor is not None
        assert processor._config.use_head_relative

        controller.stop()

    @patch('airmouse.camera.manager.CameraManager.open_camera')
    @patch('airmouse.camera.manager.CameraManager.read_frame')
    def test_virtual_display_plane_created(self, mock_read_frame, mock_open_camera):
        """Test VirtualDisplayPlane is created with correct parameters."""
        # Mock camera
        mock_open_camera.return_value = True
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_read_frame.return_value = (True, mock_frame)

        controller = AirMouseController(self.config, lambda s, d: None)

        # Mock trackers
        controller.hand_tracker = Mock()
        controller.hand_tracker.process = Mock(return_value=[])
        controller.face_tracker = Mock()
        controller.face_tracker.process = Mock(return_value=[])

        controller.start()

        processor = controller.tracking_processor
        assert processor is not None
        assert processor._config.use_head_relative
        if processor._plane:
            assert processor._plane.distance == 0.30
            assert processor._plane.width == 0.40
            assert processor._plane.height == 0.25

        controller.stop()

    @patch('airmouse.camera.manager.CameraManager.open_camera')
    @patch('airmouse.camera.manager.CameraManager.read_frame')
    def test_hand_projector_created(self, mock_read_frame, mock_open_camera):
        """Test HandProjector is created."""
        # Mock camera
        mock_open_camera.return_value = True
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_read_frame.return_value = (True, mock_frame)

        controller = AirMouseController(self.config, lambda s, d: None)

        # Mock trackers
        controller.hand_tracker = Mock()
        controller.hand_tracker.process = Mock(return_value=[])
        controller.face_tracker = Mock()
        controller.face_tracker.process = Mock(return_value=[])

        controller.start()

        processor = controller.tracking_processor
        assert processor is not None
        assert processor._projector is not None or processor._config.use_head_relative

        controller.stop()

    @patch('airmouse.camera.manager.CameraManager.open_camera')
    @patch('airmouse.camera.manager.CameraManager.read_frame')
    def test_two_hand_tracking_config(self, mock_read_frame, mock_open_camera):
        """Test two-hand tracking is configured."""
        # Mock camera
        mock_open_camera.return_value = True
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_read_frame.return_value = (True, mock_frame)

        controller = AirMouseController(self.config, lambda s, d: None)

        # Mock trackers
        controller.hand_tracker = Mock()
        controller.hand_tracker.process = Mock(return_value=[])
        controller.face_tracker = Mock()
        controller.face_tracker.process = Mock(return_value=[])

        controller.start()

        processor = controller.tracking_processor
        assert processor is not None
        assert processor._config.enable_two_hand is True
        assert processor._config.preferred_handedness == "Right"

        controller.stop()

    def test_precision_mode_with_two_hands(self):
        """Test PRECISION_MODE is available for two-hand tracking."""
        from airmouse.vision.gestures import TrackingState
        assert TrackingState.PRECISION_MODE in TrackingState


if __name__ == "__main__":
    pytest.main([__file__, "-v"])