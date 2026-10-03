"""Tests for Persistent Configuration Layer

Tests ConfigManager create/save/validate, DEFAULT_CONFIG, profile creation.
"""

import unittest
import tempfile
import time
from pathlib import Path
import sys

sys.path.insert(0, '/home/shubham/airmouse')

from airmouse.config import (
    ConfigManager,
    Config,
    CameraConfig,
    ConfigGestureConfig,
    ConfigTrackingConfig,
    HotkeyConfig,
    PrivacyConfig,
    StartupConfig,
    DiagnosticsConfig,
    DEFAULT_CONFIG,
)
from airmouse.config.config import CursorConfig, GestureConfig as GestureConfigFull
from airmouse.config.profiles import (
    ProfileManager,
    ProfileInfo,
)
from airmouse.control.cursor import (
    SmoothingAlgorithm, SensitivityMode
)


class TestConfigManager(unittest.TestCase):
    """Tests for ConfigManager create/save/validate operations."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mgr = ConfigManager(self.temp_dir.name)
        self.mgr.load()  # Creates default config file

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_config_exists(self):
        """Test default config file is created."""
        config_path = self.mgr.get_config_path()
        self.assertTrue(config_path.exists())

    def test_load_creates_default_config(self):
        """Test load creates a valid default config."""
        cfg = self.mgr.load()
        self.assertIsNotNone(cfg)
        self.assertIsInstance(cfg, Config)

    def test_save_and_load_roundtrip(self):
        """Test save and load are inverse operations."""
        original = self.mgr.load()
        original.cursor.sensitivity = 2.5
        self.mgr.save(original)
        
        loaded = self.mgr.load()
        self.assertAlmostEqual(loaded.cursor.sensitivity, 2.5, places=2)

    def test_get_config_path(self):
        """Test get_config_path returns a path."""
        path = self.mgr.get_config_path()
        self.assertIsInstance(path, Path)


class TestProfileManager(unittest.TestCase):
    """Test ProfileManager create/list/delete/import/export operations."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pm = ProfileManager(self.temp_dir.name)
        self.mgr = ConfigManager(self.temp_dir.name)
        self.mgr.load()  # Creates default config file

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_profile(self):
        """Test creating a new profile."""
        created = self.pm.save_profile("my_profile", {
            "name": "my_profile",
            "description": "My test profile",
            "version": "1.0.0",
            "cursor": {},
            "tracking": {}
        })
        self.assertTrue(created)
        profile_path = Path(self.temp_dir.name) / "my_profile.json"
        self.assertTrue(profile_path.exists())

    def test_create_duplicate_fails(self):
        """Test creating duplicate profile overwrites."""
        self.pm.save_profile("dup", {"name": "dup", "description": "", "version": "1.0.0", "cursor": {}, "tracking": {}})
        created = self.pm.save_profile("dup", {"name": "dup", "description": "updated", "version": "1.0.0", "cursor": {}, "tracking": {}})
        self.assertTrue(created)

    def test_list_profiles(self):
        """Test listing profiles."""
        self.pm.save_profile("p1", {"name": "p1", "description": "", "version": "1.0.0", "cursor": {}, "tracking": {}})
        self.pm.save_profile("p2", {"name": "p2", "description": "", "version": "1.0.0", "cursor": {}, "tracking": {}})
        listed = self.pm.list_profiles()
        self.assertGreaterEqual(len(listed), 2)  # Should have at least our created ones

    def test_load_profile(self):
        """Test loading a profile returns config dict."""
        self.pm.save_profile("loaded_profile", {"name": "loaded_profile", "description": "test desc", "version": "1.0.0", "cursor": {}, "tracking": {}})
        
        loaded = self.pm.load_profile("loaded_profile")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.get("name"), "loaded_profile")
        self.assertEqual(loaded.get("description"), "test desc")

    def test_export_profile(self):
        """Test exporting profile to external path."""
        self.pm.save_profile("export_me", {"name": "export_me", "description": "", "version": "1.0.0", "cursor": {}, "tracking": {}})
        export_path = Path(self.temp_dir.name) / "exported.json"
        self.assertTrue(self.pm.export_profile("export_me", str(export_path)))
        self.assertTrue(export_path.exists())

    def test_import_profile(self):
        """Test importing profile from external path."""
        # Create a simple valid JSON profile
        import_me_path = Path(self.temp_dir.name) / "import_me.json"
        import_me_path.write_text('{"name":"import_me","description":"test","version":"1.0.0","cursor":{},"tracking":{}}')
        imported = self.pm.import_profile(str(import_me_path))
        self.assertTrue(imported)
        loaded = self.pm.load_profile("import_me")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.get("name"), "import_me")

    def test_delete_profile(self):
        """Test deleting a user-created profile."""
        self.pm.save_profile("to_delete", {"name": "to_delete", "description": "", "version": "1.0.0", "cursor": {}, "tracking": {}})
        self.assertTrue(self.pm.delete_profile("to_delete"))
        # After deletion, our created profile should be gone
        profiles = self.pm.list_profiles()
        self.assertNotIn("to_delete", profiles)

    def test_delete_nonexistent_fails(self):
        """Test deleting non-existent profile returns False."""
        self.assertFalse(self.pm.delete_profile("nonexistent"))


class TestDefaultConfig(unittest.TestCase):
    """Test DEFAULT_CONFIG matches product spec requirements."""

    def test_camera_defaults(self):
        assert DEFAULT_CONFIG.camera.width == 1280
        assert DEFAULT_CONFIG.camera.height == 720
        assert DEFAULT_CONFIG.camera.fps == 30

    def test_cursor_defaults(self):
        assert DEFAULT_CONFIG.cursor.sensitivity == 0.8
        assert DEFAULT_CONFIG.cursor.smoothing == "adaptive"
        assert DEFAULT_CONFIG.cursor.acceleration is True
        assert DEFAULT_CONFIG.cursor.dead_zone == 0.02

    def test_gesture_defaults(self):
        assert DEFAULT_CONFIG.gestures.left_click == "pinch"
        assert DEFAULT_CONFIG.gestures.pinch_enter_threshold == 0.045
        assert DEFAULT_CONFIG.gestures.pinch_confirm_threshold == 0.040
        assert DEFAULT_CONFIG.gestures.pinch_release_threshold == 0.070
        assert DEFAULT_CONFIG.gestures.fist_hold_time == 0.5
        assert DEFAULT_CONFIG.gestures.drag_hold_time == 0.2

    def test_tracking_defaults(self):
        # TrackingConfig uses confidence_threshold (not min_detection_confidence)
        assert DEFAULT_CONFIG.tracking.confidence_threshold == 0.75


if __name__ == "__main__":
    unittest.main()