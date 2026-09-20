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
    ProfileSource,
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
        self.mgr = ConfigManager(self.temp_dir.name)
        self.pm = ProfileManager(self.mgr)
        self.mgr.load()  # Creates default config file

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_profile(self):
        """Test creating a new profile."""
        created = self.pm.create("my_profile", "My test profile")
        self.assertIsNotNone(created)
        self.assertTrue(created.exists())
        self.assertIn("my_profile.json", str(created))

    def test_create_duplicate_fails(self):
        """Test creating duplicate profile fails gracefully."""
        self.pm.create("dup")
        created = self.pm.create("dup")
        self.assertIsNone(created)

    def test_list_profiles(self):
        """Test listing profiles."""
        self.pm.create("p1")
        self.pm.create("p2")
        listed = self.pm.list_profiles()
        names = [p.name for p in listed]
        self.assertGreaterEqual(len(names), 2)  # Should have at least our created ones

    def test_load_profile(self):
        """Test loading a profile returns ProfileConfig."""
        self.pm.create("loaded_profile")
        
        loaded = self.pm.load_profile("loaded_profile")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "loaded_profile")
        self.assertEqual(loaded.description, "")
        self.assertEqual(loaded.source, ProfileSource.APP_SPECIFIC)

    def test_export_profile(self):
        """Test exporting profile to external path."""
        self.pm.create("export_me")
        export_path = Path(self.temp_dir.name) / "exported.json"
        self.assertTrue(self.pm.export_profile("export_me", export_path))
        self.assertTrue(export_path.exists())

    def test_import_profile(self):
        """Test importing profile from external path."""
        # Create a simple valid JSON profile
        import_me_path = Path(self.temp_dir.name) / "import_me.json"
        import_me_path.write_text('{"name":"test_import","description":"test"}')
        imported = self.pm.import_profile(import_me_path)
        self.assertIsNotNone(imported)
        self.assertTrue(imported.exists())
        loaded = self.pm.load_profile("test_import")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "test_import")

    def test_delete_profile(self):
        """Test deleting a user-created profile."""
        self.pm.create("to_delete")
        self.assertTrue(self.pm.delete("to_delete"))
        # After deletion, our created profile should be gone
        # (system profiles remain)
        deleted = self.pm.get_profile("to_delete")
        self.assertIsNone(deleted)

    def test_delete_nonexistent_fails(self):
        """Test deleting non-existent profile returns False."""
        self.assertFalse(self.pm.delete("nonexistent"))


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