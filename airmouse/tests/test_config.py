"""Tests for Air Mouse Config Module"""

import pytest
import tempfile
from pathlib import Path

from airmouse.config import ConfigManager, ProfileManager, Config, DEFAULT_CONFIG


class TestConfigManager:
    """Test ConfigManager load/save/validate operations."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mgr = ConfigManager(self.temp_dir.name)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_default_config_loaded(self):
        """Test that loading missing config returns defaults."""
        cfg = self.mgr.load()
        assert cfg.camera.width == 1280
        assert cfg.camera.height == 720
        assert cfg.cursor.sensitivity == 0.8
        assert cfg.tracking.use_head_relative is True

    def test_save_and_reload(self):
        """Test saving and reloading config preserves values."""
        cfg = self.mgr.load()
        cfg.cursor.sensitivity = 1.5
        cfg.camera.fps = 60
        assert self.mgr.save(cfg)

        cfg2 = self.mgr.load()
        assert cfg2.cursor.sensitivity == 1.5
        assert cfg2.camera.fps == 60

    def test_validation_pass(self):
        """Test valid config passes validation."""
        cfg = self.mgr.load()
        assert self.mgr.validate(cfg)

    def test_validation_fails_invalid_sensitivity(self):
        """Test validation rejects out-of-bounds sensitivity."""
        cfg = self.mgr.load()
        cfg.cursor.sensitivity = 10.0  # Too high
        assert not self.mgr.validate(cfg)

    def test_validation_fails_invalid_fps(self):
        """Test validation rejects unsupported FPS values."""
        cfg = self.mgr.load()
        cfg.camera.fps = 42  # Not in allowed list
        assert not self.mgr.validate(cfg)

    def test_validation_fails_invalid_threshold_order(self):
        """Test validation enforces hysteresis threshold order."""
        cfg = self.mgr.load()
        # confirm must be <= enter
        cfg.gestures.pinch_confirm_threshold = 0.05
        cfg.gestures.pinch_enter_threshold = 0.04
        assert not self.mgr.validate(cfg)

    def test_config_persists_to_file(self):
        """Test config file is created and contains TOML."""
        cfg = self.mgr.load()
        cfg.cursor.sensitivity = 2.0
        self.mgr.save(cfg)

        config_path = self.mgr.get_config_path()
        assert config_path.exists()
        content = config_path.read_text()
        assert "sensitivity = 2.0" in content


class TestProfileManager:
    """Test ProfileManager create/list/delete/import/export operations."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mgr = ConfigManager(self.temp_dir.name)
        self.pm = ProfileManager(self.mgr)
        self.mgr.load()  # Creates default config file

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_create_profile(self):
        """Test creating a new profile."""
        created = self.pm.create("my_profile", "My test profile")
        assert created is not None
        assert created.exists()
        assert "my_profile.toml" in str(created)

    def test_create_duplicate_fails(self):
        """Test creating duplicate profile fails gracefully."""
        self.pm.create("dup")
        created = self.pm.create("dup")
        assert created is None

    def test_list_profiles(self):
        """Test listing profiles."""
        self.pm.create("p1")
        self.pm.create("p2")
        listed = self.pm.list_profiles()
        names = [p.name for p in listed]
        assert set(names) == {"p1", "p2"}

    def test_load_profile(self):
        """Test loading a profile returns Config."""
        cfg = self.mgr.load()
        cfg.cursor.sensitivity = 3.0
        self.mgr.save(cfg)
        self.pm.create("loaded_profile")

        loaded = self.pm.load_profile("loaded_profile")
        assert loaded is not None
        assert loaded.cursor.sensitivity == 3.0

    def test_export_profile(self):
        """Test exporting profile to external path."""
        self.pm.create("export_me")
        export_path = Path(self.temp_dir.name) / "exported.toml"
        assert self.pm.export("export_me", export_path)
        assert export_path.exists()

    def test_import_profile(self):
        """Test importing profile from external path."""
        export_path = Path(self.temp_dir.name) / "import_me.toml"
        # Create a simple valid TOML
        export_path.write_text("""
[camera]
width = 640
height = 480
""")
        imported = self.pm.import_profile(export_path)
        assert imported is not None
        assert imported.exists()

    def test_delete_profile(self):
        """Test deleting a profile."""
        self.pm.create("to_delete")
        assert self.pm.delete("to_delete")
        listed = self.pm.list_profiles()
        assert len(listed) == 0

    def test_delete_nonexistent_fails(self):
        """Test deleting non-existent profile returns False."""
        assert not self.pm.delete("nonexistent")


class TestDefaultConfig:
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
        assert DEFAULT_CONFIG.tracking.confidence_threshold == 0.75
        assert DEFAULT_CONFIG.tracking.use_head_relative is True
        assert DEFAULT_CONFIG.tracking.virtual_plane_distance == 0.30
        assert DEFAULT_CONFIG.tracking.virtual_plane_width == 0.40
        assert DEFAULT_CONFIG.tracking.virtual_plane_height == 0.25

    def test_hotkey_defaults(self):
        assert DEFAULT_CONFIG.hotkeys.emergency_stop == "Super+Alt+A"
        assert DEFAULT_CONFIG.hotkeys.pause_resume == "Super+Alt+P"
        assert DEFAULT_CONFIG.hotkeys.debug_overlay == "Ctrl+Shift+G"

    def test_privacy_defaults(self):
        assert DEFAULT_CONFIG.privacy.local_only is True
        assert DEFAULT_CONFIG.privacy.telemetry_enabled is False

    def test_startup_defaults(self):
        assert DEFAULT_CONFIG.startup.background_mode is True
        assert DEFAULT_CONFIG.startup.start_on_boot is False

    def test_diagnostics_defaults(self):
        assert DEFAULT_CONFIG.diagnostics.log_level == "INFO"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])