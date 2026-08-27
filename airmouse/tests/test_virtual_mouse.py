"""
Unit tests for VirtualMouse module.

Tests uinput event generation (requires /dev/uinput).
"""
import sys
sys.path.insert(0, '/home/shubham/airmouse')

import pytest
import os
import time
from airmouse.input.uinput_mouse import VirtualMouse, UInputDeviceConfig


# Check if uinput is available
UINPUT_AVAILABLE = os.path.exists('/dev/uinput') and os.access('/dev/uinput', os.W_OK)


@pytest.mark.skipif(not UINPUT_AVAILABLE, reason="/dev/uinput not available or not writable")
class TestVirtualMouse:
    """Tests for VirtualMouse class (requires uinput)."""

    def setup_method(self):
        """Set up virtual mouse."""
        self.mouse = VirtualMouse(UInputDeviceConfig(name="Air Mouse Test"))
        assert self.mouse.create(), "Failed to create virtual mouse"

    def teardown_method(self):
        """Clean up virtual mouse."""
        self.mouse.destroy()

    def test_device_creation(self):
        """Test virtual mouse device creation."""
        assert self.mouse.is_created()

    def test_movement(self):
        """Test relative mouse movement."""
        # Move right
        self.mouse.move(100, 0)
        time.sleep(0.01)

        # Move left
        self.mouse.move(-50, 0)
        time.sleep(0.01)

        # Move up
        self.mouse.move(0, -50)
        time.sleep(0.01)

        # Move down
        self.mouse.move(0, 50)
        time.sleep(0.01)

    def test_smooth_movement(self):
        """Test smooth movement with steps."""
        self.mouse.move_smooth(100, 100, steps=10, delay=0.001)
        time.sleep(0.05)

    def test_left_click(self):
        """Test left click."""
        self.mouse.left_click()
        time.sleep(0.01)

    def test_right_click(self):
        """Test right click."""
        self.mouse.right_click()
        time.sleep(0.01)

    def test_middle_click(self):
        """Test middle click."""
        self.mouse.middle_click()
        time.sleep(0.01)

    def test_button_down_up(self):
        """Test button press and release separately."""
        self.mouse.button_down(0x110)  # BTN_LEFT
        time.sleep(0.01)
        self.mouse.button_up(0x110)
        time.sleep(0.01)

    def test_scroll_vertical(self):
        """Test vertical scroll."""
        self.mouse.scroll(3)   # Scroll up
        time.sleep(0.01)
        self.mouse.scroll(-3)  # Scroll down
        time.sleep(0.01)

    def test_scroll_horizontal(self):
        """Test horizontal scroll."""
        self.mouse.scroll_horizontal(3)   # Scroll right
        time.sleep(0.01)
        self.mouse.scroll_horizontal(-3)  # Scroll left
        time.sleep(0.01)

    def test_release_all(self):
        """Test emergency release all buttons."""
        self.mouse.button_down(0x110)  # Left
        self.mouse.button_down(0x111)  # Right
        self.mouse.release_all()
        time.sleep(0.01)

    def test_position_tracking(self):
        """Test virtual cursor position tracking."""
        self.mouse.set_position(100, 200)
        pos = self.mouse.get_position()
        assert pos == (100, 200)

        self.mouse.move(50, -30)
        pos = self.mouse.get_position()
        assert pos == (150, 170)

    def test_context_manager(self):
        """Test context manager usage."""
        with VirtualMouse(UInputDeviceConfig(name="Context Test")) as mouse:
            assert mouse.is_created()
            mouse.move(10, 10)
        # Should be destroyed after context

    def test_config_customization(self):
        """Test custom device configuration."""
        config = UInputDeviceConfig(
            name="Custom Mouse",
            vendor_id=0xABCD,
            product_id=0xEF01,
            version=0x0200
        )
        mouse = VirtualMouse(config)
        assert mouse.create()
        mouse.destroy()


class TestUInputDeviceConfig:
    """Tests for UInputDeviceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UInputDeviceConfig()

        assert config.name == "Air Mouse"
        assert config.vendor_id == 0x1234
        assert config.product_id == 0x5678
        assert config.version == 0x0100
        assert config.bustype == 0x03

    def test_custom_config(self):
        """Test custom configuration values."""
        config = UInputDeviceConfig(
            name="Test Mouse",
            vendor_id=0x0001,
            product_id=0x0002,
            version=0x0003,
            bustype=0x05
        )

        assert config.name == "Test Mouse"
        assert config.vendor_id == 0x0001
        assert config.product_id == 0x0002
        assert config.version == 0x0003
        assert config.bustype == 0x05


class TestVirtualMouseWithoutUinput:
    """Tests that work without uinput (mock mode)."""

    def test_create_fails_without_uinput(self):
        """Test that creation fails gracefully without uinput."""
        if UINPUT_AVAILABLE:
            pytest.skip("uinput is available")

        mouse = VirtualMouse()
        result = mouse.create()
        assert result is False

    def test_destroy_without_create(self):
        """Test destroy without create doesn't crash."""
        mouse = VirtualMouse()
        mouse.destroy()  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])