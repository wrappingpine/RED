"""
Linux uinput Virtual Mouse Module for Air Mouse

Creates a virtual mouse device using Linux uinput kernel interface.
No external dependencies (uses ctypes for direct syscalls).

Requires:
- /dev/uinput exists and is writable
- User in 'input' group or appropriate permissions
"""

import os
import struct
import fcntl
import logging
import time
from typing import Optional
from dataclasses import dataclass
from enum import IntEnum


logger = logging.getLogger(__name__)


# uinput constants (from linux/uinput.h)
UI_DEV_CREATE = 0x5501
UI_DEV_DESTROY = 0x5502
UI_SET_EVBIT = 0x40045564
UI_SET_KEYBIT = 0x40045565
UI_SET_RELBIT = 0x40045566
UI_SET_ABSBIT = 0x40045567

# Event types
EV_SYN = 0x00
EV_KEY = 0x01
EV_REL = 0x02
EV_ABS = 0x03

# Synchronization
SYN_REPORT = 0

# Relative axes
REL_X = 0x00
REL_Y = 0x01
REL_WHEEL = 0x08
REL_HWHEEL = 0x09

# Mouse buttons
BTN_LEFT = 0x110
BTN_RIGHT = 0x111
BTN_MIDDLE = 0x112
BTN_SIDE = 0x113
BTN_EXTRA = 0x114

# UInput device structure constants
UINPUT_MAX_NAME_SIZE = 80
ABS_CNT = 64

# struct uinput_user_dev size:
# name[80] + input_id[8] + ff_effects_max[4] + absmax[256] + absmin[256] + absfuzz[256] + absflat[256]
UINPUT_DEV_SIZE = UINPUT_MAX_NAME_SIZE + 8 + 4 + 4 * ABS_CNT * 4  # = 1116 bytes


@dataclass
class UInputDeviceConfig:
    """Configuration for virtual mouse device."""
    name: str = "Air Mouse"
    vendor_id: int = 0x1234
    product_id: int = 0x5678
    version: int = 0x0100
    bustype: int = 0x03  # BUS_USB


class VirtualMouse:
    """
    Virtual mouse using Linux uinput.

    Provides mouse movement, clicks, and scroll events
    through a kernel-level virtual device.
    """

    def __init__(self, config: Optional[UInputDeviceConfig] = None):
        self.config = config or UInputDeviceConfig()
        self._fd: Optional[int] = None
        self._device_path: Optional[str] = None
        self._created = False
        self._buttons_pressed = set()  # Track which buttons are pressed
        # Virtual cursor position (maintained locally since uinput doesn't expose it)
        self._cursor_x: int = 0
        self._cursor_y: int = 0

    def create(self) -> bool:
        """
        Create the virtual mouse device.

        Returns:
            True if device created successfully
        """
        if self._created:
            logger.warning("Virtual mouse already created")
            return True

        try:
            # Open uinput device
            self._fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
            logger.info("Opened /dev/uinput")

            # Enable event types
            self._ioctl(UI_SET_EVBIT, EV_SYN)
            self._ioctl(UI_SET_EVBIT, EV_KEY)
            self._ioctl(UI_SET_EVBIT, EV_REL)

            # Enable relative axes
            self._ioctl(UI_SET_RELBIT, REL_X)
            self._ioctl(UI_SET_RELBIT, REL_Y)
            self._ioctl(UI_SET_RELBIT, REL_WHEEL)
            self._ioctl(UI_SET_RELBIT, REL_HWHEEL)

            # Enable mouse buttons
            self._ioctl(UI_SET_KEYBIT, BTN_LEFT)
            self._ioctl(UI_SET_KEYBIT, BTN_RIGHT)
            self._ioctl(UI_SET_KEYBIT, BTN_MIDDLE)
            self._ioctl(UI_SET_KEYBIT, BTN_SIDE)
            self._ioctl(UI_SET_KEYBIT, BTN_EXTRA)

            # Create device structure (full uinput_user_dev)
            name_bytes = self.config.name.encode('utf-8')[:UINPUT_MAX_NAME_SIZE - 1]
            name_padded = name_bytes + b'\x00' * (UINPUT_MAX_NAME_SIZE - len(name_bytes))

            # Build full uinput_user_dev structure (1116 bytes)
            dev_data = bytearray(UINPUT_DEV_SIZE)
            dev_data[0:len(name_padded)] = name_padded

            # Set device ID (offset 80)
            struct.pack_into("HHHH", dev_data, 80,
                           self.config.bustype,
                           self.config.vendor_id,
                           self.config.product_id,
                           self.config.version)

            # ff_effects_max at offset 88 (set to 0)
            # absmax, absmin, absfuzz, absflat at offsets 92+ (all zeros for relative device)

            # Write device info
            os.write(self._fd, dev_data)

            # Create device (UI_DEV_CREATE takes no argument)
            fcntl.ioctl(self._fd, UI_DEV_CREATE, 0)
            self._created = True

            logger.info(f"Virtual mouse created: {self.config.name}")
            return True

        except PermissionError:
            logger.error("Permission denied: cannot access /dev/uinput. "
                        "Add user to 'input' group: sudo usermod -aG input $USER")
            self._cleanup()
            return False
        except FileNotFoundError:
            logger.error("/dev/uinput not found. Load kernel module: sudo modprobe uinput")
            self._cleanup()
            return False
        except OSError as e:
            logger.error(f"Failed to create virtual mouse: {e}")
            self._cleanup()
            return False

    def _ioctl(self, request: int, arg: int) -> int:
        """Perform ioctl syscall."""
        if self._fd is None:
            raise RuntimeError("Device not open")
        return fcntl.ioctl(self._fd, request, arg)

    def _write_event(self, ev_type: int, code: int, value: int):
        """Write an input event."""
        if self._fd is None:
            return

        # struct input_event { timeval time; unsigned short type; unsigned short code; unsigned int value; }
        # Use 0 for timestamp (kernel will fill in)
        event = struct.pack("@LLHHi",
                          0,      # tv_sec
                          0,      # tv_usec
                          ev_type, code, value)
        try:
            os.write(self._fd, event)
        except OSError as e:
            logger.warning(f"Failed to write event: {e}")

    def _sync(self):
        """Send synchronization event."""
        self._write_event(EV_SYN, SYN_REPORT, 0)

    def move(self, dx: int, dy: int):
        """
        Move mouse cursor relatively.

        Args:
            dx: X movement (positive = right)
            dy: Y movement (positive = down)
        """
        if dx != 0:
            self._write_event(EV_REL, REL_X, dx)
            self._cursor_x += dx
        if dy != 0:
            self._write_event(EV_REL, REL_Y, dy)
            self._cursor_y += dy
        self._sync()

    def get_position(self) -> tuple:
        """
        Get current virtual cursor position.

        Returns:
            Tuple of (x, y) cursor position
        """
        return (self._cursor_x, self._cursor_y)

    def set_position(self, x: int, y: int):
        """
        Set virtual cursor position (for debugging/reset).

        Args:
            x: X position
            y: Y position
        """
        self._cursor_x = x
        self._cursor_y = y

    def move_smooth(self, dx: int, dy: int, steps: int = 5, delay: float = 0.001):
        """
        Move mouse in small steps for smoother motion.

        Args:
            dx: Total X movement
            dy: Total Y movement
            steps: Number of steps to divide movement
            delay: Delay between steps (seconds)
        """
        if steps <= 1:
            self.move(dx, dy)
            return

        step_x = dx / steps
        step_y = dy / steps

        for i in range(steps):
            # Use integer steps, accumulate remainder
            move_x = int(round(step_x))
            move_y = int(round(step_y))
            self.move(move_x, move_y)
            if delay > 0:
                time.sleep(delay)

    def left_click(self):
        """Perform left click (press + release)."""
        self.button_down(BTN_LEFT)
        self.button_up(BTN_LEFT)

    def right_click(self):
        """Perform right click (press + release)."""
        self.button_down(BTN_RIGHT)
        self.button_up(BTN_RIGHT)

    def middle_click(self):
        """Perform middle click (press + release)."""
        self.button_down(BTN_MIDDLE)
        self.button_up(BTN_MIDDLE)

    def button_down(self, button: int):
        """Press a mouse button."""
        if button not in self._buttons_pressed:
            self._write_event(EV_KEY, button, 1)
            self._sync()
            self._buttons_pressed.add(button)

    def button_up(self, button: int):
        """Release a mouse button."""
        if button in self._buttons_pressed:
            self._write_event(EV_KEY, button, 0)
            self._sync()
            self._buttons_pressed.discard(button)

    def scroll(self, amount: int):
        """
        Scroll vertically.

        Args:
            amount: Scroll amount (positive = up, negative = down)
        """
        self._write_event(EV_REL, REL_WHEEL, amount)
        self._sync()

    def scroll_horizontal(self, amount: int):
        """
        Scroll horizontally.

        Args:
            amount: Scroll amount (positive = right, negative = left)
        """
        self._write_event(EV_REL, REL_HWHEEL, amount)
        self._sync()

    def release_all(self):
        """Release all mouse buttons (emergency stop)."""
        for btn in list(self._buttons_pressed):
            self._write_event(EV_KEY, btn, 0)
        self._sync()
        if self._buttons_pressed:
            logger.info("All buttons released")
        self._buttons_pressed.clear()

    def destroy(self):
        """Destroy the virtual mouse device."""
        if self._fd is not None:
            try:
                self.release_all()
                fcntl.ioctl(self._fd, UI_DEV_DESTROY, 0)
                logger.info("Virtual mouse destroyed")
            except Exception as e:
                logger.warning(f"Error destroying device: {e}")
            finally:
                self._cleanup()

    def _cleanup(self):
        """Clean up file descriptor."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
        self._created = False

    def is_created(self) -> bool:
        """Check if device is created."""
        return self._created

    def __enter__(self):
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()


def test_uinput():
    """Test uinput virtual mouse."""
    print("Testing Virtual Mouse...")
    print("This will move the cursor and click. Press Ctrl+C to stop.")

    mouse = VirtualMouse(UInputDeviceConfig(name="Air Mouse Test"))

    if not mouse.create():
        print("Failed to create virtual mouse")
        return False

    print("Virtual mouse created. Testing in 2 seconds...")
    time.sleep(2)

    try:
        # Test movement
        print("Moving right...")
        mouse.move(100, 0)
        time.sleep(0.5)

        print("Moving down...")
        mouse.move(0, 100)
        time.sleep(0.5)

        print("Moving left...")
        mouse.move(-100, 0)
        time.sleep(0.5)

        print("Moving up...")
        mouse.move(0, -100)
        time.sleep(0.5)

        # Test clicks
        print("Left click...")
        mouse.left_click()
        time.sleep(0.5)

        print("Right click...")
        mouse.right_click()
        time.sleep(0.5)

        # Test scroll
        print("Scroll up...")
        mouse.scroll(3)
        time.sleep(0.5)

        print("Scroll down...")
        mouse.scroll(-3)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        mouse.destroy()
        print("Virtual mouse destroyed")

    return True


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    test_uinput()