"""
Linux Input Manager for Air Mouse

Provides a unified interface for mouse input injection across different
Linux display servers and backends:
- Wayland: Uses virtual keyboard/mouse protocols via ydotool or native Wayland protocols
- X11: Uses XTest extension
- uinput: Direct kernel interface (fallback/headless)

Automatically detects the best available backend.
"""

import os
import subprocess
import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List
import threading
import time

logger = logging.getLogger(__name__)


class InputBackend(Enum):
    """Available input injection backends."""
    WAYLAND = "wayland"
    X11 = "x11"
    UINPUT = "uinput"
    YDOTOL = "ydotool"
    NONE = "none"


class DesktopEnvironment(Enum):
    """Known desktop environments."""
    GNOME = "gnome"
    KDE = "kde"
    COSMIC = "cosmic"
    XFCE = "xfce"
    SWAY = "sway"
    HYPRLAND = "hyprland"
    UNKNOWN = "unknown"


@dataclass
class InputCapabilities:
    """Capabilities of the current input backend."""
    backend: InputBackend
    desktop_env: DesktopEnvironment
    supports_mouse_movement: bool = True
    supports_mouse_buttons: bool = True
    supports_scroll: bool = True
    supports_global_hotkeys: bool = False
    supports_system_tray: bool = False
    requires_daemon: bool = False
    notes: str = ""


class InputBackendBase(ABC):
    """Abstract base class for input backends."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the backend. Returns True on success."""
        pass

    @abstractmethod
    def move(self, dx: int, dy: int) -> bool:
        """Move mouse relatively. Returns True on success."""
        pass

    @abstractmethod
    def move_absolute(self, x: int, y: int) -> bool:
        """Move mouse to absolute position. Returns True on success."""
        pass

    @abstractmethod
    def click(self, button: int) -> bool:
        """Click a mouse button (press + release)."""
        pass

    @abstractmethod
    def button_down(self, button: int) -> bool:
        """Press a mouse button."""
        pass

    @abstractmethod
    def button_up(self, button: int) -> bool:
        """Release a mouse button."""
        pass

    @abstractmethod
    def scroll(self, amount: int) -> bool:
        """Vertical scroll. Positive = up."""
        pass

    @abstractmethod
    def scroll_horizontal(self, amount: int) -> bool:
        """Horizontal scroll. Positive = right."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources."""
        pass


class UInputBackend(InputBackendBase):
    """uinput kernel interface backend (current implementation)."""

    def __init__(self):
        self._virtual_mouse = None
        self._initialized = False

    def is_available(self) -> bool:
        return os.path.exists('/dev/uinput') and os.access('/dev/uinput', os.W_OK)

    def initialize(self) -> bool:
        try:
            from ..input.uinput_mouse import VirtualMouse, UInputDeviceConfig
            self._virtual_mouse = VirtualMouse(UInputDeviceConfig(name="Air Mouse"))
            return self._virtual_mouse.create()
        except Exception as e:
            logger.error(f"Failed to initialize uinput backend: {e}")
            return False

    def move(self, dx: int, dy: int) -> bool:
        if self._virtual_mouse and self._virtual_mouse.is_created():
            self._virtual_mouse.move(dx, dy)
            return True
        return False

    def move_absolute(self, x: int, y: int) -> bool:
        if self._virtual_mouse and self._virtual_mouse.is_created():
            cur_x, cur_y = self._virtual_mouse.get_position()
            self._virtual_mouse.move(x - cur_x, y - cur_y)
            return True
        return False

    def click(self, button: int) -> bool:
        if not self._virtual_mouse or not self._virtual_mouse.is_created():
            return False
        if button == 1:  # Left
            self._virtual_mouse.left_click()
        elif button == 2:  # Middle
            self._virtual_mouse.middle_click()
        elif button == 3:  # Right
            self._virtual_mouse.right_click()
        else:
            return False
        return True

    def button_down(self, button: int) -> bool:
        if not self._virtual_mouse or not self._virtual_mouse.is_created():
            return False
        from ..input.uinput_mouse import BTN_LEFT, BTN_RIGHT, BTN_MIDDLE
        btn_map = {1: BTN_LEFT, 2: BTN_MIDDLE, 3: BTN_RIGHT}
        if button in btn_map:
            self._virtual_mouse.button_down(btn_map[button])
            return True
        return False

    def button_up(self, button: int) -> bool:
        if not self._virtual_mouse or not self._virtual_mouse.is_created():
            return False
        from ..input.uinput_mouse import BTN_LEFT, BTN_RIGHT, BTN_MIDDLE
        btn_map = {1: BTN_LEFT, 2: BTN_MIDDLE, 3: BTN_RIGHT}
        if button in btn_map:
            self._virtual_mouse.button_up(btn_map[button])
            return True
        return False

    def scroll(self, amount: int) -> bool:
        if self._virtual_mouse and self._virtual_mouse.is_created():
            self._virtual_mouse.scroll(amount)
            return True
        return False

    def scroll_horizontal(self, amount: int) -> bool:
        if self._virtual_mouse and self._virtual_mouse.is_created():
            self._virtual_mouse.scroll_horizontal(amount)
            return True
        return False

    def cleanup(self):
        if self._virtual_mouse:
            self._virtual_mouse.destroy()
            self._virtual_mouse = None
        self._initialized = False


class X11Backend(InputBackendBase):
    """X11 XTest extension backend."""

    def __init__(self):
        self._display = None
        self._initialized = False

    def is_available(self) -> bool:
        # Check if we're on X11 and XTest is available
        if os.environ.get('WAYLAND_DISPLAY'):
            return False
        if not os.environ.get('DISPLAY'):
            return False
        # Check for XTest extension
        try:
            import subprocess
            result = subprocess.run(
                ['xdpyinfo', '-queryExtensions'],
                capture_output=True, text=True, timeout=2
            )
            return 'XTEST' in result.stdout
        except Exception:
            return False

    def initialize(self) -> bool:
        try:
            from Xlib import display
            from Xlib.ext import xtest
            self._display = display.Display()
            self._xtest = xtest
            self._initialized = True
            logger.info("X11 backend initialized")
            return True
        except ImportError:
            logger.warning("python3-xlib not installed, X11 backend unavailable")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize X11 backend: {e}")
            return False

    def move(self, dx: int, dy: int) -> bool:
        if not self._initialized or not self._display:
            return False
        try:
            self._xtest.fake_input(self._display, self._display.MotionNotify, x=dx, y=dy)
            self._display.sync()
            return True
        except Exception as e:
            logger.error(f"X11 move failed: {e}")
            return False

    def move_absolute(self, x: int, y: int) -> bool:
        if not self._initialized or not self._display:
            return False
        try:
            # XTest doesn't have absolute motion easily, simulate via relative
            # Get current position first (would need XQueryPointer)
            self._xtest.fake_input(self._display, self._display.MotionNotify, x=x, y=y)
            self._display.sync()
            return True
        except Exception as e:
            logger.error(f"X11 absolute move failed: {e}")
            return False

    def click(self, button: int) -> bool:
        if not self._initialized:
            return False
        try:
            self.button_down(button)
            time.sleep(0.01)
            self.button_up(button)
            return True
        except Exception:
            return False

    def button_down(self, button: int) -> bool:
        if not self._initialized:
            return False
        try:
            self._xtest.fake_input(self._display, self._display.ButtonPress, button)
            self._display.sync()
            return True
        except Exception:
            return False

    def button_up(self, button: int) -> bool:
        if not self._initialized:
            return False
        try:
            self._xtest.fake_input(self._display, self._display.ButtonRelease, button)
            self._display.sync()
            return True
        except Exception:
            return False

    def scroll(self, amount: int) -> bool:
        if not self._initialized:
            return False
        try:
            # X11 uses buttons 4/5 for vertical scroll
            btn = 4 if amount > 0 else 5
            for _ in range(abs(amount)):
                self._xtest.fake_input(self._display, self._display.ButtonPress, btn)
                self._xtest.fake_input(self._display, self._display.ButtonRelease, btn)
            self._display.sync()
            return True
        except Exception:
            return False

    def scroll_horizontal(self, amount: int) -> bool:
        if not self._initialized:
            return False
        try:
            # X11 uses buttons 6/7 for horizontal scroll
            btn = 6 if amount > 0 else 7
            for _ in range(abs(amount)):
                self._xtest.fake_input(self._display, self._display.ButtonPress, btn)
                self._xtest.fake_input(self._display, self._display.ButtonRelease, btn)
            self._display.sync()
            return True
        except Exception:
            return False

    def cleanup(self):
        if self._display:
            self._display.close()
            self._display = None
        self._initialized = False


class YdotoolBackend(InputBackendBase):
    """ydotool backend for Wayland (and X11)."""

    def __init__(self):
        self._ydotool_path = None
        self._ydotoold_running = False
        self._initialized = False

    def is_available(self) -> bool:
        # Check if ydotool is installed
        self._ydotool_path = shutil.which('ydotool')
        if not self._ydotool_path:
            return False

        # Check if ydotoold daemon is running or can be started
        try:
            result = subprocess.run(
                ['ydotool', 'mousemove', '0', '0'],
                capture_output=True, timeout=2
            )
            # If it works (exit code 0 or specific error), ydotool is functional
            return result.returncode == 0 or "not connected" not in result.stderr.lower()
        except Exception:
            return False

    def initialize(self) -> bool:
        if not self.is_available():
            return False

        # Check if ydotoold is running
        try:
            result = subprocess.run(
                ['pidof', 'ydotoold'],
                capture_output=True, timeout=1
            )
            self._ydotoold_running = result.returncode == 0

            if not self._ydotoold_running:
                logger.warning("ydotoold daemon not running. Start with: sudo systemctl start ydotoold")
                logger.warning("Or run: sudo ydotoold &")

            self._initialized = True
            logger.info("ydotool backend initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ydotool backend: {e}")
            return False

    def _run_ydotool(self, *args) -> bool:
        """Run ydotool command."""
        if not self._initialized:
            return False
        try:
            result = subprocess.run(
                [self._ydotool_path, *args],
                capture_output=True, timeout=1
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"ydotool command failed: {e}")
            return False

    def move(self, dx: int, dy: int) -> bool:
        return self._run_ydotool('mousemove', '--', str(dx), str(dy))

    def move_absolute(self, x: int, y: int) -> bool:
        return self._run_ydotool('mousemove', str(x), str(y))

    def click(self, button: int) -> bool:
        btn_map = {1: 'left', 2: 'middle', 3: 'right'}
        if button not in btn_map:
            return False
        return self._run_ydotool('click', btn_map[button])

    def button_down(self, button: int) -> bool:
        btn_map = {1: 'left', 2: 'middle', 3: 'right'}
        if button not in btn_map:
            return False
        return self._run_ydotool('mousedown', btn_map[button])

    def button_up(self, button: int) -> bool:
        btn_map = {1: 'left', 2: 'middle', 3: 'right'}
        if button not in btn_map:
            return False
        return self._run_ydotool('mouseup', btn_map[button])

    def scroll(self, amount: int) -> bool:
        if amount > 0:
            return self._run_ydotool('mousemove', '--', '0', '0', '&&', 'ydotool', 'click', 'wheel_up')
        else:
            return self._run_ydotool('mousemove', '--', '0', '0', '&&', 'ydotool', 'click', 'wheel_down')

    def scroll_horizontal(self, amount: int) -> bool:
        # ydotool doesn't have direct horizontal scroll, use key events
        return False

    def cleanup(self):
        self._initialized = False


class WaylandNativeBackend(InputBackendBase):
    """Native Wayland protocol backend (for future use with wlr-virtual-pointer-unstable-v1)."""

    def __init__(self):
        self._initialized = False
        self._compositor = None

    def is_available(self) -> bool:
        # Check if we're on Wayland and have the required protocols
        if not os.environ.get('WAYLAND_DISPLAY'):
            return False

        # Check for wlr-virtual-pointer-unstable-v1 support
        # This requires compositor support (wlr-based: sway, hyprland, etc.)
        try:
            import subprocess
            # Check if we can query wayland protocols
            result = subprocess.run(
                ['weston-info'],
                capture_output=True, text=True, timeout=2
            )
            return 'wlr-virtual-pointer-unstable-v1' in result.stdout
        except Exception:
            return False

    def initialize(self) -> bool:
        # Native Wayland implementation would require:
        # - wayland-client library
        # - wlr-virtual-pointer-unstable-v1 protocol
        # - Compositor support
        logger.warning("Native Wayland backend not yet implemented")
        return False

    def move(self, dx: int, dy: int) -> bool:
        return False

    def move_absolute(self, x: int, y: int) -> bool:
        return False

    def click(self, button: int) -> bool:
        return False

    def button_down(self, button: int) -> bool:
        return False

    def button_up(self, button: int) -> bool:
        return False

    def scroll(self, amount: int) -> bool:
        return False

    def scroll_horizontal(self, amount: int) -> bool:
        return False

    def cleanup(self):
        pass


class LinuxInputManager:
    """
    Manages mouse input injection across Linux display servers.

    Automatically detects and selects the best available backend:
    1. Wayland native protocols (if compositor supports it)
    2. ydotool (works on Wayland and X11)
    3. X11 XTest (if on X11)
    4. uinput (fallback, works everywhere with /dev/uinput)
    """

    # Backend priority order (higher = preferred)
    BACKEND_PRIORITY = [
        (InputBackend.WAYLAND, 100),
        (InputBackend.YDOTOL, 80),
        (InputBackend.X11, 60),
        (InputBackend.UINPUT, 40),
    ]

    def __init__(self):
        self._backend: Optional[InputBackendBase] = None
        self._backend_type: InputBackend = InputBackend.NONE
        self._capabilities: Optional[InputCapabilities] = None
        self._desktop_env: DesktopEnvironment = DesktopEnvironment.UNKNOWN
        self._lock = threading.Lock()

    def detect_desktop_environment(self) -> DesktopEnvironment:
        """Detect the current desktop environment."""
        # Check XDG_CURRENT_DESKTOP
        xdg_desktop = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
        xdg_session = os.environ.get('XDG_SESSION_TYPE', '').lower()
        desk_session = os.environ.get('DESKTOP_SESSION', '').lower()

        all_desktop = f"{xdg_desktop} {xdg_session} {desk_session}"

        if 'cosmic' in all_desktop:
            return DesktopEnvironment.COSMIC
        elif 'gnome' in all_desktop or 'ubuntu:gnome' in all_desktop:
            return DesktopEnvironment.GNOME
        elif 'kde' in all_desktop or 'plasma' in all_desktop:
            return DesktopEnvironment.KDE
        elif 'xfce' in all_desktop or 'xubuntu' in all_desktop:
            return DesktopEnvironment.XFCE
        elif 'sway' in all_desktop:
            return DesktopEnvironment.SWAY
        elif 'hyprland' in all_desktop:
            return DesktopEnvironment.HYPRLAND
        elif xdg_session == 'wayland':
            return DesktopEnvironment.UNKNOWN  # Wayland but unknown DE
        elif xdg_session == 'x11':
            return DesktopEnvironment.UNKNOWN  # X11 but unknown DE

        return DesktopEnvironment.UNKNOWN

    def detect_best_backend(self) -> Tuple[InputBackend, InputBackendBase]:
        """Detect and return the best available backend."""
        self._desktop_env = self.detect_desktop_environment()
        logger.info(f"Detected desktop environment: {self._desktop_env.value}")
        logger.info(f"Session type: {os.environ.get('XDG_SESSION_TYPE', 'unknown')}")
        logger.info(f"WAYLAND_DISPLAY: {os.environ.get('WAYLAND_DISPLAY', 'not set')}")
        logger.info(f"DISPLAY: {os.environ.get('DISPLAY', 'not set')}")

        # Create backend instances
        backends = {
            InputBackend.WAYLAND: WaylandNativeBackend(),
            InputBackend.YDOTOL: YdotoolBackend(),
            InputBackend.X11: X11Backend(),
            InputBackend.UINPUT: UInputBackend(),
        }

        # Check availability and initialize
        available_backends = []
        for backend_type, priority in self.BACKEND_PRIORITY:
            backend = backends[backend_type]
            if backend.is_available():
                logger.info(f"Backend {backend_type.value} is available")
                if backend.initialize():
                    available_backends.append((backend_type, backend, priority))
                    logger.info(f"Backend {backend_type.value} initialized successfully")
                else:
                    logger.warning(f"Backend {backend_type.value} available but failed to initialize")
            else:
                logger.debug(f"Backend {backend_type.value} not available")

        if not available_backends:
            logger.error("No input backends available!")
            return InputBackend.NONE, None

        # Sort by priority
        available_backends.sort(key=lambda x: x[2], reverse=True)
        best_type, best_backend, _ = available_backends[0]

        logger.info(f"Selected backend: {best_type.value}")
        return best_type, best_backend

    def initialize(self) -> bool:
        """Initialize the input manager with the best available backend."""
        with self._lock:
            if self._backend is not None:
                logger.warning("Input manager already initialized")
                return True

            backend_type, backend = self.detect_best_backend()

            if backend is None:
                self._capabilities = InputCapabilities(
                    backend=InputBackend.NONE,
                    desktop_env=self._desktop_env,
                    supports_mouse_movement=False,
                    supports_mouse_buttons=False,
                    supports_scroll=False,
                    notes="No input backend available"
                )
                return False

            self._backend = backend
            self._backend_type = backend_type

            # Determine capabilities based on backend and desktop
            self._capabilities = self._determine_capabilities()

            logger.info(f"Input manager initialized with {backend_type.value} backend")
            logger.info(f"Capabilities: {self._capabilities}")
            return True

    def _determine_capabilities(self) -> InputCapabilities:
        """Determine capabilities based on backend and desktop environment."""
        caps = InputCapabilities(
            backend=self._backend_type,
            desktop_env=self._desktop_env,
        )

        # All backends support basic mouse operations
        caps.supports_mouse_movement = True
        caps.supports_mouse_buttons = True
        caps.supports_scroll = True

        # Backend-specific capabilities
        if self._backend_type == InputBackend.YDOTOL:
            caps.requires_daemon = True
            caps.supports_global_hotkeys = False  # ydotool doesn't do global hotkeys
            caps.notes = "Requires ydotoold daemon running"

        elif self._backend_type == InputBackend.X11:
            caps.supports_global_hotkeys = True  # Can use XGrabKey
            caps.supports_system_tray = True  # X11 has system tray
            caps.notes = "Running on X11"

        elif self._backend_type == InputBackend.UINPUT:
            caps.supports_global_hotkeys = False  # Kernel level, no hotkeys
            caps.notes = "Kernel-level uinput (works everywhere)"

        elif self._backend_type == InputBackend.WAYLAND:
            caps.supports_global_hotkeys = False  # Wayland security model
            caps.notes = "Native Wayland protocols"

        # Desktop environment specific notes
        if self._desktop_env == DesktopEnvironment.GNOME:
            caps.notes += "; GNOME: may need extension for system tray"
        elif self._desktop_env == DesktopEnvironment.KDE:
            caps.notes += "; KDE: full system tray support"
        elif self._desktop_env == DesktopEnvironment.COSMIC:
            caps.notes += "; COSMIC: system tray supported"

        return caps

    def get_capabilities(self) -> Optional[InputCapabilities]:
        """Get the capabilities of the current backend."""
        return self._capabilities

    def get_backend_type(self) -> InputBackend:
        """Get the current backend type."""
        return self._backend_type

    def get_desktop_environment(self) -> DesktopEnvironment:
        """Get the detected desktop environment."""
        return self._desktop_env

    def move(self, dx: int, dy: int) -> bool:
        """Move mouse relatively."""
        with self._lock:
            if self._backend:
                return self._backend.move(dx, dy)
            return False

    def move_absolute(self, x: int, y: int) -> bool:
        """Move mouse to absolute position."""
        with self._lock:
            if self._backend:
                return self._backend.move_absolute(x, y)
            return False

    def click(self, button: int) -> bool:
        """Click a mouse button."""
        with self._lock:
            if self._backend:
                return self._backend.click(button)
            return False

    def button_down(self, button: int) -> bool:
        """Press a mouse button."""
        with self._lock:
            if self._backend:
                return self._backend.button_down(button)
            return False

    def button_up(self, button: int) -> bool:
        """Release a mouse button."""
        with self._lock:
            if self._backend:
                return self._backend.button_up(button)
            return False

    def scroll(self, amount: int) -> bool:
        """Vertical scroll."""
        with self._lock:
            if self._backend:
                return self._backend.scroll(amount)
            return False

    def scroll_horizontal(self, amount: int) -> bool:
        """Horizontal scroll."""
        with self._lock:
            if self._backend:
                return self._backend.scroll_horizontal(amount)
            return False

    def release_all(self):
        """Release all mouse buttons (emergency stop)."""
        with self._lock:
            if self._backend:
                # Release common buttons
                for btn in [1, 2, 3]:
                    try:
                        self._backend.button_up(btn)
                    except Exception:
                        pass

    def cleanup(self):
        """Clean up resources."""
        with self._lock:
            if self._backend:
                self._backend.cleanup()
                self._backend = None
            self._backend_type = InputBackend.NONE
            self._capabilities = None

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def get_input_manager() -> LinuxInputManager:
    """Get a global input manager instance."""
    # This could be a singleton if needed
    return LinuxInputManager()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Linux Input Manager Test ===")

    manager = LinuxInputManager()

    print(f"\nDesktop Environment: {manager.detect_desktop_environment().value}")
    print(f"Session Type: {os.environ.get('XDG_SESSION_TYPE', 'unknown')}")
    print(f"WAYLAND_DISPLAY: {os.environ.get('WAYLAND_DISPLAY', 'not set')}")
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'not set')}")

    if manager.initialize():
        caps = manager.get_capabilities()
        print(f"\nSelected Backend: {caps.backend.value}")
        print(f"Capabilities:")
        print(f"  Mouse Movement: {caps.supports_mouse_movement}")
        print(f"  Mouse Buttons: {caps.supports_mouse_buttons}")
        print(f"  Scroll: {caps.supports_scroll}")
        print(f"  Global Hotkeys: {caps.supports_global_hotkeys}")
        print(f"  System Tray: {caps.supports_system_tray}")
        print(f"  Requires Daemon: {caps.requires_daemon}")
        print(f"  Notes: {caps.notes}")

        # Test basic movement
        print("\nTesting mouse movement (5 seconds)...")
        import time
        for i in range(5):
            manager.move(10, 0)
            time.sleep(0.5)

        manager.cleanup()
        print("Test complete")
    else:
        print("Failed to initialize any backend")