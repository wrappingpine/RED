"""
Global Hotkey Manager for Air Mouse

Provides system-wide hotkey registration:
- X11: Uses XGrabKey via python-xlib
- Wayland: Uses xdg-desktop-portal (via dbus) or fallback to ydotool/evdev
- uinput/fallback: Not supported (kernel level)

Common hotkeys:
- Super+Alt+A: Emergency disable (kill switch)
- Super+Alt+P: Pause/Resume tracking
- Super+Alt+C: Calibrate
- Super+Alt+S: Settings
"""

import os
import logging
import threading
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable, Dict, List, Set
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QTimer, QSocketNotifier
from PySide6.QtWidgets import QApplication

logger = logging.getLogger(__name__)


class HotkeyBackend(Enum):
    """Hotkey backend types."""
    X11 = "x11"
    PORTAL = "portal"  # xdg-desktop-portal
    YDOTOL = "ydotool"
    EVDEV = "evdev"  # Direct kernel event device access
    NONE = "none"


class KeyModifier(Enum):
    """Keyboard modifiers."""
    NONE = 0
    SHIFT = auto()
    CTRL = auto()
    ALT = auto()
    SUPER = auto()  # Meta/Windows key
    META = auto()   # Same as SUPER


class KeyCode(Enum):
    """Common key codes (X11 keysyms)."""
    # Letters
    A = 0x61
    B = 0x62
    C = 0x63
    D = 0x64
    E = 0x65
    F = 0x66
    G = 0x67
    H = 0x68
    I = 0x69
    J = 0x6a
    K = 0x6b
    L = 0x6c
    M = 0x6d
    N = 0x6e
    O = 0x6f
    P = 0x70
    Q = 0x71
    R = 0x72
    S = 0x73
    T = 0x74
    U = 0x75
    V = 0x76
    W = 0x77
    X = 0x78
    Y = 0x79
    Z = 0x7a

    # Function keys
    F1 = 0xffbe
    F2 = 0xffbf
    F3 = 0xffc0
    F4 = 0xffc1
    F5 = 0xffc2
    F6 = 0xffc3
    F7 = 0xffc4
    F8 = 0xffc5
    F9 = 0xffc6
    F10 = 0xffc7
    F11 = 0xffc8
    F12 = 0xffc9

    # Special
    ESCAPE = 0xff1b
    TAB = 0xff09
    SPACE = 0x20
    RETURN = 0xff0d
    BACKSPACE = 0xff08
    DELETE = 0xffff
    INSERT = 0xff63
    HOME = 0xff50
    END = 0xff57
    PAGE_UP = 0xff55
    PAGE_DOWN = 0xff56
    UP = 0xff52
    DOWN = 0xff54
    LEFT = 0xff51
    RIGHT = 0xff53


@dataclass
class Hotkey:
    """Hotkey definition."""
    id: str
    modifiers: Set[KeyModifier]
    key: KeyCode
    callback: Callable
    description: str = ""
    enabled: bool = True

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Hotkey):
            return self.id == other.id
        return False


class HotkeyBackendBase(ABC):
    """Abstract base for hotkey backends."""

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def initialize(self) -> bool:
        pass

    @abstractmethod
    def register_hotkey(self, hotkey: Hotkey) -> bool:
        pass

    @abstractmethod
    def unregister_hotkey(self, hotkey_id: str) -> bool:
        pass

    @abstractmethod
    def unregister_all(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass


class X11HotkeyBackend(HotkeyBackendBase):
    """X11 hotkey backend using XGrabKey."""

    def __init__(self):
        self._display = None
        self._root_window = None
        self._registered_hotkeys: Dict[str, tuple] = {}  # id -> (modifiers_mask, keycode)
        self._initialized = False
        self._event_thread = None
        self._running = False

    def is_available(self) -> bool:
        if os.environ.get('WAYLAND_DISPLAY') and not os.environ.get('DISPLAY'):
            return False
        if not os.environ.get('DISPLAY'):
            return False
        try:
            import subprocess
            result = subprocess.run(
                ['xdpyinfo', '-queryExtensions'],
                capture_output=True, text=True, timeout=2
            )
            return 'XTEST' in result.stdout  # XTest usually implies XGrabKey works
        except Exception:
            return False

    def initialize(self) -> bool:
        try:
            from Xlib import display, X
            self._display = display.Display()
            self._root_window = self._display.screen().root
            self._initialized = True

            # Start event processing thread
            self._running = True
            self._event_thread = threading.Thread(target=self._event_loop, daemon=True)
            self._event_thread.start()

            logger.info("X11 hotkey backend initialized")
            return True
        except ImportError:
            logger.warning("python3-xlib not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize X11 hotkey backend: {e}")
            return False

    def _modifiers_to_mask(self, modifiers: Set[KeyModifier]) -> int:
        """Convert modifier set to X11 modifier mask."""
        from Xlib import X
        mask = 0
        mod_map = {
            KeyModifier.SHIFT: X.ShiftMask,
            KeyModifier.CTRL: X.ControlMask,
            KeyModifier.ALT: X.Mod1Mask,
            KeyModifier.SUPER: X.Mod4Mask,
            KeyModifier.META: X.Mod4Mask,
        }
        for mod in modifiers:
            if mod in mod_map:
                mask |= mod_map[mod]
        return mask

    def _keycode_to_xkeysym(self, key: KeyCode) -> int:
        """Convert KeyCode to X11 keysym."""
        return key.value

    def register_hotkey(self, hotkey: Hotkey) -> bool:
        if not self._initialized:
            return False

        try:
            from Xlib import X

            modifiers_mask = self._modifiers_to_mask(hotkey.modifiers)
            keysym = self._keycode_to_xkeysym(hotkey.key)
            keycode = self._display.keysym_to_keycode(keysym)

            if keycode == 0:
                logger.error(f"Invalid keysym for {hotkey.key}")
                return False

            # Grab the key combination
            # Also grab with NumLock, CapsLock, ScrollLock variations
            lock_masks = [0, X.LockMask, X.Mod2Mask, X.Mod2Mask | X.LockMask]

            for lock_mask in lock_masks:
                self._root_window.grab_key(
                    keycode,
                    modifiers_mask | lock_mask,
                    True,  # owner_events
                    X.GrabModeAsync,
                    X.GrabModeAsync
                )

            self._registered_hotkeys[hotkey.id] = (modifiers_mask, keycode)
            logger.info(f"Registered X11 hotkey: {hotkey.id} ({hotkey.modifiers}+{hotkey.key.name})")
            return True

        except Exception as e:
            logger.error(f"Failed to register X11 hotkey {hotkey.id}: {e}")
            return False

    def unregister_hotkey(self, hotkey_id: str) -> bool:
        if not self._initialized or hotkey_id not in self._registered_hotkeys:
            return False

        try:
            from Xlib import X

            modifiers_mask, keycode = self._registered_hotkeys[hotkey_id]

            lock_masks = [0, X.LockMask, X.Mod2Mask, X.Mod2Mask | X.LockMask]

            for lock_mask in lock_masks:
                self._root_window.ungrab_key(
                    keycode,
                    modifiers_mask | lock_mask
                )

            del self._registered_hotkeys[hotkey_id]
            self._display.sync()
            return True

        except Exception as e:
            logger.error(f"Failed to unregister X11 hotkey {hotkey_id}: {e}")
            return False

    def unregister_all(self):
        for hotkey_id in list(self._registered_hotkeys.keys()):
            self.unregister_hotkey(hotkey_id)

    def _event_loop(self):
        """Process X11 events for hotkey detection."""
        from Xlib import X

        while self._running:
            try:
                if self._display.pending_events() > 0:
                    event = self._display.next_event()

                    if event.type == X.KeyPress:
                        self._handle_key_event(event)
                else:
                    time.sleep(0.001)  # Small sleep to prevent busy loop
            except Exception as e:
                if self._running:
                    logger.error(f"X11 hotkey event loop error: {e}")
                break

    def _handle_key_event(self, event):
        """Handle X11 KeyPress event."""
        from Xlib import X

        keycode = event.detail
        state = event.state

        # Check against registered hotkeys
        for hotkey_id, (mod_mask, hk_keycode) in self._registered_hotkeys.items():
            if keycode == hk_keycode:
                # Check modifiers (ignore lock bits)
                effective_state = state & ~(X.LockMask | X.Mod2Mask)
                if effective_state == mod_mask:
                    # Found matching hotkey - trigger callback
                    # We need to call the callback somehow...
                    # This would need integration with the manager
                    logger.debug(f"Hotkey triggered: {hotkey_id}")

    def cleanup(self):
        self._running = False
        self.unregister_all()
        if self._display:
            self._display.close()
            self._display = None
        self._initialized = False


class PortalHotkeyBackend(HotkeyBackendBase):
    """xdg-desktop-portal hotkey backend for Wayland."""

    def __init__(self):
        self._initialized = False
        self._bus = None
        self._portal_proxy = None
        self._registered_hotkeys: Dict[str, str] = {}  # id -> handle

    def is_available(self) -> bool:
        if not os.environ.get('WAYLAND_DISPLAY'):
            return False
        try:
            import dbus
            bus = dbus.SessionBus()
            # Check for xdg-desktop-portal
            try:
                bus.get_object('org.freedesktop.portal.Desktop', '/org/freedesktop/portal/desktop')
                return True
            except dbus.DBusException:
                return False
        except Exception:
            return False

    def initialize(self) -> bool:
        try:
            import dbus
            self._bus = dbus.SessionBus()
            self._portal_proxy = self._bus.get_object(
                'org.freedesktop.portal.Desktop',
                '/org/freedesktop/portal/desktop'
            )
            self._initialized = True
            logger.info("xdg-desktop-portal hotkey backend initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize portal hotkey backend: {e}")
            return False

    def register_hotkey(self, hotkey: Hotkey) -> bool:
        if not self._initialized:
            return False

        # xdg-desktop-portal doesn't have a direct global hotkey API yet
        # This would need the GlobalShortcuts portal which is not widely implemented
        logger.warning("xdg-desktop-portal global shortcuts not yet implemented")
        return False

    def unregister_hotkey(self, hotkey_id: str) -> bool:
        return False

    def unregister_all(self):
        pass

    def cleanup(self):
        self._initialized = False


class EvdevHotkeyBackend(HotkeyBackendBase):
    """Linux evdev backend for global hotkeys (requires root or input group)."""

    def __init__(self):
        self._initialized = False
        self._devices: List[int] = []
        self._event_thread = None
        self._running = False
        self._hotkeys: Dict[str, Hotkey] = {}
        self._key_state: Dict[int, bool] = {}  # keycode -> pressed

    def is_available(self) -> bool:
        # Check if we can access input devices
        import glob
        devices = glob.glob('/dev/input/event*')
        if not devices:
            return False

        # Check if we can read at least one
        for dev in devices:
            if os.access(dev, os.R_OK):
                return True
        return False

    def initialize(self) -> bool:
        if not self.is_available():
            return False

        try:
            import evdev
            # Find keyboard devices
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            keyboards = [d for d in devices if d.capabilities().get(evdev.ecodes.EV_KEY)]

            if not keyboards:
                logger.warning("No keyboard devices found for evdev hotkeys")
                return False

            self._devices = [d.fd for d in keyboards]
            self._device_objects = keyboards

            # Check for key codes we need
            self._evdev = evdev
            self._ecodes = evdev.ecodes

            self._running = True
            self._event_thread = threading.Thread(target=self._event_loop, daemon=True)
            self._event_thread.start()

            self._initialized = True
            logger.info(f"evdev hotkey backend initialized with {len(keyboards)} keyboard(s)")
            return True

        except ImportError:
            logger.warning("python-evdev not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize evdev hotkey backend: {e}")
            return False

    def _modifiers_to_evdev(self, modifiers: Set[KeyModifier]) -> Set[int]:
        """Convert modifiers to evdev key codes."""
        mod_map = {
            KeyModifier.SHIFT: [self._ecodes.KEY_LEFTSHIFT, self._ecodes.KEY_RIGHTSHIFT],
            KeyModifier.CTRL: [self._ecodes.KEY_LEFTCTRL, self._ecodes.KEY_RIGHTCTRL],
            KeyModifier.ALT: [self._ecodes.KEY_LEFTALT, self._ecodes.KEY_RIGHTALT],
            KeyModifier.SUPER: [self._ecodes.KEY_LEFTMETA, self._ecodes.KEY_RIGHTMETA],
            KeyModifier.META: [self._ecodes.KEY_LEFTMETA, self._ecodes.KEY_RIGHTMETA],
        }
        result = set()
        for mod in modifiers:
            if mod in mod_map:
                result.update(mod_map[mod])
        return result

    def _keycode_to_evdev(self, key: KeyCode) -> int:
        """Convert KeyCode to evdev key code."""
        # Map common keys
        key_map = {
            KeyCode.A: self._ecodes.KEY_A,
            KeyCode.B: self._ecodes.KEY_B,
            KeyCode.C: self._ecodes.KEY_C,
            KeyCode.D: self._ecodes.KEY_D,
            KeyCode.E: self._ecodes.KEY_E,
            KeyCode.F: self._ecodes.KEY_F,
            KeyCode.G: self._ecodes.KEY_G,
            KeyCode.H: self._ecodes.KEY_H,
            KeyCode.I: self._ecodes.KEY_I,
            KeyCode.J: self._ecodes.KEY_J,
            KeyCode.K: self._ecodes.KEY_K,
            KeyCode.L: self._ecodes.KEY_L,
            KeyCode.M: self._ecodes.KEY_M,
            KeyCode.N: self._ecodes.KEY_N,
            KeyCode.O: self._ecodes.KEY_O,
            KeyCode.P: self._ecodes.KEY_P,
            KeyCode.Q: self._ecodes.KEY_Q,
            KeyCode.R: self._ecodes.KEY_R,
            KeyCode.S: self._ecodes.KEY_S,
            KeyCode.T: self._ecodes.KEY_T,
            KeyCode.U: self._ecodes.KEY_U,
            KeyCode.V: self._ecodes.KEY_V,
            KeyCode.W: self._ecodes.KEY_W,
            KeyCode.X: self._ecodes.KEY_X,
            KeyCode.Y: self._ecodes.KEY_Y,
            KeyCode.Z: self._ecodes.KEY_Z,
            KeyCode.F1: self._ecodes.KEY_F1,
            KeyCode.F2: self._ecodes.KEY_F2,
            KeyCode.F3: self._ecodes.KEY_F3,
            KeyCode.F4: self._ecodes.KEY_F4,
            KeyCode.F5: self._ecodes.KEY_F5,
            KeyCode.F6: self._ecodes.KEY_F6,
            KeyCode.F7: self._ecodes.KEY_F7,
            KeyCode.F8: self._ecodes.KEY_F8,
            KeyCode.F9: self._ecodes.KEY_F9,
            KeyCode.F10: self._ecodes.KEY_F10,
            KeyCode.F11: self._ecodes.KEY_F11,
            KeyCode.F12: self._ecodes.KEY_F12,
            KeyCode.ESCAPE: self._ecodes.KEY_ESC,
            KeyCode.TAB: self._ecodes.KEY_TAB,
            KeyCode.SPACE: self._ecodes.KEY_SPACE,
            KeyCode.RETURN: self._ecodes.KEY_ENTER,
            KeyCode.BACKSPACE: self._ecodes.KEY_BACKSPACE,
            KeyCode.DELETE: self._ecodes.KEY_DELETE,
        }
        return key_map.get(key, 0)

    def register_hotkey(self, hotkey: Hotkey) -> bool:
        if not self._initialized:
            return False

        try:
            mod_keys = self._modifiers_to_evdev(hotkey.modifiers)
            key_code = self._keycode_to_evdev(hotkey.key)

            if key_code == 0:
                logger.error(f"Unknown key code for {hotkey.key}")
                return False

            self._hotkeys[hotkey.id] = hotkey
            logger.info(f"Registered evdev hotkey: {hotkey.id} (mods={mod_keys}, key={key_code})")
            return True

        except Exception as e:
            logger.error(f"Failed to register evdev hotkey {hotkey.id}: {e}")
            return False

    def unregister_hotkey(self, hotkey_id: str) -> bool:
        if hotkey_id in self._hotkeys:
            del self._hotkeys[hotkey_id]
            return True
        return False

    def unregister_all(self):
        self._hotkeys.clear()

    def _event_loop(self):
        """Process evdev events."""
        import select

        while self._running:
            try:
                # Wait for events on any device
                r, _, _ = select.select(self._devices, [], [], 0.1)

                for fd in r:
                    # Find device object
                    dev = next((d for d in self._device_objects if d.fd == fd), None)
                    if not dev:
                        continue

                    try:
                        events = dev.read()
                        for event in events:
                            if event.type == self._ecodes.EV_KEY:
                                self._handle_key_event(event)
                    except BlockingIOError:
                        pass
                    except Exception as e:
                        logger.error(f"evdev read error: {e}")

            except Exception as e:
                if self._running:
                    logger.error(f"evdev event loop error: {e}")
                break

    def _handle_key_event(self, event):
        """Handle evdev key event."""
        keycode = event.code
        pressed = event.value == 1  # 1=press, 0=release, 2=repeat

        self._key_state[keycode] = pressed

        if not pressed:
            return  # Only trigger on press

        # Check each registered hotkey
        for hotkey_id, hotkey in self._hotkeys.items():
            if not hotkey.enabled:
                continue

            mod_keys = self._modifiers_to_evdev(hotkey.modifiers)
            key_code = self._keycode_to_evdev(hotkey.key)

            # Check if all modifiers are pressed
            mods_pressed = all(self._key_state.get(k, False) for k in mod_keys)
            key_pressed = self._key_state.get(key_code, False)

            if mods_pressed and key_pressed:
                # Trigger callback in main thread
                logger.info(f"evdev hotkey triggered: {hotkey_id}")
                try:
                    hotkey.callback()
                except Exception as e:
                    logger.error(f"Hotkey callback error: {e}")

    def cleanup(self):
        self._running = False
        self.unregister_all()
        self._initialized = False


class GlobalHotkeyManager(QObject):
    """
    Cross-platform global hotkey manager.

    Automatically selects best backend:
    1. X11 (XGrabKey) - most reliable on X11
    2. evdev - works on Wayland but needs input group access
    3. xdg-desktop-portal - Wayland standard but limited support
    """

    # Signals
    hotkey_triggered = Signal(str)  # hotkey_id

    # Predefined hotkeys for Air Mouse
    DEFAULT_HOTKEYS = {
        "emergency_disable": Hotkey(
            id="emergency_disable",
            modifiers={KeyModifier.SUPER, KeyModifier.ALT},
            key=KeyCode.A,
            callback=lambda: None,  # Set by user
            description="Emergency disable (kill switch)"
        ),
        "pause_resume": Hotkey(
            id="pause_resume",
            modifiers={KeyModifier.SUPER, KeyModifier.ALT},
            key=KeyCode.P,
            callback=lambda: None,
            description="Pause/Resume tracking"
        ),
        "calibrate": Hotkey(
            id="calibrate",
            modifiers={KeyModifier.SUPER, KeyModifier.ALT},
            key=KeyCode.C,
            callback=lambda: None,
            description="Open calibration"
        ),
        "settings": Hotkey(
            id="settings",
            modifiers={KeyModifier.SUPER, KeyModifier.ALT},
            key=KeyCode.S,
            callback=lambda: None,
            description="Open settings"
        ),
        "precision_toggle": Hotkey(
            id="precision_toggle",
            modifiers={KeyModifier.SUPER, KeyModifier.ALT},
            key=KeyCode.M,
            callback=lambda: None,
            description="Toggle precision mode"
        ),
    }

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._backend: Optional[HotkeyBackendBase] = None
        self._backend_type: HotkeyBackend = HotkeyBackend.NONE
        self._hotkeys: Dict[str, Hotkey] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._initialized = False

    def detect_best_backend(self) -> HotkeyBackend:
        """Detect the best available hotkey backend."""
        backends = [
            (HotkeyBackend.X11, X11HotkeyBackend()),
            (HotkeyBackend.EVDEV, EvdevHotkeyBackend()),
            (HotkeyBackend.PORTAL, PortalHotkeyBackend()),
        ]

        for backend_type, backend in backends:
            if backend.is_available():
                logger.info(f"Hotkey backend {backend_type.value} available")
                if backend.initialize():
                    logger.info(f"Hotkey backend {backend_type.value} initialized")
                    return backend_type
                else:
                    logger.warning(f"Hotkey backend {backend_type.value} failed to initialize")

        logger.warning("No hotkey backend available")
        return HotkeyBackend.NONE

    def initialize(self) -> bool:
        """Initialize the hotkey manager."""
        if self._initialized:
            return True

        self._backend_type = self.detect_best_backend()

        if self._backend_type == HotkeyBackend.X11:
            self._backend = X11HotkeyBackend()
        elif self._backend_type == HotkeyBackend.EVDEV:
            self._backend = EvdevHotkeyBackend()
        elif self._backend_type == HotkeyBackend.PORTAL:
            self._backend = PortalHotkeyBackend()
        else:
            logger.error("No hotkey backend available")
            return False

        if not self._backend.initialize():
            self._backend = None
            self._backend_type = HotkeyBackend.NONE
            return False

        self._initialized = True
        logger.info(f"Global hotkey manager initialized with {self._backend_type.value} backend")
        return True

    def register_hotkey(self, hotkey: Hotkey) -> bool:
        """Register a global hotkey."""
        if not self._initialized or not self._backend:
            return False

        # Store callback separately for triggering
        self._callbacks[hotkey.id] = hotkey.callback
        self._hotkeys[hotkey.id] = hotkey

        # Register with backend
        # For X11 and evdev, we need to inject the callback trigger
        return self._backend.register_hotkey(hotkey)

    def unregister_hotkey(self, hotkey_id: str) -> bool:
        """Unregister a hotkey."""
        if not self._backend:
            return False

        self._callbacks.pop(hotkey_id, None)
        self._hotkeys.pop(hotkey_id, None)
        return self._backend.unregister_hotkey(hotkey_id)

    def register_default_hotkeys(self, callbacks: Dict[str, Callable]) -> Dict[str, bool]:
        """Register default Air Mouse hotkeys with custom callbacks."""
        results = {}
        for key, hotkey in self.DEFAULT_HOTKEYS.items():
            if key in callbacks:
                hotkey.callback = callbacks[key]
            results[key] = self.register_hotkey(hotkey)
        return results

    def enable_hotkey(self, hotkey_id: str, enabled: bool = True):
        """Enable/disable a hotkey."""
        if hotkey_id in self._hotkeys:
            self._hotkeys[hotkey_id].enabled = enabled

    def get_backend_type(self) -> HotkeyBackend:
        return self._backend_type

    def is_initialized(self) -> bool:
        return self._initialized

    def cleanup(self):
        """Clean up resources."""
        if self._backend:
            self._backend.cleanup()
            self._backend = None
        self._hotkeys.clear()
        self._callbacks.clear()
        self._initialized = False
        self._backend_type = HotkeyBackend.NONE

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def create_emergency_hotkey_manager(emergency_callback: Callable) -> GlobalHotkeyManager:
    """Create a hotkey manager with just the emergency disable hotkey."""
    manager = GlobalHotkeyManager()

    def on_emergency():
        logger.critical("EMERGENCY DISABLE TRIGGERED!")
        emergency_callback()

    manager.initialize()
    manager.register_hotkey(Hotkey(
        id="emergency_disable",
        modifiers={KeyModifier.SUPER, KeyModifier.ALT},
        key=KeyCode.A,
        callback=on_emergency,
        description="Emergency disable - immediate stop"
    ))

    return manager


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)

    def on_emergency():
        print("!!! EMERGENCY DISABLE TRIGGERED !!!")
        app.quit()

    def on_pause():
        print("Pause/Resume")

    def on_calibrate():
        print("Calibrate")

    manager = GlobalHotkeyManager()

    callbacks = {
        "emergency_disable": on_emergency,
        "pause_resume": on_pause,
        "calibrate": on_calibrate,
    }

    if manager.initialize():
        results = manager.register_default_hotkeys(callbacks)
        print(f"Registered hotkeys: {results}")
        print(f"Backend: {manager.get_backend_type().value}")
        print("Press Super+Alt+A to trigger emergency disable")
        print("Press Super+Alt+P for pause/resume")
        print("Press Super+Alt+C for calibrate")
        sys.exit(app.exec())
    else:
        print("Failed to initialize hotkey manager")
        sys.exit(1)