"""Air Mouse UI Module

Provides GUI, system tray, hotkeys, and safety components.
"""

from .main_window import MainWindow, run_gui
from .system_tray import (
    SystemTrayManager,
    TrayBackend,
    TrayMenuItem,
    create_airmouse_tray_menu,
)
from .hotkeys import (
    GlobalHotkeyManager,
    Hotkey,
    HotkeyBackend,
    KeyModifier,
    KeyCode,
    create_emergency_hotkey_manager,
)
from .safety import (
    SafetyManager,
    SafetyConfig,
    SafetyTrigger,
    SafetyLevel,
    SafetyEvent,
    DEFAULT_SAFETY_CONFIG,
)

__all__ = [
    'MainWindow',
    'run_gui',
    'SystemTrayManager',
    'TrayBackend',
    'TrayMenuItem',
    'create_airmouse_tray_menu',
    'GlobalHotkeyManager',
    'Hotkey',
    'HotkeyBackend',
    'KeyModifier',
    'KeyCode',
    'create_emergency_hotkey_manager',
    'SafetyManager',
    'SafetyConfig',
    'SafetyTrigger',
    'SafetyLevel',
    'SafetyEvent',
    'DEFAULT_SAFETY_CONFIG',
]