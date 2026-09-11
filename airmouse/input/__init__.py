"""Air Mouse Input Module

Provides virtual mouse and Linux input backends.
"""

from .uinput_mouse import VirtualMouse, UInputDeviceConfig
from .linux_input import (
    LinuxInputManager,
    InputBackend,
    DesktopEnvironment,
    InputCapabilities,
    get_input_manager,
)

__all__ = [
    'VirtualMouse',
    'UInputDeviceConfig',
    'LinuxInputManager',
    'InputBackend',
    'DesktopEnvironment',
    'InputCapabilities',
    'get_input_manager',
]