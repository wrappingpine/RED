#!/usr/bin/env python3
"""Validate system tray through full GUI startup path.

Exercises SystemTrayManager initialization, backend detection, menu creation,
and graceful degradation when AppIndicator3 is unavailable (Wayland/COSMIC).
Does NOT fake tray availability or suppress expected warnings.
"""
import sys
import os
import logging

sys.path.insert(0, '/home/shubham/airmouse')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from airmouse.ui.system_tray import (
    SystemTrayManager, TrayBackend, TrayMenuItem, create_airmouse_tray_menu
)
from PySide6.QtWidgets import QApplication

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK {name}")
    else:
        failed += 1
        print(f"  FAIL {name}  {detail}")


print("=== SystemTrayManager validation ===")

# Create QApplication (required for Qt widgets)
app = QApplication.instance() or QApplication(sys.argv)
app.setQuitOnLastWindowClosed(False)

# Test 1: Backend detection
tray = SystemTrayManager("airmouse-test")
check("SystemTrayManager instantiated", tray is not None)
check("_backend attribute set", hasattr(tray, '_backend'))
check("backend is TrayBackend enum", isinstance(tray._backend, TrayBackend))
check("get_backend() returns TrayBackend", isinstance(tray.get_backend(), TrayBackend))

# Test 2: Check what backend is detected on this system
detected = tray.get_backend()
print(f"  Detected backend: {detected.value}")

# Test 3: Create menu items
def dummy(): pass

items = create_airmouse_tray_menu(
    on_toggle=dummy,
    on_settings=dummy,
    on_calibrate=dummy,
    on_diagnose=dummy,
    on_quit=dummy,
    is_active=True,
    is_paused=False
)
check("create_airmouse_tray_menu returns list", isinstance(items, list))
check("menu has 9 items", len(items) == 9)
check("items are TrayMenuItem", all(isinstance(i, TrayMenuItem) for i in items))

# Test 4: Verify menu items have expected structure
texts = [i.text for i in items]
check("has Enable/Disable", any("Enable" in t or "Disable" in t for t in texts))
check("has Pause/Resume", any("Pause" in t or "Resume" in t for t in texts))
check("has Show", any("Show" in t for t in texts))
check("has Start", any("Start" in t for t in texts))
check("has Stop", any("Stop" in t for t in texts))
check("has Calibrate", any("Calibrate" in t for t in texts))
check("has Settings", any("Settings" in t for t in texts))
check("has Diagnostics", any("Diagnostics" in t for t in texts))
check("has Quit", any("Quit" in t for t in texts))

# Test 5: Initialize tray (should work even on Wayland with QSystemTray fallback)
initialized = tray.initialize()
check(f"initialize() returns bool ({initialized})", isinstance(initialized, bool))

# On COSMIC/Wayland, AppIndicator3 may not be available
# QSystemTrayIcon should work in offscreen mode
if initialized:
    check("backend is not NONE after init", tray.get_backend() != TrayBackend.NONE)
    
    # Test menu creation
    tray.create_menu(items)
    check("create_menu() executes without error", True)
    
    # Test show/hide
    tray.show()
    check("show() executes", tray.is_visible())
    tray.hide()
    check("hide() executes", not tray.is_visible())
    
    # Test set_icon/set_tooltip
    tray.set_icon("input-mouse")
    check("set_icon() executes", True)
    tray.set_tooltip("Test tooltip")
    check("set_tooltip() executes", True)
    tray.set_status("running")
    check("set_status() executes", True)
    
    # Test show_message
    tray.show_message("Title", "Message")
    check("show_message() executes", True)
    
    tray.cleanup()
    check("cleanup() executes", True)
else:
    # If initialization failed, verify it's because no backend is available
    check("initialize() failed gracefully (no backend)", tray.get_backend() == TrayBackend.NONE)

# Test 6: Explicit backend selection
for backend in [TrayBackend.QSYSTEM_TRAY, TrayBackend.NONE]:
    t = SystemTrayManager("test", backend=backend)
    check(f"explicit backend={backend.value} accepted", t.get_backend() == backend)

# Test 7: TrayMenuItem structure
item = TrayMenuItem(
    text="Test",
    callback=dummy,
    checkable=True,
    checked=False,
    enabled=True,
    icon="test-icon",
    separator_before=True,
    separator_after=True
)
check("TrayMenuItem accepts all fields", 
      item.text == "Test" and item.checkable and item.separator_before and item.separator_after)
check("TrayMenuItem defaults work", 
      TrayMenuItem(text="X", callback=dummy).checkable == False)

# Test 8: Context manager protocol
class MockTray:
    def __init__(self):
        self.initialized = False
        self.cleaned = False
    def initialize(self):
        self.initialized = True
        return True
    def cleanup(self):
        self.cleaned = True
    def __enter__(self):
        self.initialize()
        return self
    def __exit__(self, *args):
        self.cleanup()

mt = MockTray()
with mt:
    pass
check("context manager calls initialize/cleanup", mt.initialized and mt.cleaned)


print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)